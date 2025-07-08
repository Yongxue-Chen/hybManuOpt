import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Callable, Any, Dict, List, Optional, Union, Tuple
import time
from model import Model, OperationTime, subModelPara
from optCons import consHMLN
from functools import partial

class GurobiOptimizationFramework:
    """
    基于Gurobi原生API的优化框架
    使用Gurobi内置回调函数处理延迟约束，支持初始解
    """
    
    def __init__(self, targetModel: Model, gurobi_params: Optional[Dict] = None, threshold: float = 1e-6):
        """
        初始化优化框架
        
        Args:
            gurobi_params: Gurobi参数字典
            threshold: 变量阈值，低于此值需要重新构造问题
        """
        self.gurobi_params = gurobi_params or {}
        self.threshold = threshold
        self.model = None
        self.targetModel = targetModel
        
        # 回调函数相关
        self.constraint_check_functions: List[Callable] = []
        
        # 优化历史记录
        self.optimization_history = []
        self.callback_count = 0
        self.lazy_constraints_added = 0
        
        # 初始解 - 修改为numpy数组
        self.initial_solution: Optional[np.ndarray] = None
        
    def set_initial_solution(self, initial_solution: np.ndarray):
        """
        设置初始解
        
        Args:
            initial_solution: 初始解向量，numpy数组格式
        """
        self.initial_solution = initial_solution
    
    def set_constraint_check_functions(self, functions: List[Callable]):
        """
        设置约束检查函数列表
        
        Args:
            functions: 约束检查函数列表，每个函数应该返回(is_satisfied, lazy_constraints)
        """
        self.constraint_check_functions = functions
    
    def create_model_from_construction_function(self, current_solution: Optional[np.ndarray], **kwargs) -> gp.Model:
        """
        创建Gurobi模型，包含完整的模型构建逻辑
        
        Args:
            current_solution: 当前解（首次构建时为None或初始解）
            nx, ny, nz: 模型维度
            tEnd: 结束时间
            **kwargs: 其他参数
            
        Returns:
            Gurobi模型对象
        """
        # 参数验证 - 添加类型检查防止"Never"值
        if str(self.targetModel.nx) == "Never" or str(self.targetModel.ny) == "Never" or str(self.targetModel.nz) == "Never" or str(self.targetModel.tEnd) == "Never":
            raise ValueError(f"参数包含无效值: nx={self.targetModel.nx}, ny={self.targetModel.ny}, nz={self.targetModel.nz}, tEnd={self.targetModel.tEnd}")
        
        # 确保参数为数值类型
        nx, ny, nz, tEnd = int(self.targetModel.nx), int(self.targetModel.ny), int(self.targetModel.nz), float(self.targetModel.tEnd)
        
        # 确定使用的解
        solution_to_use = current_solution
        if solution_to_use is None:
            if self.initial_solution is not None:
                solution_to_use = self.initial_solution
            else:
                # 如果没有提供任何解，创建零向量
                solution_to_use = np.zeros(2 * nx * ny * nz)
        
        n = len(solution_to_use)  # 获取变量维度
        
        # 定义变量类型
        try:
            vtype = GRB.CONTINUOUS
        except NameError:
            vtype = 'C'  # Gurobi的字符串等价值
            
        # 构建当前操作时间
        currentOptTime = OperationTime(nx, ny, nz)
        currentOptTime.from_optimization_variables(solution_to_use)

        # 创建Gurobi模型
        model = gp.Model("optimization_model")
        
        # 设置参数
        for param, value in self.gurobi_params.items():
            model.setParam(param, value)
        
        # 创建变量并存储在模型中
        model._variables = {}
        
        # 直接通过 addVars 添加自变量 'Dt'
        dt_vars = model.addVars(n, lb=0, ub=tEnd + 1, vtype=vtype, name="Dt")
        model._variables['Dt'] = dt_vars
        
        # 设置初始值
        if solution_to_use is not None:
            # 将 numpy 数组转换为列表以设置初始值
            model.setAttr("Start", list(dt_vars.values()), solution_to_use.tolist())
        
        # 更新模型以添加变量
        model.update()
        
        # 设置目标函数
        w1 = 1.0
        objective = gp.LinExpr()
        for i in range(nx*ny*nz):
            if solution_to_use[2*i] + solution_to_use[2*i+1] < tEnd:
                objective += w1 * dt_vars[2*i+1]
        model.setObjective(objective, GRB.MINIMIZE)
        
        # 添加约束
        subModel = subModelPara(posStart=(0, 0, 0), modelSize=(nx, ny, nz))
        satisfied, AElement, ACol, bElement = consHMLN(currentOptTime, subModel, self.targetModel, tEnd)
        if not satisfied:
            raise ValueError("Constraint not satisfied")
        
        # 创建线性不等式约束
        constraints = create_linear_constraints_from_sparse_format(
            AElement, ACol, bElement, model._variables, 'Dt'
        )
        
        # 添加约束到模型
        for i, constraint in enumerate(constraints):
            if constraint is not None:
                model.addConstr(constraint, name=f"constraint_{i}")
        
        return model
    
    def lazy_constraint_callback(self, model, where):
        """
        Gurobi延迟约束回调函数
        
        Args:
            model: Gurobi模型对象
            where: 回调触发位置
        """
        if where == GRB.Callback.MIPSOL:
            self.callback_count += 1
            
            # 获取当前解 - 修改为构造numpy数组
            current_solution = None
            for var_name, var_obj in model._variables.items():
                if hasattr(var_obj, 'select'):  # 多维变量
                    # 假设主要变量是多维数组，构造numpy向量
                    if var_name == 'Dt':  # 假设主要变量名为 'Dt'
                        solution_values = []
                        for key in sorted(var_obj.keys()):
                            solution_values.append(model.cbGetSolution(var_obj[key]))
                        current_solution = np.array(solution_values)
                        break
                else:  # 单个变量
                    # 如果是单个变量，创建包含单个元素的数组
                    if var_name == 'Dt':
                        current_solution = np.array([model.cbGetSolution(var_obj)])
                        break
            
            # 检查约束并添加延迟约束
            constraints_added_this_callback = 0
            for check_function in self.constraint_check_functions:
                try:
                    # 调用用户定义的约束检查函数
                    is_satisfied, lazy_constraints = check_function(
                        model._variables, current_solution
                    )
                    
                    if not is_satisfied and lazy_constraints:
                        # 添加延迟约束
                        for constraint_expr in lazy_constraints:
                            if constraint_expr is not None:
                                model.cbLazy(constraint_expr)
                                self.lazy_constraints_added += 1
                                constraints_added_this_callback += 1
                        
                except Exception as e:
                    print(f"约束检查函数执行出错: {e}")
                    continue
            
            if constraints_added_this_callback > 0:
                print(f"回调 {self.callback_count}: 添加了 {constraints_added_this_callback} 个延迟约束")
    
    def check_threshold_condition(self, solution: np.ndarray) -> bool:
        """
        检查是否有变量值低于阈值
        
        Args:
            solution: 当前解，numpy数组
            
        Returns:
            是否需要重新构造模型
        """
        # 检查numpy数组中的所有元素
        for value in solution:
            if isinstance(value, (int, float)) and 0 < value < self.threshold:
                return True
        return False
    
    def optimize(self, max_reconstructions: int = 10, **kwargs) -> Dict[str, Any]:
        """
        主优化函数
        
        Args:
            max_reconstructions: 最大重构次数
            **kwargs: 其他参数
            
        Returns:
            最终优化结果
        """
        reconstruction_count = 0
        current_solution = self.initial_solution  # 现在是numpy数组
        
        while reconstruction_count <= max_reconstructions:
            print(f"开始第 {reconstruction_count + 1} 次建模和优化...")
            
            # 重置回调计数器
            self.callback_count = 0
            self.lazy_constraints_added = 0
            
            try:
                # 创建模型
                model = self.create_model_from_construction_function(
                    current_solution=current_solution, **kwargs)
                
                # 设置延迟约束回调
                model.setParam('LazyConstraints', 1)
                
                # 开始优化
                start_time = time.time()
                model.optimize(self.lazy_constraint_callback)
                solve_time = time.time() - start_time
                
                # 处理求解结果
                solution_info = {
                    'status': model.status,
                    'solve_time': solve_time,
                    'objective_value': None,
                    'variables': None,  # 修改为存储numpy数组
                    'success': False,
                    'callback_count': self.callback_count,
                    'lazy_constraints_added': self.lazy_constraints_added,
                    'reconstruction_count': reconstruction_count
                }
                
                if model.status == GRB.OPTIMAL:
                    solution_info['success'] = True
                    solution_info['objective_value'] = model.objVal
                    
                    # 提取变量值 - 修改为构造numpy数组
                    for var_name, var_obj in model._variables.items():
                        if hasattr(var_obj, 'select'):  # 多维变量
                            if var_name == 'Dt':  # 假设主要变量名为 'Dt'
                                solution_values = []
                                for key in sorted(var_obj.keys()):
                                    solution_values.append(var_obj[key].x)
                                current_solution = np.array(solution_values)
                                solution_info['variables'] = current_solution
                                break
                        else:  # 单个变量
                            if var_name == 'Dt':
                                current_solution = np.array([var_obj.x])
                                solution_info['variables'] = current_solution
                                break
                    
                    # 记录优化历史
                    self.optimization_history.append(solution_info.copy())
                    
                    print(f"优化完成! 目标值: {model.objVal}")
                    print(f"回调次数: {self.callback_count}, 添加延迟约束: {self.lazy_constraints_added}")
                    
                    # 检查是否需要重新构造模型
                    if current_solution is not None and self.check_threshold_condition(current_solution):
                        print("检测到变量值低于阈值，需要重新构造模型...")
                        reconstruction_count += 1
                        continue
                    else:
                        # 优化成功完成
                        break
                        
                else:
                    print(f"优化失败，状态: {model.status}")
                    solution_info['success'] = False
                    break
                    
            except Exception as e:
                print(f"优化过程中出现错误: {e}")
                solution_info = {
                    'success': False,
                    'error': str(e),
                    'reconstruction_count': reconstruction_count
                }
                break
            
            finally:
                # 清理模型
                if 'model' in locals():
                    model.dispose()
        
        if reconstruction_count > max_reconstructions:
            print(f"达到最大重构次数 {max_reconstructions}，优化结束")
        
        return solution_info
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        获取优化过程摘要
        
        Returns:
            优化摘要信息
        """
        if not self.optimization_history:
            return {'message': '尚未进行优化'}
        
        summary = {
            'total_reconstructions': len(self.optimization_history),
            'total_time': sum(record['solve_time'] for record in self.optimization_history),
            'total_callbacks': sum(record['callback_count'] for record in self.optimization_history),
            'total_lazy_constraints': sum(record['lazy_constraints_added'] for record in self.optimization_history),
            'objective_progression': [record['objective_value'] 
                                    for record in self.optimization_history 
                                    if record['success']],
            'success_rate': sum(1 for record in self.optimization_history 
                              if record['success']) / len(self.optimization_history)
        }
        
        return summary


# 使用示例和辅助函数

def create_linear_constraints_from_sparse_format(AElement, ACol, bElement, variables_dict, variable_name='Dt'):
    """
    将稀疏矩阵格式的线性不等式约束转换为 Gurobi 约束
    
    Args:
        AElement: 包含向量 a 中不等于0的元素的值
        ACol: 包含向量 a 中不等于0的元素的位置索引
        bElement: 包含对应的 b 的值
        variables_dict: Gurobi 变量字典
        variable_name: 变量名称（默认为 'Dt'）
        
    Returns:
        constraints: 可以直接添加到 Gurobi 模型的约束列表
    """
    constraints = []
    
    if len(AElement) != len(ACol):
        raise ValueError("AElement 和 ACol 的长度必须相同")
    
    constraints = []
    num_constraints = len(bElement)
    
    for i in range(num_constraints):
        elementA=AElement[i]
        elementCol=ACol[i]
        elementB=bElement[i]

        constraint_expr = gp.LinExpr()
        for j in range(len(elementA)):
            constraint_expr += elementA[j] * variables_dict[variable_name][elementCol[j]]
        
        # 添加约束 a'*Dt <= b
        constraints.append(constraint_expr <= elementB)
    
    return constraints

def stability_check_function(variables, solution, nx, ny, nz, tEnd):
    """
    示例约束检查函数（用于延迟约束）
    
    Args:
        variables: 变量对象字典
        solution: 当前解，numpy数组
        nx, ny, nz: 模型维度
        tEnd: 结束时间
        
    Returns:
        (is_satisfied, lazy_constraints): 是否满足约束，以及需要添加的延迟约束
    """

    optTime=OperationTime(nx, ny, nz)
    optTime.from_optimization_variables(solution)

    # 对optTime.time_matrix的第二列中的元素按照从小到大排序，获取元素位置的序列
    sorted_idx=np.argsort(optTime.time_matrix[:,1])

    for voxIdx in sorted_idx:
        if optTime.time_matrix[voxIdx,1]>=tEnd:
            break

        

        




    # TODO: 在这里实现您的约束检查逻辑
    
    is_satisfied = True
    lazy_constraints = []
    
    # 示例约束检查逻辑：
    # if solution[0] + solution[1] > 10:
    #     is_satisfied = False
    #     lazy_constraints.append(variables['Dt'][0] + variables['Dt'][1] <= 10)
    
    return is_satisfied, lazy_constraints


# 使用示例
if __name__ == "__main__":
    from functools import partial

    # 目标函数和初始解（需要外部提供）
    nx, ny, nz = 5, 5, 5
    tEnd = 100
    targetModel = Model(nx, ny, nz, tEnd)
    initial_solution = np.zeros(2 * nx * ny * nz)
    
    # 创建优化框架实例
    optimizer = GurobiOptimizationFramework(
        targetModel=targetModel,
        gurobi_params={
            'MIPGap': 0.01,
            'TimeLimit': 300,
            'OutputFlag': 1
        },
        threshold=1e-6
    )
    
    # 设置初始解
    optimizer.set_initial_solution(initial_solution)
    
    # 使用 functools.partial 包装需要额外参数的约束检查函数
    stability_check_with_params = partial(stability_check_function, 
                                          nx=nx, ny=ny, nz=nz, tEnd=tEnd)
    
    # 设置约束检查函数
    optimizer.set_constraint_check_functions([
        stability_check_with_params,
        # 可以添加更多检查函数
    ])
    
    # 执行优化（现在直接传递参数）
    result = optimizer.optimize(
        max_reconstructions=5
    )
    
    # 打印结果
    print("优化结果:", result)
    print("优化摘要:", optimizer.get_optimization_summary())
