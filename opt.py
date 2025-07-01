import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Callable, Any, Dict, List, Optional, Union, Tuple
import time

class GurobiOptimizationFramework:
    """
    基于Gurobi原生API的优化框架
    使用Gurobi内置回调函数处理延迟约束，支持初始解
    """
    
    def __init__(self, target_model, gurobi_params: Optional[Dict] = None, threshold: float = 1e-6):
        """
        初始化优化框架
        
        Args:
            gurobi_params: Gurobi参数字典
            threshold: 变量阈值，低于此值需要重新构造问题
        """
        self.target_model = target_model

        self.gurobi_params = gurobi_params or {}
        self.threshold = threshold
        self.model = None
        
        # 回调函数相关
        self.constraint_check_functions: List[Callable] = []
        self.model_construction_function: Optional[Callable] = None
        
        # 优化历史记录
        self.optimization_history = []
        self.callback_count = 0
        self.lazy_constraints_added = 0
        
        # 初始解
        self.initial_solution: Optional[Dict[str, Any]] = None
        
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
    
    def set_model_construction_function(self, function: Callable):
        """
        设置统一的模型构建函数（用于初始构建和重新构建）
        
        Args:
            function: 模型构建函数，接受当前解和其他参数，返回(variables, objective_expr, constraints)
        """
        self.model_construction_function = function
    
    def create_model_from_construction_function(self, current_solution: Optional[Dict[str, Any]] = None,
                                              **kwargs) -> gp.Model:
        """
        使用统一的构建函数创建Gurobi模型
        
        Args:
            current_solution: 当前解（首次构建时为None或初始解）
            **kwargs: 传递给构建函数的其他参数
            
        Returns:
            Gurobi模型对象
        """
        if self.model_construction_function is None:
            raise ValueError("未设置模型构建函数，请先调用set_model_construction_function")
        
        # 调用用户定义的构建函数
        variables_def, objective_expr, constraints = self.model_construction_function(
            current_solution, **kwargs
        )
        
        # 创建Gurobi模型
        model = gp.Model("optimization_model")
        
        # 设置参数
        for param, value in self.gurobi_params.items():
            model.setParam(param, value)
        
        # 创建变量并存储在模型中
        model._variables = {}
        model._variable_mapping = {}  # 用于快速查找变量
        
        for var_name, var_config in variables_def.items():
            if len(var_config) == 4:
                lb, ub, vtype, shape = var_config
                initial_values = None
            elif len(var_config) == 5:
                lb, ub, vtype, shape, initial_values = var_config
            else:
                raise ValueError(f"变量 {var_name} 配置格式错误")
            
            if shape is None:
                # 单个变量
                var = model.addVar(lb=lb, ub=ub, vtype=vtype, name=var_name)
                model._variables[var_name] = var
                model._variable_mapping[var_name] = var
                
                # 设置初始值
                if initial_values is not None:
                    var.start = initial_values
                elif current_solution and var_name in current_solution:
                    var.start = current_solution[var_name]
                elif self.initial_solution and var_name in self.initial_solution:
                    var.start = self.initial_solution[var_name]
                    
            else:
                # 多维变量数组
                if isinstance(shape, int):
                    shape = (shape,)
                var_array = model.addVars(*shape, lb=lb, ub=ub, vtype=vtype, name=var_name)
                model._variables[var_name] = var_array
                
                # 设置初始值
                for key in var_array.keys():
                    model._variable_mapping[f"{var_name}_{key}"] = var_array[key]
                    
                    if initial_values is not None:
                        if isinstance(initial_values, dict) and key in initial_values:
                            var_array[key].start = initial_values[key]
                        elif hasattr(initial_values, '__getitem__'):
                            try:
                                var_array[key].start = initial_values[key]
                            except:
                                pass
                    elif current_solution and var_name in current_solution:
                        if isinstance(current_solution[var_name], dict) and key in current_solution[var_name]:
                            var_array[key].start = current_solution[var_name][key]
                    elif self.initial_solution and var_name in self.initial_solution:
                        if isinstance(self.initial_solution[var_name], dict) and key in self.initial_solution[var_name]:
                            var_array[key].start = self.initial_solution[var_name][key]
        
        # 更新模型以添加变量
        model.update()
        
        # 设置目标函数
        if callable(objective_expr):
            obj_expr = objective_expr(model._variables)
        else:
            obj_expr = objective_expr
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # 添加约束
        for i, constraint in enumerate(constraints):
            if callable(constraint):
                constraint_expr = constraint(model._variables)
            else:
                constraint_expr = constraint
                
            if constraint_expr is not None:
                model.addConstr(constraint_expr, name=f"constraint_{i}")
        
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
            
            # 获取当前解
            current_solution = {}
            for var_name, var_obj in model._variables.items():
                if hasattr(var_obj, 'select'):  # 多维变量
                    current_solution[var_name] = {
                        key: model.cbGetSolution(var_obj[key]) 
                        for key in var_obj.keys()
                    }
                else:  # 单个变量
                    current_solution[var_name] = model.cbGetSolution(var_obj)
            
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
    
    def check_threshold_condition(self, solution: Dict[str, Any]) -> bool:
        """
        检查是否有变量值低于阈值
        
        Args:
            solution: 当前解
            
        Returns:
            是否需要重新构造模型
        """
        for var_name, var_value in solution.items():
            if isinstance(var_value, dict):
                # 处理多维变量
                for index, value in var_value.items():
                    if isinstance(value, (int, float)) and 0 < value < self.threshold:
                        return True
            else:
                # 处理单个变量
                if isinstance(var_value, (int, float)) and 0 < var_value < self.threshold:
                    return True
        
        return False
    
    def optimize(self, construction_kwargs: Optional[Dict] = None,
                max_reconstructions: int = 10) -> Dict[str, Any]:
        """
        主优化函数
        
        Args:
            construction_kwargs: 传递给模型构建函数的参数
            max_reconstructions: 最大重构次数
            
        Returns:
            最终优化结果
        """
        if construction_kwargs is None:
            construction_kwargs = {}
            
        reconstruction_count = 0
        current_solution = self.initial_solution
        
        while reconstruction_count <= max_reconstructions:
            print(f"开始第 {reconstruction_count + 1} 次建模和优化...")
            
            # 重置回调计数器
            self.callback_count = 0
            self.lazy_constraints_added = 0
            
            try:
                # 使用统一的构建函数创建模型
                model = self.create_model_from_construction_function(
                    current_solution=current_solution,
                    **construction_kwargs
                )
                
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
                    'variables': {},
                    'success': False,
                    'callback_count': self.callback_count,
                    'lazy_constraints_added': self.lazy_constraints_added,
                    'reconstruction_count': reconstruction_count
                }
                
                if model.status == GRB.OPTIMAL:
                    solution_info['success'] = True
                    solution_info['objective_value'] = model.objVal
                    
                    # 提取变量值
                    current_solution = {}
                    for var_name, var_obj in model._variables.items():
                        if hasattr(var_obj, 'select'):  # 多维变量
                            current_solution[var_name] = {
                                key: var_obj[key].x for key in var_obj.keys()
                            }
                            solution_info['variables'][var_name] = current_solution[var_name]
                        else:  # 单个变量
                            current_solution[var_name] = var_obj.x
                            solution_info['variables'][var_name] = current_solution[var_name]
                    
                    # 记录优化历史
                    self.optimization_history.append(solution_info.copy())
                    
                    print(f"优化完成! 目标值: {model.objVal}")
                    print(f"回调次数: {self.callback_count}, 添加延迟约束: {self.lazy_constraints_added}")
                    
                    # 检查是否需要重新构造模型
                    if self.check_threshold_condition(current_solution):
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

def model_construction_function(current_solution, tEnd, problem_params):
    """
    Args:
        current_solution: 当前解
        problem_params: 问题参数
        
    Returns:
        (variables_def, objective_expr, constraints): 变量定义、目标函数、约束列表
    """

    # 从初始解获取变量维度
    if current_solution is None:
        raise ValueError("必须提供初始解来定义变量维度")
    
    n = len(current_solution)  # 获取变量维度
    # 定义n维连续变量向量
    variables_def = {
        'Dt': (0, tEnd+1, GRB.CONTINUOUS, (n,), current_solution),  # n维连续变量向量，使用初始解
    }
    
    # 目标函数定义
    def objective_expr(variables):
        # TODO: 定义您的目标函数
        # 示例：return variables['x'] + 2 * variables['y']
        return variables['x'] + variables['y']
    
    # 约束定义
    def constraint1(variables):
        return variables['x'] + variables['y'] <= 8
    
    def constraint2(variables):
        return variables['x'] - variables['y'] >= 0
    
    constraints = [constraint1, constraint2]
    
    # 根据当前解可能需要添加额外约束
    if current_solution is not None:
        # 例如：添加基于当前解的约束
        # def additional_constraint(variables):
        #     return variables['x'] >= current_solution['x'] * 0.8
        # constraints.append(additional_constraint)
        pass
    
    return variables_def, objective_expr, constraints

def example_constraint_check_function(variables, solution):
    """
    示例约束检查函数（用于延迟约束）
    
    Args:
        variables: 变量对象字典
        solution: 当前解
        
    Returns:
        (is_satisfied, lazy_constraints): 是否满足约束，以及需要添加的延迟约束
    """
    # TODO: 在这里实现您的约束检查逻辑
    
    is_satisfied = True
    lazy_constraints = []
    
    # 示例约束检查逻辑：
    # if solution['x'] + solution['y'] > 10:
    #     is_satisfied = False
    #     lazy_constraints.append(variables['x'] + variables['y'] <= 10)
    
    return is_satisfied, lazy_constraints


# 使用示例
if __name__ == "__main__":
    # 创建优化框架实例
    optimizer = GurobiOptimizationFramework(
        gurobi_params={
            'MIPGap': 0.01,
            'TimeLimit': 300,
            'OutputFlag': 1
        },
        threshold=1e-6
    )
    
    # 设置初始解（可选）
    # initial_solution = 
    optimizer.set_initial_solution(initial_solution)
    
    # 设置约束检查函数
    optimizer.set_constraint_check_functions([
        example_constraint_check_function,
        # 可以添加更多检查函数
    ])
    
    # 设置统一的模型构建函数
    optimizer.set_model_construction_function(
        example_model_construction_function
    )
    
    # 定义传递给构建函数的参数
    construction_kwargs = {
        'problem_params': {
            'param1': 10,
            'param2': 'some_value'
        }
    }
    
    # 执行优化
    result = optimizer.optimize(
        construction_kwargs=construction_kwargs,
        max_reconstructions=5
    )
    
    # 打印结果
    print("优化结果:", result)
    print("优化摘要:", optimizer.get_optimization_summary())
