import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Callable, Any, Dict, List, Optional, Union, Tuple
import time
import inspect
from model import Model, OperationTime, subModelPara
from optCons import consHMLN
from functools import partial
from stabilityCheck import getModelAtSM, checkStabilityAtSM

class GurobiOptimizationFramework:
    """
    An optimization framework based on the Gurobi native API.
    It uses Gurobi's built-in callback functions to handle lazy constraints and supports initial solutions.
    """
    
    def __init__(self, targetModel: Model, subModel: subModelPara, gurobi_params: Optional[Dict] = None, threshold: float = 1e-6):
        """
        Initializes the optimization framework.
        
        Args:
            gurobi_params: A dictionary of Gurobi parameters.
            threshold: Variable threshold; if a value falls below this, the problem needs to be reconstructed.
        """
        self.gurobi_params = gurobi_params or {}
        self.threshold = threshold
        self.model = None
        self.targetModel = targetModel
        self.subModel = subModel
        
        # Callback related
        self.constraint_check_functions: List[Callable] = []
        self.constraint_check_function_templates: List[Tuple[Callable, Dict]] = []
        
        # Optimization history
        self.optimization_history = []
        self.callback_count = 0
        self.lazy_constraints_added = 0
        
        # Initial solution - modified to a numpy array
        self.initial_solution: Optional[np.ndarray] = None
        self.last_solution: Optional[np.ndarray] = None # To store the solution from the previous iteration
        
    def set_initial_solution(self, initial_solution: np.ndarray):
        """
        Sets the initial solution.
        
        Args:
            initial_solution: The initial solution vector as a numpy array.
        """
        self.initial_solution = initial_solution
    
    def set_constraint_check_function_templates(self, functions: List[Tuple[Callable, Dict]]):
        """
        Sets the templates for constraint checking functions.
        Each template is a tuple of (function, kwargs).
        
        Args:
            functions: A list of tuples, where each tuple contains a function and a dictionary of its keyword arguments.
        """
        self.constraint_check_function_templates = functions
    
    def set_constraint_check_functions(self, functions: List[Callable]):
        """
        Sets the list of constraint checking functions.
        
        Args:
            functions: A list of constraint checking functions. Each function should return (is_satisfied, lazy_constraints).
        """
        self.constraint_check_functions = functions
    
    def create_model_from_construction_function(self, current_solution: Optional[np.ndarray], **kwargs) -> gp.Model:
        """
        Creates a Gurobi model, containing the complete model construction logic.
        
        Args:
            current_solution: The current solution (None or initial solution for the first construction).
            **kwargs: Other parameters.
            
        Returns:
            A Gurobi model object.
        """

        # Ensure parameters are of numeric types
        nx, ny, nz, tEnd = int(self.targetModel.nx), int(self.targetModel.ny), int(self.targetModel.nz), float(self.targetModel.tEnd)
        
        # Determine which solution to use
        solution_to_use = current_solution
        if solution_to_use is None:
            if self.initial_solution is not None:
                solution_to_use = self.initial_solution
            else:
                # If no solution is provided, create a zero vector
                solution_to_use = np.zeros(2 * nx * ny * nz)
        
        n = len(solution_to_use)  # Get variable dimension
            
        # Construct current operation time
        currentOptTime = OperationTime(nx, ny, nz)
        currentOptTime.from_optimization_variables(solution_to_use)

        # Create Gurobi model
        model = gp.Model("optimization_model")
        
        # Set parameters
        for param, value in self.gurobi_params.items():
            model.setParam(param, value)
        
        # Create variables and store them in the model
        model._variables = {}
        
        # Add independent variables 'Dt' via addVars
        dt_vars = model.addVars(n, lb=0, ub=tEnd + 2, vtype=GRB.CONTINUOUS, name="Dt")
        model._variables['Dt'] = dt_vars
        
        # Set initial values
        if solution_to_use is not None:
            # Convert numpy array to list to set initial values
            model.setAttr("Start", list(dt_vars.values()), solution_to_use.tolist())
        
        # Update model to add variables
        model.update()
        
        # Set objective function
        w1 = 1.0
        objective = gp.LinExpr()
        for i in range(nx*ny*nz):
            if solution_to_use[2*i] + solution_to_use[2*i+1] < tEnd:
                objective += w1 * dt_vars[2*i+1]
        model.setObjective(objective, GRB.MINIMIZE)
        
        # Add constraints
        satisfied, AElement, ACol, bElement = consHMLN(currentOptTime, self.subModel, self.targetModel, tEnd)
        if not satisfied:
            raise ValueError("Constraint not satisfied")
        
        # Create linear inequality constraints from sparse format
        constraints = create_linear_constraints_from_sparse_format(
            AElement, ACol, bElement, model._variables, 'Dt'
        )
        
        # Add constraints to the model
        for i, constraint in enumerate(constraints):
            if constraint is not None:
                model.addConstr(constraint, name=f"constraint_{i}")
        
        return model
    
    def lazy_constraint_callback(self, model, where):
        """
        Gurobi lazy constraint callback function.
        
        Args:
            model: Gurobi model object.
            where: The location where the callback was triggered.
        """
        if where == GRB.Callback.MIPSOL:
            self.callback_count += 1
            
            # Get the current solution from the callback and construct a Numpy array
            dt_vars = model._variables['Dt']
            
            # Sort the keys to ensure a consistent order of values in the solution vector
            solution_values = [
                model.cbGetSolution(dt_vars[key]) for key in sorted(dt_vars.keys())
            ]
            current_solution = np.array(solution_values)
            
            # Check constraints and add lazy constraints
            constraints_added_this_callback = 0
            for check_function in self.constraint_check_functions:
                try:
                    # Call user-defined constraint checking function
                    # Inspect the function signature to pass arguments correctly
                    sig = inspect.signature(check_function)
                    if 'lastSolution' in sig.parameters:
                        is_satisfied, lazy_constraints = check_function(
                            model._variables, current_solution, lastSolution=self.last_solution
                        )
                    else:
                        is_satisfied, lazy_constraints = check_function(
                            model._variables, current_solution
                        )
                    
                    if not is_satisfied and lazy_constraints:
                        # Add lazy constraints
                        for constraint_expr in lazy_constraints:
                            if constraint_expr is not None:
                                model.cbLazy(constraint_expr)
                                self.lazy_constraints_added += 1
                                constraints_added_this_callback += 1
                        
                except Exception as e:
                    print(f"Error executing constraint check function: {e}")
                    continue
            
            if constraints_added_this_callback > 0:
                print(f"Callback {self.callback_count}: Added {constraints_added_this_callback} lazy constraints")
    
    def check_threshold_condition(self, solution: np.ndarray) -> bool:
        """
        Checks if any variable value is below the threshold.
        
        Args:
            solution: The current solution as a numpy array.
            
        Returns:
            Whether the model needs to be reconstructed.
        """

        # check if the number of elements in solution is even
        if len(solution) % 2 != 0:
            raise ValueError("The number of elements in solution must be even")
        
        # calculate n value
        n = len(solution) // 2

        flag=False

        for i in range(n):
            if solution[2*i]<self.targetModel.tEnd and solution[2*i+1]<self.targetModel.tEnd:
                solution[2*i]=self.targetModel.tEnd+1
                solution[2*i+1]=1
                flag=True
        
        return flag
    
    def _update_solution_for_next_iteration(self, solution: np.ndarray) -> bool:
        """
        Processes the solution to update voxels that have completed their operations
        and checks if any progress was made.
        
        Args:
            solution: The current solution as a numpy array.
            
        Returns:
            True if the solution was modified, False otherwise.
        """
        if len(solution) % 2 != 0:
            raise ValueError("The number of elements in solution must be even")
        
        n = len(solution) // 2
        progress_made = False

        for i in range(n):
            if solution[2*i] < self.targetModel.tEnd and solution[2*i+1] < self.targetModel.tEnd:
                solution[2*i] = self.targetModel.tEnd + 1
                solution[2*i+1] = 1
                progress_made = True
        
        return progress_made

    def optimize(self, max_reconstructions: int = 10, convergence_tolerance: float = 1e-4, **kwargs) -> Dict[str, Any]:
        """
        Main optimization function.
        
        Args:
            max_reconstructions: Maximum number of reconstruction iterations.
            convergence_tolerance: The tolerance for solution change to determine convergence.
            **kwargs: Other parameters.
            
        Returns:
            The final optimization result.
        """

        reconstruction_count = 0
        current_solution = self.initial_solution  # Now a numpy array
        solution_info = {}
        
        while reconstruction_count < max_reconstructions:
            print(f"Starting modeling and optimization cycle {reconstruction_count + 1}...")
            
            # Reset callback counters
            self.callback_count = 0
            self.lazy_constraints_added = 0
            
            try:
                # Store current solution for comparison and for the callback
                self.last_solution = current_solution.copy() if current_solution is not None else None

                # Dynamically create constraint check functions based on the current state
                self.constraint_check_functions = []
                for func, kwargs in self.constraint_check_function_templates:
                    current_kwargs = kwargs.copy()
                    current_kwargs['lastSolution'] = self.last_solution
                    self.constraint_check_functions.append(
                        partial(func, **current_kwargs)
                    )

                # Create model
                model = self.create_model_from_construction_function(
                    current_solution=current_solution, **kwargs)
                
                # Set lazy constraint callback
                model.setParam('LazyConstraints', 1)
                
                # Start optimization
                start_time = time.time()
                model.optimize(self.lazy_constraint_callback)
                solve_time = time.time() - start_time
                
                # Process solution results
                solution_info = {
                    'status': model.status,
                    'solve_time': solve_time,
                    'objective_value': None,
                    'variables': None,
                    'success': False,
                    'callback_count': self.callback_count,
                    'lazy_constraints_added': self.lazy_constraints_added,
                    'reconstruction_count': reconstruction_count
                }
                
                if model.status == GRB.OPTIMAL:
                    solution_info['success'] = True
                    solution_info['objective_value'] = model.objVal
                    
                    # Extract variable values into a numpy array
                    for var_name, var_obj in model._variables.items():
                        if var_name == 'Dt':
                            solution_values = [var_obj[key].x for key in sorted(var_obj.keys())]
                            current_solution = np.array(solution_values)
                            solution_info['variables'] = current_solution
                            break
                    
                    self.optimization_history.append(solution_info.copy())
                    
                    print(f"Optimization finished! Objective value: {model.objVal}")
                    print(f"Callbacks: {self.callback_count}, Lazy constraints added: {self.lazy_constraints_added}")

                    # Update solution for the next iteration. If it's updated, force another cycle.
                    solution_updated = False
                    if current_solution is not None:
                        solution_updated = self._update_solution_for_next_iteration(current_solution)

                    if solution_updated:
                        print("Solution updated, proceeding to next optimization cycle.")
                    else:
                        # Only check for convergence if the solution wasn't manually updated.
                        if self.last_solution is not None and np.linalg.norm(current_solution - self.last_solution) < convergence_tolerance:
                            print(f"Convergence reached after {reconstruction_count + 1} reconstructions.")
                            break
                    
                    reconstruction_count += 1
                        
                else:
                    print(f"Optimization failed, status: {model.status}")
                    solution_info['success'] = False
                    if self.last_solution is not None:
                        # If optimization fails, use the last known good solution.
                        solution_info['variables'] = self.last_solution
                        print("Using the last successful solution.")
                    break
                    
            except Exception as e:
                print(f"An error occurred during optimization: {e}")
                solution_info = {
                    'success': False,
                    'error': str(e),
                    'reconstruction_count': reconstruction_count
                }
                break
            
            finally:
                if 'model' in locals():
                    model.dispose()
        
        if reconstruction_count >= max_reconstructions:
            print(f"Maximum number of reconstructions ({max_reconstructions}) reached, optimization terminated.")
        
        return solution_info
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """
        Gets a summary of the optimization process.
        
        Returns:
            A dictionary with optimization summary information.
        """
        if not self.optimization_history:
            return {'message': 'No optimization has been performed yet.'}
        
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


# Example usage and helper functions

def create_linear_constraints_from_sparse_format(AElement, ACol, bElement, variables_dict, variable_name='Dt'):
    """
    Converts linear inequality constraints from a sparse matrix format to Gurobi constraints.
    
    Args:
        AElement: A list containing the non-zero element values of vector 'a'.
        ACol: A list containing the position indices of the non-zero elements in vector 'a'.
        bElement: A list containing the corresponding values of 'b'.
        variables_dict: A dictionary of Gurobi variables.
        variable_name: The name of the variables (default is 'Dt').
        
    Returns:
        A list of constraints that can be directly added to a Gurobi model.
    """
    constraints = []
    
    if len(AElement) != len(ACol):
        raise ValueError("The lengths of AElement and ACol must be the same.")
    
    constraints = []
    num_constraints = len(bElement)
    
    for i in range(num_constraints):
        elementA=AElement[i]
        elementCol=ACol[i]
        elementB=bElement[i]

        constraint_expr = gp.LinExpr()
        for j in range(len(elementA)):
            constraint_expr += elementA[j] * variables_dict[variable_name][elementCol[j]]
        
        # Add constraint a'*Dt <= b
        constraints.append(constraint_expr <= elementB)
    
    return constraints

def stability_check_function(variables, solution, nx, ny, nz, tEnd, lastSolution):
    """
    Example constraint checking function (for lazy constraints).
    
    Args:
        variables: Dictionary of variable objects.
        solution: The current solution as a numpy array.
        nx, ny, nz: Model dimensions.
        tEnd: End time.
        lastSolution: The solution from the previous iteration.
        
    Returns:
        (is_satisfied, lazy_constraints): A tuple indicating if the constraint is satisfied and a list of lazy constraints to add.
    """

    optTime=OperationTime(nx, ny, nz)
    optTime.from_optimization_variables(solution)

    lastOptTime=OperationTime(nx, ny, nz)
    lastOptTime.from_optimization_variables(lastSolution)

    # Sort the elements in the second column of optTime.time_matrix in ascending order
    # and get the sequence of element indices.
    sorted_idx=np.argsort(optTime.time_matrix[:,1])

    lazy_constraints = []

    for voxIdx in sorted_idx:
        if optTime.time_matrix[voxIdx,1]>=tEnd:
            break
        modelAtSM = getModelAtSM(optTime, voxIdx, tEnd)
        is_satisfied, outerBoundaryVoxels, islandVoxels = checkStabilityAtSM(modelAtSM, [voxIdx], nx, ny, nz, boxSize=4)
        if not is_satisfied:
            lastModelAtSM = getModelAtSM(lastOptTime, voxIdx, tEnd, equalCase=-1, flagOutIdx=True)

            solid_voxels = []
            for vox in outerBoundaryVoxels:
                if vox==voxIdx:
                    continue
                if lastModelAtSM[vox] == 1:  # 1 represents a solid voxel
                    solid_voxels.append(vox)

            if len(solid_voxels)==0:
                # Add constraints requiring all voxels of isolated components to be empty in the current model.
                for islandVox in islandVoxels:
                    if lastModelAtSM[islandVox] == -1:
                        raise ValueError("same time!")
                    elif lastModelAtSM[islandVox] == 1:
                        raise ValueError("solid island voxels!")
                    elif lastModelAtSM[islandVox] == 0:
                        lazy_constraints.append(variables['Dt'][2*voxIdx+1] - variables['Dt'][2*islandVox] <= 0)
                    else:
                        lazy_constraints.append(variables['Dt'][2*islandVox+1] - variables['Dt'][2*voxIdx+1] <= 0)
            else:
                # Add constraints requiring the voxels in solid_voxels to be solid in the current model.
                for solidVox in solid_voxels:
                    lazy_constraints.append(variables['Dt'][2*solidVox] - variables['Dt'][2*voxIdx+1] <= 0)
                    lazy_constraints.append(variables['Dt'][2*voxIdx+1] - variables['Dt'][2*solidVox+1] <= 0)
            
            return False, lazy_constraints
    
    return True, []


# Example Usage
if __name__ == "__main__":
    from functools import partial

    # Target function and initial solution (to be provided externally)
    nx, ny, nz = 5, 5, 5
    tEnd = 100
    targetModel = Model(nx, ny, nz, tEnd)
    initial_solution = np.zeros(2 * nx * ny * nz)
    # The last solution for the first iteration is the initial solution.
    lastSolution = initial_solution
    subModel = subModelPara(posStart=(0, 0, 0), modelSize=(nx, ny, nz))
    
    # Create an instance of the optimization framework
    optimizer = GurobiOptimizationFramework(
        targetModel=targetModel,
        subModel=subModel,
        gurobi_params={
            'MIPGap': 0.01,
            'TimeLimit': 300,
            'OutputFlag': 1
        },
        threshold=1e-6
    )
    
    # Set the initial solution
    optimizer.set_initial_solution(initial_solution)
    
    # Set the constraint checking function templates
    optimizer.set_constraint_check_function_templates([
        (stability_check_function, {'nx': nx, 'ny': ny, 'nz': nz, 'tEnd': tEnd}),
        # More checking functions can be added here
    ])
    
    # Execute the optimization (pass parameters directly now)
    result = optimizer.optimize(
        max_reconstructions=5,
        convergence_tolerance=1e-4
    )
    
    # Print the results
    print("Optimization Result:", result)
    print("Optimization Summary:", optimizer.get_optimization_summary())
