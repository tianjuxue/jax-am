import numpy as onp
import jax
import jax.numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label

from jax_am.fem.core import FEM
from jax_am.fem.solver import solver, ad_wrapper
from jax_am.fem.utils import save_sol
from jax_am.fem.generate_mesh import get_meshio_cell_type, Mesh
from jax_am.common import rectangle_mesh
from fem_model import Elasticity

os.environ["CUDA_VISIBLE_DEVICES"] = "2" # --> Only activate when there is a CUDA-device in the system

def _clear_previous_output_files(self):
    """
    Clears previous outputs in the current folder.
    """
    data_path = os.path.join(os.path.dirname(__file__), 'data') 
    files = glob.glob(os.path.join(data_path, f'vtk/*'))
    for f in files:
        os.remove(f)

class ProblemSetup(Elasticity):
    def __init__(self, Nx, Ny, Lx, Ly, num_bounded_cell=2, num_loaded_cell=1, filled_density=1., void_density=0., dim=2, vec=2):
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.num_bounded_cell, self.num_loaded_cell = num_bounded_cell, num_loaded_cell
        self.filled_density, self.void_density = filled_density, void_density
        self.vec = vec
        self.dim = dim
        self.ele_type = 'QUAD4'
        self.cell_type = get_meshio_cell_type(self.ele_type)
        self.meshio_mesh = rectangle_mesh(Nx=Nx, Ny=Ny, domain_x=Lx, domain_y=Ly)
        self.mesh = Mesh(self.meshio_mesh.points, self.meshio_mesh.cells_dict[self.cell_type])
        self.cells = self.mesh.cells
        self.points = self.mesh.points
        self.cell_inds = np.arange((Nx * Ny), dtype=np.int32)
        self.cell_inds_matrix = self._state_matrix_from_array(self.cell_inds, self.Nx, self.Ny)
        self.point_inds_matrix = self._state_matrix_from_array(np.arange(len(self.points)), self.Nx+1, self.Ny+1)


    def _state_matrix_from_array(self, state_array: np.ndarray, num_row: int, num_column: int) -> np.ndarray:
        """
        Converts given array to matrix form with same topological representation of JAX-FEM format.
                                               [2   5   8]                 
        e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8] -->   [1   4   7]
                                               [0   3   6]
        Args:
            state_array (np.ndarray): input array 
            num_row (int)           : number of row of output matrix
            num_column (int)        : number of column of output matrix
        Returns:
            Output matrix in predefined geometry represantation defined by Jax-Fem
        """
        return onp.rot90(onp.reshape(state_array, (num_row, num_column)), k=1, axes=(0, 1))


    def _state_array_from_matrix(state_matrix: np.ndarray) -> np.ndarray:    
        """
        Converts a state matrix into state vector. 
                [2   5   8]                 
        e.g.    [1   4   7] --> [0, 1, 2, 3, 4, 5, 6, 7, 8]
                [0   3   6]
        Args:
            state_matrix (np.ndarray): input state matrix represents topological indexing format of Jax-FEM
        Returns:
            Array representation of given matrix.
        """
        return onp.reshape(onp.rot90(state_matrix, k=1, axes=(1, 0)), (-1))


    def _catagorize_cells(self):
        """ 
        Categorizes the cells wrt their topological locations.
        Note: Left-Top-Right-Bottom edge cell indices do not contain corner element indices at these edges
                [2   5   8]                 
        e.g.    [1   4   7]
                [0   3   6]
        Args:
        Returns:
            inner_cell_inds         :   e.g. [4]
            outer_cell_inds         :   e.g. [0, 1, 2, 3, 5, 6, 7, 8]
            outer_corner_cell_inds  :   e.g. [0, 6, 8, 2]
            left_edge_cell_inds     :   e.g. [1]
            top_edge_cell_inds      :   e.g. [5]
            right_edge_cell_inds    :   e.g. [7]
            bottom_edge_cell_inds   :   e.g. [3]
        """
        cell_inds_matrix = self._state_matrix_from_array(self.cell_inds, self.Nx, self.Ny)
        inner_cell_inds = onp.reshape((cell_inds_matrix)[1:self.Nx-1, 1:self.Ny-1], -1)
        outer_cell_inds = onp.delete(self.cell_inds, inner_cell_inds)
        outer_corner_cell_inds = onp.array((cell_inds_matrix[self.Nx-1, 0], cell_inds_matrix[self.Nx-1, self.Ny-1], cell_inds_matrix[0, self.Ny-1], cell_inds_matrix[0, 0]), dtype=int)
        left_edge_cell_inds = cell_inds_matrix[1:self.Ny-1, 0][::-1]
        top_edge_cell_inds = cell_inds_matrix[0, 1:self.Nx-1]
        right_edge_cell_inds = cell_inds_matrix[1:self.Ny-1, self.Nx-1][::-1]
        bottom_edge_cell_inds = cell_inds_matrix[self.Ny-1, 1:self.Nx-1]
        return inner_cell_inds, outer_cell_inds, outer_corner_cell_inds, left_edge_cell_inds, top_edge_cell_inds, right_edge_cell_inds, bottom_edge_cell_inds


    def select_bounded_and_loaded_cells(self):
        """ 
        Performs random cell selection to assign boundary conditions on their choses points.
        Note: Number of cells to be selected for Dirichlet and Neumann boundary conditions are passed as input for the class.
        Args:
            self.num_bounded_cell   (int)
            self.num_loaded_cell    (int)
        Returns:
            bounded_cell_inds       (list)
            loaded_cell_inds        (list)
        """
        _ , outer_cell_inds, _, _, _, _, _ = self._catagorize_cells()
        bounded_cell_inds = onp.random.choice(outer_cell_inds, self.num_bounded_cell, replace=False)
        cell_inds = onp.delete(self.cell_inds, bounded_cell_inds)
        loaded_cell_inds = onp.random.choice(cell_inds, self.num_loaded_cell, replace=False)
        #loaded_cell_inds = [30]
        return bounded_cell_inds, loaded_cell_inds

    def select_bounded_and_loaded_points(self, bounded_cell_inds: np.ndarray, loaded_cell_inds: np.ndarray) -> list:
        """
        Performs point selection to assign boundary conditions for given cells.
        Note : It contains a part to be update. For now it is implemented such to be work properly in neumann bc application, but deviates from the paper!
        Args:
            bounded_cell_inds   (np.ndarray):   Cell indices selected for Dirichlet BC assignment.
            loaded_cell_inds    (np.ndarray):   Cell indices selected for Neumann BC assignment.
        Returns:
            bounded_cell_inds   (list): Selected point indices wrt given cell indices for Dirichlet BC assignment.
            loaded_cell_inds    (list): Selected point indices wrt given cell indices for Neumann BC assignment.
        """
        inner_cell_inds, outer_cell_inds, corner_cell_inds, left_edge_cell_inds, top_edge_cell_inds, right_edge_cell_inds, bottom_edge_cell_inds= self._catagorize_cells()
        
        bounded_point_inds = []
        loaded_point_inds = []
        for bounded_cell in bounded_cell_inds:
            if bounded_cell in corner_cell_inds:
                bounded_point = self.cells[bounded_cell][onp.where(bounded_cell == corner_cell_inds)]
                bounded_point_inds.append(int(bounded_point))
            else:
                if bounded_cell in left_edge_cell_inds:
                    bounded_point1, bounded_point2 = self.cells[bounded_cell][0], self.cells[bounded_cell][3]
                elif bounded_cell in top_edge_cell_inds:
                    bounded_point1, bounded_point2 = self.cells[bounded_cell][2], self.cells[bounded_cell][3]
                elif bounded_cell in right_edge_cell_inds:
                    bounded_point1, bounded_point2 = self.cells[bounded_cell][1], self.cells[bounded_cell][2]
                else:
                    bounded_point1, bounded_point2 = self.cells[bounded_cell][0], self.cells[bounded_cell][1]
                bounded_point_inds.append(bounded_point1)
                bounded_point_inds.append(bounded_point2)
        for loaded_cell in loaded_cell_inds:
            if loaded_cell in corner_cell_inds:
                # ! FIX IN THE FUTURE (for now selects random 2 nodes around the cell)
                # loaded_point = cells[loaded_cell][onp.where(loaded_cell == outer_corner_cell_inds)]
                # loaded_points.append(loaded_point)
                index = onp.random.randint(0, 4)
                loaded_point1, loaded_point2 = self.cells[loaded_cell][index], self.cells[loaded_cell][(index+1)%4]
                loaded_point_inds.append(loaded_point1)
                loaded_point_inds.append(loaded_point2)
            else:
                if loaded_cell in bottom_edge_cell_inds:
                    loaded_point1, loaded_point2 = self.cells[loaded_cell][0], self.cells[loaded_cell][1]
                elif loaded_cell in left_edge_cell_inds:
                    loaded_point1, loaded_point2 = self.cells[loaded_cell][0], self.cells[loaded_cell][3]
                elif loaded_cell in top_edge_cell_inds:
                    loaded_point1, loaded_point2 = self.cells[loaded_cell][2], self.cells[loaded_cell][3]
                elif loaded_cell in right_edge_cell_inds:
                    loaded_point1, loaded_point2 = self.cells[loaded_cell][1], self.cells[loaded_cell][2]
                else:
                    index = onp.random.randint(0, 4)
                    loaded_point1, loaded_point2 = self.cells[loaded_cell][index], self.cells[loaded_cell][(index+1)%4]
                loaded_point_inds.append(loaded_point1)
                loaded_point_inds.append(loaded_point2)
        return sorted([*set(bounded_point_inds)]), sorted([*set(loaded_point_inds)])    # FIX IN THE FUTURE (for now if the loaded_points has 1 element, causes error!)

    def _cell_point_relation_check(self):
        """
        Easy check for confirmation of selected cell and selected point relations.
        Args: TBD
        Returns:
            Prints points inds and its index for given cell, otherwise returns a warning message.
        """
        pass

    def set_dirichlet_bc(self, selected_points: list) -> list:
        """
        Creates required Dirichlet boundary input for Jax-FEM solver for given points.
        Note : It assigns 0 displacement to given points in 2 direction.
        Note : This method includes hardcoding and right now work for len(selected_points) = 2 or 3 or 4 cases.
        Args:
            selected_points (list): Selected points for Dirichlet BC assignment
        Returns:
            Required list for JAX-FEM solver contains fix point locations, vectors (in which directions the displacement should be applied), value list (displacement value)
        """
        fix_location_list = []
        vector_list = []
        dirichlet_value_list = []
        if len(selected_points) == 2:
            fix_location1 = lambda point: np.logical_and(np.isclose(point[0], self.points[selected_points[0]][0]), np.isclose(point[1], self.points[selected_points[0]][1]))
            fix_location2 = lambda point: np.logical_and(np.isclose(point[0], self.points[selected_points[1]][0]), np.isclose(point[1], self.points[selected_points[1]][1]))
            vector_list = [0, 1, 0, 1]
            dirichlet_value = lambda point: 0.
            fix_location_list = [fix_location1, fix_location1, fix_location2, fix_location2]
            dirichlet_value_list = [dirichlet_value, dirichlet_value, dirichlet_value, dirichlet_value]
        if len(selected_points) == 3:
            fix_location1 = lambda point: np.logical_and(np.isclose(point[0], self.points[selected_points[0]][0]), np.isclose(point[1], self.points[selected_points[0]][1]))
            fix_location2 = lambda point: np.logical_and(np.isclose(point[0], self.points[selected_points[1]][0]), np.isclose(point[1], self.points[selected_points[1]][1]))
            fix_location3 = lambda point: np.logical_and(np.isclose(point[0], self.points[selected_points[2]][0]), np.isclose(point[1], self.points[selected_points[2]][1]))
            vector_list = [0, 1, 0, 1, 0, 1]
            dirichlet_value = lambda point: 0.
            fix_location_list = [fix_location1, fix_location1, fix_location2, fix_location2, fix_location3, fix_location3]
            dirichlet_value_list = [dirichlet_value, dirichlet_value, dirichlet_value, dirichlet_value, dirichlet_value, dirichlet_value]
        if len(selected_points) == 4:
            fix_location1 = lambda point: np.logical_and(np.isclose(point[0], self.points[selected_points[0]][0]), np.isclose(point[1], self.points[selected_points[0]][1]))
            fix_location2 = lambda point: np.logical_and(np.isclose(point[0], self.points[selected_points[1]][0]), np.isclose(point[1], self.points[selected_points[1]][1]))
            fix_location3 = lambda point: np.logical_and(np.isclose(point[0], self.points[selected_points[2]][0]), np.isclose(point[1], self.points[selected_points[2]][1]))
            fix_location4 = lambda point: np.logical_and(np.isclose(point[0], self.points[selected_points[3]][0]), np.isclose(point[1], self.points[selected_points[3]][1]))
            vector_list = [0, 1, 0, 1, 0, 1, 0, 1]
            dirichlet_value = lambda point: 0.
            fix_location_list = [fix_location1, fix_location1, fix_location2, fix_location2, fix_location3, fix_location3, fix_location4, fix_location4]
            dirichlet_value_list = [dirichlet_value, dirichlet_value, dirichlet_value, dirichlet_value, dirichlet_value, dirichlet_value, dirichlet_value, dirichlet_value]
        return [fix_location_list, vector_list, dirichlet_value_list]
    
    def set_neumann_bc(self, selected_points: list) -> list:
        """
        Creates required Neumann boundary input for Jax-FEM solver for given points.
        Note: NEED TO BE DEFINED MORE PROPERLY.
        Args:
            selected_points (list): Selected points for Neumann BC assignment
        Returns:
            Required list for JAX-FEM solver contains load point locations, force values in each axis assigned in random directions
        """
        load_location_list = []
        neumann_val_list = []
        # ! FIX IN THE FUTURE   (returns some values but does not work as we wish)
        load_location = lambda point: np.logical_and(
            np.isclose(point[0], (self.points[selected_points[0]][0] + self.points[selected_points[1]][0])/2, atol= 1e-5 + onp.abs(self.points[selected_points[0]][0] - self.points[selected_points[1]][0])),
            np.isclose(point[1], (self.points[selected_points[0]][1] + self.points[selected_points[1]][1])/2, atol= 1e-5 + onp.abs(self.points[selected_points[0]][1] - self.points[selected_points[1]][1])))
        neumann_val = lambda point: np.array([100., 100.]) * onp.random.choice([1, -1], 2)
        load_location_list.append(load_location)
        neumann_val_list.append(neumann_val)
        return [load_location_list, neumann_val_list]
        
    def problem_define(self, dirichlet_bc_info: list, neumann_bc_info: list):
        """
        Creates an Elasticity instance by passing required inputs.
        """
        return Elasticity(mesh=self.mesh, vec=self.vec, dim=self.dim, ele_type=self.ele_type, dirichlet_bc_info=dirichlet_bc_info, 
                    neumann_bc_info=neumann_bc_info, additional_info=('box',))
    
    def problem_solve(self, problem, rho: np.ndarray):
        """
        Advances one step the given problem instance through solver by taking rho as design input and returns the solution.
        """
        fwd_pred = ad_wrapper(problem, linear=True, use_petsc=True)
        return fwd_pred(rho) 

    def create_state_space_tensor(self, rho_vector: np.ndarray, von_mises: np.ndarray, bounded_cell_inds:np.ndarray, loaded_cell_inds: np.ndarray) -> np.ndarray:
        """
        Creates required DQN input 3 x N x N  state tensor 
        Args:
            rho_vector (np.ndarray)         : density vector
            von_mises (np.ndarray)          : von mises vector
            bounded_cell_inds (np.ndarray)  :
            loaded_cell_inds (np.ndarray)   :
        Returns:
            state_tensor_DQN                : NxNx3 tensor which will be used in DQN training
            state_tensor_check              : 3xNxN tensor used for illegality check and visualization
        """
        inverse_von_mises_array = (1 / von_mises) / np.max(1 / von_mises)
        bounded_cells_state_array = self.cell_inds[onp.where((self.cell_inds == bounded_cell_inds[0]) | (self.cell_inds == bounded_cell_inds[1]), 1, 0)]  # hard_coded to return 2 cells
        loaded_cells_state_array = self.cell_inds[onp.where((self.cell_inds == loaded_cell_inds), 1, 0)] # hard_coded to return 1 cell
        inverse_von_mises_matrix = self._state_matrix_from_array(inverse_von_mises_array, self.Nx, self.Ny)
        bounded_cells_state_matrix = self._state_matrix_from_array(bounded_cells_state_array, self.Nx, self.Ny)
        loaded_cells_state_matrix = self._state_matrix_from_array(loaded_cells_state_array, self.Nx, self.Ny)
        state_tensor_DQN = np.stack((inverse_von_mises_matrix, bounded_cells_state_matrix, loaded_cells_state_matrix), axis=2)
        state_tensor_check = np.stack((inverse_von_mises_matrix, bounded_cells_state_matrix, loaded_cells_state_matrix), axis=0)
        return state_tensor_DQN, state_tensor_check
    
    def check_illegal(self, rho_matrix: np.ndarray, new_point: int, state_tensor: np.ndarray, nb_step: int, nb_max_step: int) -> bool:
        """
        Checks whether the selected point can be removed
        Args:
            rho (np.ndarray)                    : The boolean mask of the topology (shape Nx x Ny)
            new_point (int)                     : The index of the cell to be removed 
            state_matrix (np.ndarray)           : The state matrix that contains inv_von_mises, bounded_cells, and loaded_cells arrays (shape : 3 x Nx x Ny)
            self.cell_inds_matrix (np.ndarray)  : The matrix contains cell indices in the order that represents geometry (x = 0 is at the LEFT, y = 0 is at the BOTTOM) (shape: Nx x Ny)
            self.filled_denstiy (float)         : The material intensity value for filled cells
            self.void_density (float)           : The material intensity value for void cells

        Returns:
            True if the cell can be removed and False otherwise
        """
        _, bounds, forces = state_tensor
        Nx, Ny = rho_matrix.shape
        new_point_inds = onp.argwhere(self.cell_inds_matrix == new_point)[0]
        x, y = new_point_inds

        # (A) If the to-be-removed point has the coordinates of a boundary condition or of a force origin
    
        if bounds[x, y]==self.filled_density or forces[x, y]==self.filled_density:
            print("\nIllegality check --> False")
            print(f"You are trying to remove bounded or loaded cell number {new_point}.")
            return True

        # (B) If the to-be-removed point has already been removed
        
        if rho_matrix[x, y] == self.void_density:
            print("\nIllegality check --> False")
            print(f"You are trying to remove already removed cell number {new_point}.")
            return True

        # (C) Making sure for only one connected component
        new_rho_matrix = onp.floor(rho_matrix)
        new_rho_matrix[x, y] = 0
        labeled, ncomponents = label(new_rho_matrix)
        if ncomponents > 1: 
            print("\nIllegality check --> False")
            print(f"More than one component by removing cell number {new_point}.")
            print(labeled)
            return True
        
        if nb_step > nb_max_step:
            return True
        
        # If everything complies to the rules:
        return False

    def test_check_illegal(self, state_tensor: np.ndarray, pre_created_rho: np.ndarray=None, pre_selected_cell_to_be_removed: int=None):
        """
        Created for testing check_illegal function using predefined or random scenarios.
        """
        rho_matrix = onp.where(onp.random.randint(2, size=(self.Nx, self.Ny))==0, self.void_density, self.filled_density)
        cell_to_be_removed = (self.cell_inds_matrix[onp.random.choice(onp.arange(self.Nx)), onp.random.choice(onp.arange(self.Ny))])
        if pre_created_rho is not None:
            rho_matrix = pre_created_rho
        if pre_selected_cell_to_be_removed is not None:
            cell_to_be_removed = pre_selected_cell_to_be_removed
        check = self.check_illegal(rho_matrix, cell_to_be_removed, state_tensor)
        if check:
            print("LEGAL ACTION!!!")
        else:
            print()
            print("ILLEGAL ACTION!!!")
            print(f"Selected cell '{cell_to_be_removed}' can not be removed.")
            print(f"Cell_indices_matrix = \n{self.cell_inds_matrix}")
            print(f"Rho matrix = \n{rho_matrix}")
            print(f"Bounded cell matrix = \n{state_tensor[1]}")
            print(f"Loaded cell matrix = \n{state_tensor[2]}")


    def update_density(self, rho_vector: np.ndarray, cell_index: int) -> np.ndarray:
        """
        Updates selected index of density vector with self.void_density.
        Args:
            rho_vector (np.ndarray)             : Density vector that contains denstiy values for each cell
            new_point (int)                     : The index of the cell to be removed 
        Returns:
            Updated rho vector and its state represantation formatted rho matrix
        """
        rho_vector[cell_index] = self.void_density
        rho_matrix = self._state_matrix_from_array(rho_vector, self.Nx, self.Ny)
        return rho_vector, rho_matrix
        

    def positive_reward(self, initial_von_mises: float, current_von_mises: float, num_of_voided_cells: int, num_of_total_cells: int) -> float:
        """
        Updates selected index of density vector with self.void_density.
        Args:
            init_von_mises (float)          : Von mises stresses at initial state
            current_von_mises (float)       : Von mises stresses at the current state
            num_of_voided_cells (int)       : Number of voided cell including the current state
            num_of_total_cells (int)        : Number of total cell in the topology
        Returns:
            Positive reward value after each successful action
        """
        return (np.sum(current_von_mises) / np.sum(initial_von_mises)) ** 2 + (num_of_voided_cells / num_of_total_cells) ** 2

    
if __name__ == "__main__":
    # constant decleration for problem setup
    Nx, Ny = 6, 6
    Lx, Ly = 6, 6
    num_bounded_cell = 2
    num_loaded_cell = 1
    filled_density = 1.
    void_density = 1e-4
    dim = 2
    vec = 2
    # design variable initialization
    num_of_cells = Nx * Ny
    vf = 1
    init_rho_vector = vf*onp.ones((num_of_cells, 1)) 
    # optimization paramaters decleration
    num_episodes = 10
    max_num_steps = num_of_cells - (num_bounded_cell + num_loaded_cell)
    # instance definition
    trial = ProblemSetup(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, num_bounded_cell=num_bounded_cell, num_loaded_cell=num_loaded_cell, 
                         filled_density=filled_density, void_density=void_density, dim=dim, vec=vec)
    cell_inds_matrix, point_inds_matrix = trial.cell_inds_matrix, trial.point_inds_matrix
    # Optimization Loop
    reward_history = []
    step_history = []
    episode_counter = 0
    while episode_counter <= num_episodes:
        # reset part
        step_counter = 0
        num_void_cells = 0
        episode_counter += 1
        bounded_cells, loaded_cells = trial.select_bounded_and_loaded_cells()
        bounded_points, loaded_points = trial.select_bounded_and_loaded_points(bounded_cells, loaded_cells)
        dirichlet_bc = trial.set_dirichlet_bc(bounded_points)
        neumann_bc = trial.set_neumann_bc(loaded_points)
        problem = trial.problem_define(dirichlet_bc, neumann_bc)
        rho_vector = init_rho_vector
        rho_matrix = trial._state_matrix_from_array(rho_vector, Nx, Ny)
        solution = trial.problem_solve(problem, rho_vector)
        initial_von_mises = problem.compute_von_mises_stress(solution)
        von_mises = initial_von_mises
        # init_strain_energy = problem.compute_compliance(neumann_bc, solution)
        # initial_strain_energy = np.sum(1 / von_mises)
        reward = 0.
        illegality_check = False
        while not illegality_check:
            # update step part
            step_counter += 1
            state_tensor_DQN, state_tensor_check = trial.create_state_space_tensor(rho_matrix, von_mises, bounded_cells, loaded_cells)
            cell_to_be_removed = onp.random.choice(onp.arange(num_of_cells), 1)
            illegality_check = trial.check_illegal(rho_matrix, cell_to_be_removed, state_tensor_check, step_counter, max_num_steps)
            if illegality_check:
                reward -= 1
                reward_history.append(reward)
                step_history.append(step_counter)
                print(f'Episode {episode_counter} is terminated at step {step_counter}! Reward = {reward}')
                print(rho_matrix)
            else:
                rho_vector, rho_matrix = trial.update_density(rho_vector, cell_to_be_removed)
                num_void_cells += 1
                solution = trial.problem_solve(problem, rho_vector)
                # current_strain_energy = problem.compute_compliance(neumann_bc, solution)
                von_mises = problem.compute_von_mises_stress(solution)
                #current_strain_energy = np.sum(1 / von_mises)
                current_von_mises = von_mises
                # reward += trial.positive_reward(initial_strain_energy, current_strain_energy, num_void_cells, num_of_cells)
                reward += trial.positive_reward(initial_von_mises, current_von_mises, num_void_cells, num_of_cells)
                reward_history.append(reward)
                print(f'Episode = {episode_counter}: Step = {step_counter}: Reward = {reward}')
    print(f"Max_reward = {max(reward_history)}, Max_reached_step = {max(step_history)} out of total {max_num_steps} steps!")
    print()
