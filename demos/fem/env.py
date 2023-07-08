import gym
import numpy as onp
import jax.numpy as np
import random
import pygame

from colorama import init, Fore, Back, Style
from problem import ProblemSetup

class TopOptEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size_x:int = 6, size_y:int = 6, render_mode=None, jax_model=None):
        # Dimensionality of the grid
        self.size_x, self.size_y = size_x, size_y 
        self.window_size = 512
        self.initial_rho_vector = onp.ones((self.size_x * self.size_y, 1))
        self.jax_model = jax_model
        self.points = self.jax_model.points
        self.cells = self.jax_model.cells
        

        # Our 3-dimensional array that stores the strain, boundaries points and force-load points
        self.observation_space = gym.spaces.Dict(
            {
                "strains": gym.spaces.Box(low=0.0, high=1., shape=(size_x,size_y), dtype=onp.float32),
                "boundary": gym.spaces.Box(low=0, high=1, shape=(size_x,size_y), dtype=int),
                "forces": gym.spaces.Box(low=0, high=1, shape=(size_x,size_y), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(size_x * size_y)

        
        '''
        TODO: Create the image array that we will be using for rendering, where:
            - boundary cells = 2 (red)
            - force-load cells = 3 (green)
            - removed cells = 0 (white)
            - the rest of the cells = 1 (gray)
        ''' 
        self._render_image = onp.ones((size_x, size_y))


        # The render mode we are using
        assert render_mode is None or render_mode in ["human", "rgb_array"]
        self.render_mode = render_mode
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _coloring(self):
        add_bounded_mask = onp.zeros((self.size_x, self.size_y))
        for bounded in self.bounded_cells:
            add_bounded_mask += onp.where(self.jax_model.cell_inds_matrix == bounded, 1, 0)

        add_loaded_mask = onp.zeros((self.size_x, self.size_y))
        for loaded in self.loaded_cells:
            add_loaded_mask += onp.where(self.jax_model.cell_inds_matrix == loaded, 2, 0)

        self._render_image = onp.ones((self.size_x, self.size_y)) + add_bounded_mask + add_loaded_mask

    
    def _remove_cell_color(self, x, y):
        self._render_image[x, y] = 0

    # TODO: Implement _get_obs correctly        
    def _get_obs(self):
        self._strains, self._bounds, self._forces = self.state_tensor_check[0,:,:], self.state_tensor_check[1,:,:], self.state_tensor_check[2,:,:]
        return {"strains": self._strains,
                "boundary": self._bounds,
                "forces": self._forces}
    
    def _get_info(self):
        return self.rho_matrix

    # TODO: Debug this
    def reset(self, seed=123, options=123):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.bounded_cells, self.loaded_cells = self.jax_model.select_bounded_and_loaded_cells()
        # print(f'self_loaded cells: {self.loaded_cells}')
        # exit()
        self.max_num_step = len(self.cells) - (len(self.bounded_cells) + len(self.loaded_cells))
        
        
        self.bounded_cells = [0,5]
        self.loaded_cells = np.array([30]) #[onp.random.choice([30:35],size=1)]
        self.bounded_points, self.loaded_points = self.jax_model.select_bounded_and_loaded_points(self.bounded_cells, self.loaded_cells)

        self.dirichlet_bc = self.jax_model.set_dirichlet_bc(self.bounded_points)

        self.neumann_bc = self.jax_model.set_neumann_bc(self.loaded_points)
        self.problem = self.jax_model.problem_define(self.dirichlet_bc, self.neumann_bc)
        self.rho_vector = onp.copy(self.initial_rho_vector)
        self.rho_matrix = self.jax_model._state_matrix_from_array(self.rho_vector, self.size_x, self.size_y)
        self.solution = self.jax_model.problem_solve(self.problem, self.rho_vector)
        self.initial_von_mises = self.problem.compute_von_mises_stress(self.solution)
        self.state_tensor_DQN, self.state_tensor_check = self.jax_model.create_state_space_tensor(self.rho_matrix, self.initial_von_mises, self.bounded_cells, self.loaded_cells)
        # List with the cells that we have already removed
        self.removed_cells = []

        self.current_state_tensor_DQN, self.current_state_tensor_check = self.state_tensor_DQN, self.state_tensor_check
        
        self.nb_removed_cells = 0
        self._coloring()

        observation = self._get_obs()
        info = self._get_info()

        self.special_print(0)
        return self.current_state_tensor_DQN


    def step(self, action):
        
        ## Action reperesents the cell number to remove from topology
        cell_to_be_removed = action
        print(f'TENSOR shape: {self.current_state_tensor_DQN.shape}')
        reward = 0
        self.next_state_tensor_DQN = None
        if self.jax_model.check_illegal(self.rho_matrix, cell_to_be_removed, self.current_state_tensor_check, self.nb_removed_cells, self.max_num_step):
            reward = -1
            terminated = True
            indices = onp.argwhere(self.jax_model.cell_inds_matrix==cell_to_be_removed)
            index_x, index_y = indices[0][0], indices[0][1]
            self._remove_cell_color(index_x, index_y)
            self.special_print(f"{self.nb_removed_cells + 1} --> illegal")
        else:
            terminated = False
            self.current_state_tensor = self.state_tensor_DQN

            self.nb_removed_cells += 1
            rho_vector, rho_matrix = self.jax_model.update_density(self.rho_vector, cell_to_be_removed)
            solution = self.jax_model.problem_solve(self.problem, rho_vector)
            von_mises = self.problem.compute_von_mises_stress(solution)
            reward = self.jax_model.positive_reward(self.initial_von_mises, von_mises, self.nb_removed_cells, self.size_x*self.size_y)
            self.next_state_tensor_DQN, self.next_state_tensor_check= self.jax_model.create_state_space_tensor(rho_matrix, von_mises, self.bounded_cells, self.loaded_cells)
            indices = onp.argwhere(self.jax_model.cell_inds_matrix==cell_to_be_removed)
            index_x, index_y = indices[0][0], indices[0][1]
            self._remove_cell_color(index_x, index_y)
            self.special_print(self.nb_removed_cells)

            # Add the newly removed cell to th elist of removed cells
            self.removed_cells.append(cell_to_be_removed)
            
        return self.current_state_tensor_DQN, action, reward, self.next_state_tensor_DQN, terminated
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def special_print(self, counter): 
        def aux(value: int):
            if value < 10:
                return (f"0{value}")
            else:
                return (f"{value}")
        index = -1    
        print(f"Step: {counter}")
        for i in range(self.size_y):
            for j in range(self.size_x):
                index += 1
                if self._render_image[i][j] == 0:
                    print(Style.BRIGHT + Back.WHITE + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # White background
                elif self._render_image[i][j] == 1:
                    print(Style.BRIGHT + Back.BLUE + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # Blue background
                elif self._render_image[i][j] == 2:
                    print(Style.BRIGHT + Back.RED + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # Red background
                elif self._render_image[i][j] == 3:
                    print(Style.BRIGHT + Back.GREEN + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # Green background
                else:
                    print(Style.BRIGHT + Back.MAGENTA + Fore.RED + f"|{aux((self.size_y-i-1) + (self.size_x*j))}|", end="") # Magenta background
            print()
        print()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((190,190,190))
        pix_square_size = (
            self.window_size / self.size_x,
            self.window_size / self.size_y
        )  # The size of a single grid square in pixels

        # We draw the missing squares
        for i in range(self.size_x):
            for j in range(self.size_y):
                if self._render_image.T[i][j] == 0:
                    cell_coordinates = onp.array([i,j])
                    pygame.draw.rect(
                        canvas,
                        (255, 255, 255),
                        pygame.Rect(
                            pix_square_size[0] * cell_coordinates,
                            (pix_square_size[0], pix_square_size[1]),
                        )
                    )
                # We draw the boundary
                elif self._render_image.T[i][j] == 2:
                    cell_coordinates = onp.array([i,j])
                    pygame.draw.rect(
                        canvas,
                        (255, 0, 0),
                        pygame.Rect(
                            pix_square_size[0] * cell_coordinates,
                            (pix_square_size[0], pix_square_size[1]),
                        )
                    )
                # We draw the forces
                elif self._render_image.T[i][j] == 3:
                    cell_coordinates = onp.array([i,j])
                    pygame.draw.rect(
                        canvas,
                        (0, 255, 0),
                        pygame.Rect(
                            pix_square_size[0] * cell_coordinates,
                            (pix_square_size[0], pix_square_size[1]),
                        )
                    )

        # Finally, add some gridlines
        for x in range(self.size_x + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size[0] * x),
                (self.window_size, pix_square_size[1] * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size[0] * x, 0),
                (pix_square_size[1] * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(10)
        else:  # rgb_array
            return onp.transpose(
                onp.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

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
    num_steps = num_of_cells - (num_bounded_cell + num_loaded_cell)
    # instance definition
    simulator = ProblemSetup(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, num_bounded_cell=num_bounded_cell, num_loaded_cell=num_loaded_cell, 
                         filled_density=filled_density, void_density=void_density, dim=dim, vec=vec)
    env = TopOptEnv(size_x=Nx, size_y=Ny, render_mode="human", jax_model=simulator)
    init(autoreset=True)


    terminated = False
    env.reset()
    while not terminated:
        cell_remove = onp.random.choice(onp.arange(num_of_cells), 1)
        _, _, _, _, terminated = env.step(cell_remove)

    print()
    # env.close()
