import numpy as onp
import jax
import jax.numpy as np
import time
import os
import glob
from functools import partial
from scipy.stats import qmc

from src.fem.apps.multi_scale.arguments import args

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

from src.fem.generate_mesh import box_mesh
from src.fem.jax_fem import Mesh, LinearElasticity, HyperElasticity, Laplace
from src.fem.solver import solver, assign_bc, get_A_fn_linear_fn, apply_bc
from src.fem.utils import save_sol
from src.fem.apps.multi_scale.utils import flat_to_tensor



def aug_solve(problem, initial_guess=None):
    print(f"Start timing, H_bar = \n{problem.H_bar}")
    start = time.time()

    p_splits = np.cumsum(np.array([len(x) for x in problem.p_node_inds_list_B])).tolist()
    d_splits = np.cumsum(np.array([len(x) for x in problem.node_inds_list])).tolist()
 
    num_dofs = problem.num_total_nodes * problem.vec
    p_lmbda_len = p_splits[-1]  
    d_lmbda_len = d_splits[-1]

    def operator_to_matrix(operator_fn):
        """Only used for debugging purpose.
        Can be used to print the matrix, check the conditional number, etc.
        """
        J = jax.jacfwd(operator_fn)(np.zeros(num_dofs + p_lmbda_len + d_lmbda_len))
        return J

    def get_Lagrangian():
        def split_lamda(lmbda):
            p_lmbda = lmbda[:p_lmbda_len]
            d_lmbda = lmbda[p_lmbda_len:]
            p_lmbda_split = np.split(p_lmbda, p_splits)
            d_lmbda_split = np.split(d_lmbda, d_splits)
            return p_lmbda_split, d_lmbda_split

        @jax.jit
        def Lagrangian_fn(aug_dofs):
            dofs, lmbda = aug_dofs[:num_dofs], aug_dofs[num_dofs:]
            sol = dofs.reshape((problem.num_total_nodes, problem.vec))
            lag_1 = problem.compute_energy(sol)

            p_lmbda_split, d_lmbda_split = split_lamda(lmbda)
            lag_2 = 0.
            for i in range(len(problem.p_node_inds_list_B)):
                lag_2 += np.sum(p_lmbda_split[i] * (sol[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]] - 
                                                    sol[problem.p_node_inds_list_A[i], problem.p_vec_inds_list[i]]))
            for i in range(len(problem.node_inds_list)):
                lag_2 += np.sum(d_lmbda_split[i] * (sol[problem.node_inds_list[i], problem.vec_inds_list[i]] - problem.vals_list[i]))

            return lag_1 + 1e2*lag_2

        return Lagrangian_fn

    print(f"num_dofs = {num_dofs}, p_lmbda_len = {p_lmbda_len}, d_lmbda_len = {d_lmbda_len}")

    if initial_guess is not None:
        aug_dofs = np.hstack((initial_guess.reshape(-1), np.zeros(p_lmbda_len + d_lmbda_len)))
    else:
        aug_dofs = np.zeros(num_dofs + p_lmbda_len + d_lmbda_len)

    Lagrangian_fn = get_Lagrangian()
    A_fn = jax.grad(Lagrangian_fn)

    linear_solve_step = 0

    b = -A_fn(aug_dofs)
    res_val = np.linalg.norm(b)
    print(f"Before calling Newton's method, res l_2 = {res_val}") 
    tol = 1e-6
    while res_val > tol:
        A_fn_linear = get_A_fn_linear_fn(aug_dofs, A_fn)
        debug = False
        if debug:
            # Check onditional number of the matrix
            A_dense = operator_to_matrix(A_fn_linear)
            print(f"conditional number = {np.linalg.cond(A_dense)}")
            # print(f"max A = {np.max(A_dense)}")
            # print(A_dense.shape)
            # print(A_dense)
            inc = jax.numpy.linalg.solve(A_dense, b)
        else:
            inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab

        linear_solve_step += 1
        aug_dofs = aug_dofs + inc
        b = -A_fn(aug_dofs)
        res_val = np.linalg.norm(b)
        print(f"step = {linear_solve_step}, res l_2 = {res_val}") 

    sol = aug_dofs[:num_dofs].reshape((problem.num_total_nodes, problem.vec))

    # print(f"lmbda = {aug_dofs[num_dofs:]}")

    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s], finished in {linear_solve_step} linear solve steps")

    return sol


class RVE(Laplace):
    """Solves for the fluctuation field, not the displacement field.
    """
    def __init__(self, name, mesh, dirichlet_bc_info=None, periodic_bc_info=None, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, dirichlet_bc_info, periodic_bc_info, neumann_bc_info, source_info)
        self.H_bar = None
        self.physical_quad_points = self.get_physical_quad_points()
        self.E, self.nu = self.compute_moduli()

    def stress_strain_fns(self):
        def psi(F, E, nu):
            # E = 70e3
            # nu = 0.3
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2. 
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, E, nu):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, E, nu)
            return P
        vmap_stress = jax.vmap(first_PK_stress)
        vmap_energy = jax.vmap(psi)
        return vmap_stress, vmap_energy

    def compute_physics(self, sol, u_grads):
        u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim) + self.H_bar[None, :, :]
        vmap_stress, _ = self.stress_strain_fns()
        sigmas = vmap_stress(u_grads_reshape, self.E, self.nu).reshape(u_grads.shape)
        return sigmas

    def compute_moduli(self):
        # TODO: a lot of redundant code here
        center = np.array([args.L/2., args.L/2., args.L/2.])
        def E_map(point):
            E = np.where(np.max(np.absolute(point - center)) < args.L*0.3, 1e3, 1e2) # 1e3, 1e2
            # E = np.where(point[0] < 0.5, 1e3, 1e3) # 1e3, 1e2
            return E

        def nu_map(point):
            nu = np.where(np.max(np.absolute(point - center)) < args.L*0.3, 0.3, 0.4) # 0.3, 0.4
            return nu

        E = jax.vmap(jax.vmap(E_map))(self.physical_quad_points).reshape(-1)
        nu = jax.vmap(jax.vmap(nu_map))(self.physical_quad_points).reshape(-1)

        return E, nu

    def fluc_to_disp(self, sol_fluc):
        sol_disp = (self.H_bar @ self.points.T).T + sol_fluc
        return sol_disp

    def compute_energy(self, sol):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim) 
        F_reshape = u_grads.reshape(-1, self.vec, self.dim) + self.H_bar[None, :, :] + np.eye(self.dim)[None, :, :]
        _, vmap_energy  = self.stress_strain_fns()
        psi = vmap_energy(F_reshape, self.E, self.nu).reshape(u_grads.shape[:2]) # (num_cells, num_quads)
        energy = np.sum(psi * self.JxW)
        return energy


def compute_periodic_inds(location_fns_A, location_fns_B, mappings, vecs, mesh):
    p_node_inds_list_A = []
    p_node_inds_list_B = []
    p_vec_inds_list = []
    for i in range(len(location_fns_A)):
        node_inds_A = np.argwhere(jax.vmap(location_fns_A[i])(mesh.points)).reshape(-1)
        node_inds_B = np.argwhere(jax.vmap(location_fns_B[i])(mesh.points)).reshape(-1)
        points_set_A = mesh.points[node_inds_A]
        points_set_B = mesh.points[node_inds_B]

        EPS = 1e-8
        node_inds_B_ordered = []
        for node_ind in node_inds_A:
            point_A = mesh.points[node_ind]
            dist = np.linalg.norm(mappings[i](point_A)[None, :] - points_set_B, axis=-1)
            node_ind_B_ordered = node_inds_B[np.argwhere(dist < EPS)].reshape(-1)
            node_inds_B_ordered.append(node_ind_B_ordered)

        node_inds_B_ordered = np.array(node_inds_B_ordered).reshape(-1)
        vec_inds = np.ones_like(node_inds_A, dtype=np.int32)*vecs[i]

        p_node_inds_list_A.append(node_inds_A)
        p_node_inds_list_B.append(node_inds_B_ordered)
        p_vec_inds_list.append(vec_inds)

        # TODO: A better way needed
        assert len(node_inds_A) == len(node_inds_B_ordered)

    return p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list


def rve_problem():
    problem_name = "rve"
    L = args.L
    meshio_mesh = box_mesh(10, 10, 10, L, L, L)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def corner(point):
        return np.isclose(np.linalg.norm(point), 0., atol=1e-5)

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], L, atol=1e-5)

    def front(point):
        return np.isclose(point[1], 0., atol=1e-5)

    def back(point):
        return np.isclose(point[1], L, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], L, atol=1e-5)

    def dirichlet(point):
        return 0.

    def mapping_x(point_A):
        point_B = point_A + np.array([L, 0., 0.])
        return point_B

    def mapping_y(point_A):
        point_B = point_A + np.array([0., L, 0.])
        return point_B

    def mapping_z(point_A):
        point_B = point_A + np.array([0., 0., L])
        return point_B

    location_fns = [corner]*3
    value_fns = [dirichlet]*3
    vecs = [0, 1, 2]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    location_fns_A = [left]*3 + [front]*3 + [bottom]*3
    location_fns_B = [right]*3 + [back]*3 + [top]*3
    mappings = [mapping_x]*3 + [mapping_y]*3 + [mapping_z]*3
    vecs = [0, 1, 2]*3

    periodic_bc_info = compute_periodic_inds(location_fns_A, location_fns_B, mappings, vecs, jax_mesh)
    problem = RVE(f"{problem_name}", jax_mesh, dirichlet_bc_info=dirichlet_bc_info, periodic_bc_info=periodic_bc_info)

    return problem


def debug_periodic(problem, sol_fluc):
    p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list = problem.periodic_bc_info
    a = sol_fluc[p_node_inds_list_A[0], p_vec_inds_list[0]]
    b = sol_fluc[p_node_inds_list_B[0], p_vec_inds_list[0]]
    ap = problem.mesh.points[p_node_inds_list_A[0]]
    bp = problem.mesh.points[p_node_inds_list_B[0]]
    print(np.hstack((ap, bp, a[:, None], b[:, None]))[:10])


def exp():
    problem = rve_problem()
    H_bar = np.array([[-0.009, 0., 0.],
                      [0., -0.009, 0.],
                      [0., 0., 0.025]])

    problem.H_bar = H_bar

    # material = np.where(problem.E > 2*1e2, 0., 1.)

    sol_fluc_ini = np.zeros((problem.num_total_nodes, problem.vec))
    sol_fluc_ini = assign_bc(sol_fluc_ini, problem)
    energy = problem.compute_energy(sol_fluc_ini)
    print(f"Initial energy = {energy}")

    sol_disp_ini = problem.fluc_to_disp(sol_fluc_ini)
    jax_vtu_path = f"src/fem/apps/multi_scale/data/vtk/{problem.name}/sol_disp_ini.vtu"
    save_sol(problem, sol_disp_ini, jax_vtu_path, [("E", problem.E.reshape((problem.num_cells, problem.num_quads))[:, 0])])

    sol_fluc = aug_solve(problem)

    # ratios = [1.5, 1.8, 2.]
    # for ratio in ratios:
    #     problem.H_bar = ratio * H_bar
    #     sol_fluc = aug_solve(problem, initial_guess=sol_fluc)


    energy = problem.compute_energy(sol_fluc)
    print(f"Final energy = {energy}")

    sol_disp = problem.fluc_to_disp(sol_fluc)
    jax_vtu_path = f"src/fem/apps/multi_scale/data/vtk/{problem.name}/sol_disp.vtu"
    save_sol(problem, sol_disp, jax_vtu_path)

    jax_vtu_path = f"src/fem/apps/multi_scale/data/vtk/{problem.name}/sol_fluc.vtu"
    save_sol(problem, sol_fluc, jax_vtu_path)

    debug_periodic(problem, sol_fluc)



def solve_rve_problem(problem, sample_H_bar):
    base_H_bar = flat_to_tensor(sample_H_bar)
    problem.H_bar = base_H_bar
    sol_fluc = aug_solve(problem)
    energy = problem.compute_energy(sol_fluc)
    ratios = [0.5, 0.75, 0.9, 1.]
    if np.any(np.isnan(energy)):
        sol_fluc = np.zeros((problem.num_total_nodes, problem.vec))
        for ratio in ratios:
            problem.H_bar = ratio * base_H_bar
            sol_fluc = aug_solve(problem, initial_guess=sol_fluc)
        energy = problem.compute_energy(sol_fluc)

    return sol_fluc, np.hstack((sample_H_bar, energy))


def generate_samples():
    dim_H = 6
    sampler = qmc.Sobol(d=dim_H, scramble=False, seed=0)
    sample = sampler.random_base2(m=10)
    l_bounds = [-0.2]*dim_H
    u_bounds = [0.2]*dim_H
    scaled_sample = qmc.scale(sample, l_bounds, u_bounds)
    return scaled_sample
 

def collect_data():
    problem = rve_problem()
    date = f"09052022"
    root_numpy = os.path.join(f"src/fem/apps/multi_scale/data/numpy/training", date)
    if not os.path.exists(root_numpy):
        os.makedirs(root_numpy)

    root_vtk = os.path.join(f"src/fem/apps/multi_scale/data/vtk/training", date)
    if not os.path.exists(root_vtk):
        os.makedirs(root_vtk)

    samples = generate_samples()
    complete = [i for i in range(len(samples))]

    # print(files)
    # print(os.path.exists(files[0]))
    # print(f"{int(files[0][-9:-4]):05d}")
    # print(todo)
    print(args.device)
    print(samples[-10:])
    # exit()

    onp.random.seed(args.device)
    while True:
        files = glob.glob(root_numpy + f"/*.npy")
        done = [int(file[-9:-4]) for file in files]
        todo = list(set(complete) - set(done))
        if len(todo) == 0:
            break
        chosen_ind = onp.random.choice(todo)
        print(f"Solving problem # {chosen_ind} on device = {args.device}, done = {len(done)}, todo = {len(todo)}, total = {len(complete)} ")
        sample_H_bar = samples[chosen_ind]
        sol_fluc, data = solve_rve_problem(problem, sample_H_bar)
        if np.any(np.isnan(data)):
            print(f"######################################### Failed solve, check why!")
            onp.savetxt(os.path.join(root_numpy, f"{chosen_ind:05d}.txt"), sample_H_bar)
        else:
            onp.save(os.path.join(root_numpy, f"{chosen_ind:05d}.npy"), data)

        sol_disp = problem.fluc_to_disp(sol_fluc)
        jax_vtu_path = os.path.join(root_vtk, f"sol_disp_{chosen_ind:05d}.vtu")
        save_sol(problem, sol_disp, jax_vtu_path)


    # print(f"step {i + 1} of {len(ts[1:])}, unix timestamp = {time.time()}")


if __name__=="__main__":
    # exp()
    # collect_data()
    exp()
