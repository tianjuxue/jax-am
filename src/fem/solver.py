import jax
import jax.numpy as np
import numpy as onp
import time
from functools import partial


def apply_bc(res_fn, problem):
    def A_fn(dofs):
        """Apply Dirichlet boundary conditions
        """
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        res = res_fn(dofs).reshape(sol.shape)
        for i in range(len(problem.node_inds_list)):
            res = (res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set
                   (sol[problem.node_inds_list[i], problem.vec_inds_list[i]], unique_indices=True))
            res = res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].add(-problem.vals_list[i])
        return res.reshape(-1)
    return A_fn


def row_elimination(res_fn, problem):
    def fn_dofs_row(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        res = res_fn(dofs).reshape(sol.shape)
        for i in range(len(problem.node_inds_list)):
            res = (res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set
                   (sol[problem.node_inds_list[i], problem.vec_inds_list[i]], unique_indices=True))
        return res.reshape(-1)
    return fn_dofs_row


def assign_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(problem.vals_list[i])
    return sol.reshape(-1)


def assign_zero_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(0.)
    return sol.reshape(-1)

 
 
def assign_ones_bc(dofs, problem):
    sol = dofs.reshape((problem.num_total_nodes, problem.vec))
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(1.)
    return sol.reshape(-1)


def get_flatten_fn(fn_sol, problem):
    def fn_dofs(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        val_sol = fn_sol(sol)
        return val_sol.reshape(-1)
    return fn_dofs


def get_A_fn_linear_fn(dofs, fn):
    def A_fn_linear_fn(inc):
        primals, tangents = jax.jvp(fn, (dofs,), (inc,))
        return tangents
    return A_fn_linear_fn


def get_A_fn_linear_fn_JFNK(dofs, fn):
    """Jacobian-free Newton–Krylov (JFNK) method. 
    Not quite used since we have auto diff to compute exact JVP.
    Knoll, Dana A., and David E. Keyes. 
    "Jacobian-free Newton–Krylov methods: a survey of approaches and applications." 
    Journal of Computational Physics 193.2 (2004): 357-397.
    """
    def A_fn_linear_fn(inc):
        EPS = 1e-3
        return (fn(dofs + EPS*inc) - fn(dofs))/EPS
    return A_fn_linear_fn


def operator_to_matrix(operator_fn, problem):
    """Only used for debugging purpose.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_nodes*problem.vec))
    return J


def jacobi_preconditioner(problem):
    C_sub = []
    for i in range(problem.vec):
        # (num_cells*num_quads, dim, dim)
        C_sub.append(problem.C[:, i*problem.dim:(i+1)*problem.dim, i*problem.dim:(i+1)*problem.dim])
    # (num_cells, num_quads, num_nodes, dim) -> (num_cells*num_quads, num_nodes, 1, dim)
    shape_grads_reshape = problem.shape_grads.reshape(-1, problem.num_nodes, 1, problem.dim)
    vals = []
    for i in range(problem.vec):
    # (num_cells*num_quads, num_nodes, 1, dim) @ (num_cells*num_quads, 1, dim, dim) @ (num_cells*num_quads, num_nodes, dim, 1)
    # (num_cells*num_quads, num_nodes) -> (num_cells, num_quads, num_nodes) -> (num_cells, num_nodes)
        vals.append(np.sum((shape_grads_reshape @ C_sub[i][:, None, :, :] @ np.transpose(shape_grads_reshape, 
                   axes=(0, 1, 3, 2))).reshape(problem.num_cells, problem.num_quads, problem.num_nodes) * problem.JxW[:, :, None], axis=1))
    # (vec, num_cells, num_nodes) -> (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
    vals = np.transpose(np.stack(vals), axes=(1, 2, 0)).reshape(-1, problem.vec)
    jacobi = np.zeros((problem.num_total_nodes, problem.vec))
    jacobi = jacobi.at[problem.cells.reshape(-1)].add(vals)

    jacobi = assign_ones_bc(jacobi.reshape(-1), problem) 
    return jacobi


def get_jacobi_precond(jacobi):
    def jacobi_precond(x):
        return x * (1./jacobi)
    return jacobi_precond


def test_jacobi_precond(problem, dofs, jacobi, A_fn):
    # TODO
    for ind in range(len(dofs)):
        test_vec = np.zeros(problem.num_total_nodes*problem.vec)
        test_vec = test_vec.at[ind].set(1.)
        print(f"{A_fn(test_vec)[ind]}, {jacobi[ind]}, ratio = {A_fn(test_vec)[ind]/jacobi[ind]}")

    print(f"compute jacobi preconditioner")
    print(f"np.min(jacobi) = {np.min(jacobi)}, np.max(jacobi) = {np.max(jacobi)}")
    print(f"finish jacobi preconditioner")
 

def linear_full_solve(problem, A_fn, precond):
    b = np.zeros((problem.num_total_nodes, problem.vec))
    b = assign_bc(b, problem).reshape(-1)
    jacobi = jacobi_preconditioner(problem)
    pc = get_jacobi_precond(jacobi) if precond else None
    dofs, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, x0=b, M=pc, tol=1e-10, atol=1e-10, maxiter=10000)
    return dofs


def linear_incremental_solver(problem, res_fn, A_fn, dofs, precond):
    """
    Lift solver
    dofs must already satisfy Dirichlet boundary conditions
    """
    b = -res_fn(dofs)
    jacobi = jacobi_preconditioner(problem)
    pc = get_jacobi_precond(jacobi) if precond else None
    # test_jacobi_precond(problem, dofs, jacobi_preconditioner(problem), A_fn)
    inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, x0=None, M=pc, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab
    dofs = dofs + inc
    return dofs


def compute_residual_val(res_fn, dofs):
   res_vec = res_fn(dofs)
   res_val = np.linalg.norm(res_vec)
   return res_val


def solver(problem, initial_guess=None, linear=False, precond=True):
    print("Start timing")
    start = time.time()

    if initial_guess is not None:
        sol = initial_guess
    else:
        sol = np.zeros((problem.num_total_nodes, problem.vec))

    dofs = sol.reshape(-1)

    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem) 

    if linear:
        # If the problem is known to be linear, there's no need to perform linearization.
        # Specifically, we save the cost of computing the fourh-order tangent tensor C.
        A_fn = get_A_fn_linear_fn(dofs, res_fn)
        dofs = assign_bc(dofs, problem).reshape(-1)
        dofs = linear_incremental_solver(problem, res_fn, A_fn, dofs, False)
    else:
        A_fn = problem.compute_linearized_residual
        A_fn = get_flatten_fn(A_fn, problem)
        A_fn = row_elimination(A_fn, problem)

        problem.newton_update(dofs.reshape(sol.shape))
        dofs = linear_full_solve(problem, A_fn, precond)
        res_val = compute_residual_val(res_fn, dofs)
        print(f"Before, res l_2 = {res_val}") 
        tol = 1e-6
        while res_val > tol:
            problem.newton_update(dofs.reshape(sol.shape))
            dofs = linear_incremental_solver(problem, res_fn, A_fn, dofs, precond)
            res_val = compute_residual_val(res_fn, dofs)
            print(f"res l_2 = {res_val}") 

    sol = dofs.reshape(sol.shape)
    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s]")
    print(f"max of sol = {np.max(sol)}")
    print(f"min of sol = {np.min(sol)}")

    return sol
