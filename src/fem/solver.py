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
        res = res_fn(sol)
        for i in range(len(problem.node_inds_list)):
            res = (res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set
                   (sol[problem.node_inds_list[i], problem.vec_inds_list[i]], unique_indices=True))
            res = res.at[problem.node_inds_list[i], problem.vec_inds_list[i]].add(-problem.vals_list[i])
        return res.reshape(-1)
    return A_fn


def row_elimination(fn_dofs, problem):
    def fn_dofs_row(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        res_dofs = fn_dofs(dofs)
        res_sol = res_dofs.reshape((problem.num_total_nodes, problem.vec))
        for i in range(len(problem.node_inds_list)):
            res_sol = (res_sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set
                      (sol[problem.node_inds_list[i], problem.vec_inds_list[i]], unique_indices=True))
        return res_sol.reshape(-1)
    return fn_dofs_row


def assign_bc(sol, problem):
    for i in range(len(problem.node_inds_list)):
        sol = sol.at[problem.node_inds_list[i], problem.vec_inds_list[i]].set(problem.vals_list[i])
    return sol


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
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_nodes*problem.vec))
    return J


def get_flatten_fn(fn_sol, problem):
    def fn_dofs(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        val_sol = fn_sol(sol)
        return val_sol.reshape(-1)
    return fn_dofs


@partial(jax.jit, static_argnums=(0,))
def linear_solver(problem):
    node_inds_list, vec_inds_list, vals_list = problem.node_inds_list, problem.vec_inds_list, problem.vals_list
    res_fn = problem.compute_residual
    sol = np.zeros((problem.num_total_nodes, problem.vec))
    dofs = sol.reshape(-1)
    dofs = assign_bc(sol, problem).reshape(-1)
    A_fn = apply_bc(res_fn, problem)
    b = -A_fn(dofs)
    A_fn_linear = get_A_fn_linear_fn(dofs, A_fn)
    inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab
    dofs = dofs + inc
    sol = dofs.reshape(sol.shape)
    return sol


def solver(problem, initial_guess=None, use_linearization_guess=True):
    res_fn = problem.compute_residual
    node_inds_list, vec_inds_list, vals_list = problem.node_inds_list, problem.vec_inds_list, problem.vals_list
    
    print("Start timing")
    start = time.time()

    if initial_guess is not None:
        sol = initial_guess
    else:
        sol = np.zeros((problem.num_total_nodes, problem.vec))

    linear_solve_step = 0
    # This seems to be a quite good initial guess
    # TODO: There's room for improvement.
    if use_linearization_guess:
        print("Solving a linearized problem to get a good initial guess...")
        dofs = sol.reshape(-1)
        res_fn_dofs = get_flatten_fn(res_fn, problem)
        res_fn_linear = get_A_fn_linear_fn(dofs, res_fn_dofs)
        res_fn_final = row_elimination(res_fn_linear, problem)
        b = -res_fn(sol)
        b = assign_bc(b, problem).reshape(-1)
        # print(f"step = 0, res l_2 = {np.linalg.norm(res_fn_final(assign_bc(sol).reshape(-1)))}") 
        dofs = assign_bc(sol, problem).reshape(-1)
        dofs, info = jax.scipy.sparse.linalg.bicgstab(res_fn_final, b, x0=dofs, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab
        linear_solve_step += 1
    else:
        dofs = assign_bc(sol, problem).reshape(-1)

    # Newton's method begins here.
    # If the problem is linear, the Newton's iteration will not be triggered.
    tol = 1e-6
    A_fn = apply_bc(res_fn, problem)
    b = -A_fn(dofs)
    res_val = np.linalg.norm(b)
    print(f"Before calling Newton's method, res l_2 = {res_val}") 
    while res_val > tol:
        A_fn_linear = get_A_fn_linear_fn(dofs, A_fn)
        debug = False
        if debug:
            # Check onditional number of the matrix
            A_dense = operator_to_matrix(A_fn_linear, problem)
            print(np.linalg.cond(A_dense))
            print(np.max(A_dense))
            print(A_dense)

        inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab
        linear_solve_step += 1
        dofs = dofs + inc
        b = -A_fn(dofs)
        res_val = np.linalg.norm(b)
        print(f"step = {linear_solve_step}, res l_2 = {res_val}") 

    sol = dofs.reshape(sol.shape)
    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s], finished in {linear_solve_step} linear solve steps")
    print(f"max of sol = {np.max(sol)}")
    print(f"min of sol = {np.min(sol)}")

    return sol
