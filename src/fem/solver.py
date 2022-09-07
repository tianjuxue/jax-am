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


def assign_zero_bc(sol, problem):
    for i in range(len(problem.p_node_inds_list_B)):
        sol = sol.at[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]].set(0.)
    return sol


######################################################################################
# Tyring impoising periodic B.C. like Dirichlet B.C., not working.
# Leave these functions here in case there is more insight in the future.
# One way for example is https://fenics2021.com/slides/dokken.pdf
# Currently periodic B.C. is implemented using Lagragian multiplier (not in this module).

def periodic_apply_bc_before(res_fn, problem):
    """Helper function. Not working.
    """
    def fn_dofs_row(sol):
        for i in range(len(problem.p_node_inds_list_B)):
            sol = (sol.at[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]].set
                  (sol[problem.p_node_inds_list_A[i], problem.p_vec_inds_list[i]], unique_indices=True))
        res = res_fn(sol)
        return res
    return fn_dofs_row


def periodic_apply_bc_after(fn_dofs, problem):
    """Helper function. Not working.
    """
    def fn_dofs_row(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        res_dofs = fn_dofs(dofs)
        res_sol = res_dofs.reshape((problem.num_total_nodes, problem.vec))
        for i in range(len(problem.p_node_inds_list_B)):
            res_sol = (res_sol.at[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]].set
                      (sol[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]], unique_indices=True))
            res_sol = (res_sol.at[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]].add
                      (-sol[problem.p_node_inds_list_A[i], problem.p_vec_inds_list[i]], unique_indices=True))
        return res_sol.reshape(-1)
    return fn_dofs_row


def periodic_apply_bc_penalty(fn_dofs, problem):
    """Penaly approach. Not working.
    """
    def fn_dofs_row(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        res_dofs = fn_dofs(dofs)
        res_sol = res_dofs.reshape((problem.num_total_nodes, problem.vec))
        for i in range(len(problem.p_node_inds_list_B)):
            alpha = 1e1
            sol_A = sol[problem.p_node_inds_list_A[i], problem.p_vec_inds_list[i]]
            sol_B = sol[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]]
            res_sol = (res_sol.at[problem.p_node_inds_list_A[i], problem.p_vec_inds_list[i]].add
                      (alpha*(sol_A - sol_B), unique_indices=True))
            res_sol = (res_sol.at[problem.p_node_inds_list_B[i], problem.p_vec_inds_list[i]].add
                      (alpha*(sol_B - sol_A), unique_indices=True))
        return res_sol.reshape(-1)
    return fn_dofs_row

# End periodic B.C.
######################################################################################


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


def get_flatten_fn(fn_sol, problem):
    def fn_dofs(dofs):
        sol = dofs.reshape((problem.num_total_nodes, problem.vec))
        val_sol = fn_sol(sol)
        return val_sol.reshape(-1)
    return fn_dofs


@partial(jax.jit, static_argnums=(0,))
def linear_solver(problem):
    """Exp with external jit and see if that makes the solve faster. Seems not...
    """
    # node_inds_list, vec_inds_list, vals_list = problem.node_inds_list, problem.vec_inds_list, problem.vals_list
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
    print("Start timing")
    start = time.time()

    res_fn = problem.compute_residual

    if initial_guess is not None:
        sol = initial_guess
    else:
        sol = np.zeros((problem.num_total_nodes, problem.vec))

    linear_solve_step = 0
    # This seems to be a quite good initial guess
    if use_linearization_guess:
        print("Solving a linearized problem to get a good initial guess...")
        dofs = sol.reshape(-1)
        res_fn_dofs = get_flatten_fn(res_fn, problem)
        res_fn_linear = get_A_fn_linear_fn(dofs, res_fn_dofs)
        res_fn_final = row_elimination(res_fn_linear, problem)
        b = -res_fn(sol)
        b = assign_bc(b, problem)
        # print(f"step = 0, res l_2 = {np.linalg.norm(res_fn_final(assign_bc(sol).reshape(-1)))}") 
        dofs = assign_bc(sol, problem).reshape(-1)
        dofs, info = jax.scipy.sparse.linalg.bicgstab(res_fn_final, b.reshape(-1), x0=dofs, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab
        linear_solve_step += 1
    else:
        dofs = assign_bc(sol, problem).reshape(-1)

    # Newton's method begins here.
    # If the problem is linear, the Newton's iteration will not be triggered.
    A_fn = apply_bc(res_fn, problem)
    if problem.periodic_bc_info is not None:
        A_fn = periodic_apply_bc_after(A_fn, problem)

    b = -A_fn(dofs)
    res_val = np.linalg.norm(b)
    print(f"Before calling Newton's method, res l_2 = {res_val}") 
    tol = 1e-6
    while res_val > tol:
        A_fn_linear = get_A_fn_linear_fn(dofs, A_fn)
        debug = False
        if debug:
            # Check onditional number of the matrix
            A_dense = operator_to_matrix(A_fn_linear, problem)
            print(f"conditional number = {np.linalg.cond(A_dense)}")
            print(f"max A = {np.max(A_dense)}")
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
