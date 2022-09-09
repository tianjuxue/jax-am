import time
import jax
import jax.numpy as np


def get_A_fn_linear_fn(dofs, fn):
# get Jacobian-vector-product function from residual function
    def A_fn_linear_fn(inc):
        primals, tangents = jax.jvp(fn, (dofs,), (inc,))
        return tangents
    return A_fn_linear_fn


def solver_linear(A_fn,dofs,tol=1e-5):
# solve linear problems
# input: A_fun - residual function
#         dofs - inital guess
    b = -A_fn(dofs)
    A_fn_linear = get_A_fn_linear_fn(dofs, A_fn)
    inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, tol=tol)
    dofs = dofs + inc
    
    res_norm = np.linalg.norm(A_fn(dofs))/np.linalg.norm(b)
    return dofs,res_norm


def solver_nonlinear(A_fn,dofs,tol=1e-5):
# solve nonlinear problems
    def cond_fun(dofs):
        b = -A_fn(dofs)
        res_val = np.linalg.norm(b)
        return res_val > tol

    def body_fun(dofs):
        b = -A_fn(dofs)
        A_fn_linear = get_A_fn_linear_fn(dofs, A_fn)
        inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b)
        dofs = dofs + inc
        return dofs
    dofs = jax.lax.while_loop(cond_fun, body_fun, dofs)
    return dofs