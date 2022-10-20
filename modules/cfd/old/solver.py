import time
import jax
import jax.numpy as np


def get_A_fn_linear_fn(dofs, fn):
# get Jacobian-vector-product function from residual function
    def A_fn_linear_fn(inc):
        primals, tangents = jax.jvp(fn, (dofs,), (inc,))
        return tangents
    return A_fn_linear_fn


def solver_linear(A_fn,dofs,tol=1e-6):
# solve linear problems
# input: A_fun - residual function
#         dofs - inital guess
    b = -A_fn(dofs)
    A_fn_linear = get_A_fn_linear_fn(dofs, A_fn)
    inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, tol=tol)
    dofs = dofs + inc
    
    res_norm = np.linalg.norm(A_fn(dofs))/np.linalg.norm(b)
    return dofs,res_norm


# def solver_linear_debug(A_fn,dofs,tol=1e-5):
# # solve linear problems: no jvp
# # input: A_fun - residual function
# #         dofs - inital guess
#     b = -A_fn(dofs)
#     A_fn_linear = lambda x: A_fn(x) + b
    
#     dofs, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, tol=tol)
    
#     res_norm = np.linalg.norm(A_fn(dofs))/np.linalg.norm(b)
#     return dofs,res_norm


def solver_linear_cg(A_fn,dofs,tol=1e-5):
# solve symmetric linear problems
# input: A_fun - residual function
#         dofs - inital guess
    b = -A_fn(dofs)
    A_fn_linear = get_A_fn_linear_fn(dofs, A_fn)
    inc, info = jax.scipy.sparse.linalg.cg(A_fn_linear, b, tol=tol)
    dofs = dofs + inc
    
    res_norm = np.linalg.norm(A_fn(dofs))/np.linalg.norm(b)
    return dofs,res_norm


def solver_nonlinear(A_fn,dofs,tol=1e-5,max_it=1000,alpha=1.):
# solve nonlinear problems

    def cond_fun(carry):
        dofs,inc,it = carry
        return (np.linalg.norm(inc)/np.linalg.norm(dofs) > tol) & (it < max_it)

    def body_fun(carry):
        dofs,inc,it = carry
        b = -A_fn(dofs)
        A_fn_linear = get_A_fn_linear_fn(dofs, A_fn)
        inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b)
        dofs = dofs + inc*alpha
        return (dofs,inc,it+1)
    
    it = 0
    inc = np.ones_like(dofs)
    dofs,inc,it = jax.lax.while_loop(cond_fun, body_fun, (dofs,inc,it))
    
    return dofs,it