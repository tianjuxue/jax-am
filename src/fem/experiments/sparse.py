from scipy import sparse
import numpy as onp
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
from src.fem.experiments import sparsejac

from jax.config import config
config.update("jax_enable_x64", True)



def test():
    I = onp.array([0,3,1,0])
    J = onp.array([0,3,1,2])
    V = onp.array([4.,5,7,9])
    A = sparse.coo_matrix((V,(I,J)),shape=(4,4))

    A_sp = BCOO.from_scipy_sparse(A)

    print(A_sp.data)

    fn = lambda x: x**2
    sparsity = BCOO.fromdense(np.eye(10000))
    x = jax.random.uniform(jax.random.PRNGKey(0), shape=(10000,))

    sparse_fn = jax.jit(sparsejac.jacrev(fn, sparsity))

    print(f"Finished JIT")

    J = sparse_fn(x)
    print(J)


if __name__=="__main__":
    test()
