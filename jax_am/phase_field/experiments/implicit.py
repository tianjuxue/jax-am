import jax
import jax.numpy as np
from jaxopt import linear_solve
import numpy as onp
import scipy
import os
import meshio
import time
import sys

from jax.config import config
config.update("jax_enable_x64", True)

onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=200, suppress=True)
onp.set_printoptions(precision=10)


args = {} 


def onp_operator(left_inds, right_inds):

    def operator_fn(x):
        print(f"callback")
        state_xyz = onp.reshape(x, (args['Nz'], args['Ny'], args['Nx']), order='F')

        state_neg_x = onp.concatenate((state_xyz[:, :, :1], state_xyz[:, :, :-1]), axis=2)
        state_pos_x = onp.concatenate((state_xyz[:, :, 1:], state_xyz[:, :, -1:]), axis=2)
        state_neg_y = onp.concatenate((state_xyz[:, :1, :], state_xyz[:, :-1, :]), axis=1)
        state_pos_y = onp.concatenate((state_xyz[:, 1:, :], state_xyz[:, -1:, :]), axis=1)
        state_neg_z = onp.concatenate((state_xyz[:1, :, :], state_xyz[:-1, :, :]), axis=0)
        state_pos_z = onp.concatenate((state_xyz[1:, :, :], state_xyz[-1:, :, :]), axis=0)

        # See https://en.wikipedia.org/wiki/Finite_difference "Second-order central"
        laplace_xyz = -onp.stack((state_pos_x - 2*state_xyz + state_neg_x, 
                                 state_pos_y - 2*state_xyz + state_neg_y, 
                                 state_pos_z - 2*state_xyz + state_neg_z), axis=-1) / onp.array([args['hx'], args['hy'], args['hz']])[None, None, None, :]**2

        assert laplace_xyz.shape == (args['Nz'], args['Ny'], args['Nx'], args['dim'])
        laplace = onp.sum(laplace_xyz.reshape(-1, args['dim'], order='F'), axis=-1)

        laplace[left_inds] = x[left_inds]
        laplace[right_inds] = x[right_inds]

        return laplace

    A = scipy.sparse.linalg.LinearOperator((args['num_nodes'], args['num_nodes']), matvec=operator_fn)

    return A


def lap_operator(left_inds, right_inds):

    # @jax.jit
    def operator_fn(x):
        print(f"callback")
        state_xyz = np.reshape(x, (args['Nz'], args['Ny'], args['Nx']), order='F')

        state_neg_x = np.concatenate((state_xyz[:, :, :1], state_xyz[:, :, :-1]), axis=2)
        state_pos_x = np.concatenate((state_xyz[:, :, 1:], state_xyz[:, :, -1:]), axis=2)
        state_neg_y = np.concatenate((state_xyz[:, :1, :], state_xyz[:, :-1, :]), axis=1)
        state_pos_y = np.concatenate((state_xyz[:, 1:, :], state_xyz[:, -1:, :]), axis=1)
        state_neg_z = np.concatenate((state_xyz[:1, :, :], state_xyz[:-1, :, :]), axis=0)
        state_pos_z = np.concatenate((state_xyz[1:, :, :], state_xyz[-1:, :, :]), axis=0)

        # See https://en.wikipedia.org/wiki/Finite_difference "Second-order central"
        laplace_xyz = -np.stack((state_pos_x - 2*state_xyz + state_neg_x, 
                                 state_pos_y - 2*state_xyz + state_neg_y, 
                                 state_pos_z - 2*state_xyz + state_neg_z), axis=-1) / np.array([args['hx'], args['hy'], args['hz']])[None, None, None, :]**2

        assert laplace_xyz.shape == (args['Nz'], args['Ny'], args['Nx'], args['dim'])
        laplace = np.sum(laplace_xyz.reshape(-1, args['dim'], order='F'), axis=-1)

        laplace = laplace.at[left_inds].set(x[left_inds], unique_indices=True)
        laplace = laplace.at[right_inds].set(x[right_inds], unique_indices=True)

        return laplace

    def precond(x):
        y = 1. / (2./args['hx']**2 + 2./args['hy']**2 + 2./args['hz']**2) * x
        y = y.at[left_inds].set(x[left_inds], unique_indices=True)
        y = y.at[right_inds].set(x[right_inds], unique_indices=True)
        return y     

    def operator_fn1(x):
        x = x.at[0:10].set(10*x[0:10])
        return 2*x

    return operator_fn, precond


def operator_to_matrix(operator_fn):
    J = jax.jacfwd(operator_fn)(np.zeros(args['num_nodes']))
    return J


def run():
    domain_x = 1.
    domain_y = 1.
    domain_z = 1.
    Nx = 100
    Ny = 100
    Nz = 100
    num_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)
    hx = domain_x / Nx
    hy = domain_y / Ny
    hz = domain_z / Nz
    x = np.linspace(0., domain_x, Nx + 1)
    y = np.linspace(0., domain_y, Ny + 1)
    z = np.linspace(0., domain_z, Nz + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T

    args['dim'] = 3
    args['Nx'] = Nx + 1
    args['Ny'] = Ny + 1
    args['Nz'] = Nz + 1
    args['hx'] = hx
    args['hy'] = hy
    args['hz'] = hz

    args['num_nodes'] = num_nodes
    print(f"Nx = {Nx}, Ny = {Ny}, Nz = {Nz}, hx = {hx}, hy = {hy}, hz = {hz}")
    print(f"Total num of finite difference nodes = {num_nodes}")

    EPS = 1e-5
    left_inds = onp.argwhere(points[:, 0] < EPS).reshape(-1)
    right_inds = onp.argwhere(points[:, 0] >  domain_x - EPS).reshape(-1)

    onp_scipy = False
    if onp_scipy:
        b = 10*onp.ones(args['num_nodes'])
        b[left_inds] = 0.
        b[right_inds] = 0.
        A_fn = onp_operator(left_inds, right_inds)
        start = time.time()
        sol, info = scipy.sparse.linalg.cg(A_fn, b)
        end = time.time()
        print(f"First solve took {end - start}")
        print(info)
     
        print(f"res l_2 = {np.linalg.norm(A_fn(sol) - b)}")  
        print(f"res l_inf = {np.max(np.absolute(A_fn(sol) - b))}")  
        print(f"max of sol = {np.max(sol)}")


    b = 10*np.ones(num_nodes)
    b = b.at[left_inds].set(0.)
    b = b.at[right_inds].set(0.)

    A_fn, precond = lap_operator(left_inds, right_inds)


    jax_direct = False
    if jax_direct:
        A = operator_to_matrix(A_fn)

        M = np.linalg.inv(np.diag(np.diag(A)))

        # M2 = operator_to_matrix(precond)

        MA = np.matmul(M, A)

        # print(A)
        # print(M)
        print(f"condtion number of A: {np.linalg.cond(A)}")
        print(f"condtion number of M: {np.linalg.cond(M)}")
        print(f"condtion number of MA: {np.linalg.cond(MA)}")

        sol = jax.scipy.linalg.solve(A, b)

        print(f"res l_2 = {np.linalg.norm(np.matmul(A, sol) - b)}")  
        print(f"res l_inf = {np.max(np.absolute(np.matmul(A, sol) - b))}")  
        print(f"max of sol = {np.max(sol)}")


    jax_iterative = True
    if jax_iterative:
        for i in range(1):
            start = time.time()
            # sol = linear_solve.solve_bicgstab(A, b, tol=1e-10)
            sol, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # gmres, bicgstab, cg
            end = time.time()
            print(f"The {i + 1}th solve took {end - start}")

        print(f"res l_2 = {np.linalg.norm(A_fn(sol) - b)}")  
        print(f"res l_inf = {np.max(np.absolute(A_fn(sol) - b))}")  
        print(f"max of sol = {np.max(sol)}")

    # mesh.cell_data['sol'] = [onp.array(sol, dtype=onp.float32)]
    # mesh.cell_data['rhs'] = [onp.array(b, dtype=onp.float32)]
    # mesh.write(f"post-processing/vtk/implicit/rhs.vtu")


if __name__ == "__main__":
    run()
