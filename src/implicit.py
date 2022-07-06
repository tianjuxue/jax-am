import jax
import jax.numpy as np
from jaxopt import linear_solve
import numpy as onp
import os
import meshio
import time
from src.integrator import MultiVarSolver
from src.utils import Field
from src.yaml_parse import args


def lap_operator(left_inds, right_inds):

    # @jax.jit
    def operator_fn(x):
        print(f"callback")
        state_xyz = np.reshape(x, (args['Nz'], args['Ny'], args['Nx']))
        state_neg_x = np.concatenate((state_xyz[:, :, :1], state_xyz[:, :, :-1]), axis=2)
        state_pos_x = np.concatenate((state_xyz[:, :, 1:], state_xyz[:, :, -1:]), axis=2)
        state_neg_y = np.concatenate((state_xyz[:, :1, :], state_xyz[:, :-1, :]), axis=1)
        state_pos_y = np.concatenate((state_xyz[:, 1:, :], state_xyz[:, -1:, :]), axis=1)
        state_neg_z = np.concatenate((state_xyz[:1, :, :], state_xyz[:-1, :, :]), axis=0)
        state_pos_z = np.concatenate((state_xyz[1:, :, :], state_xyz[-1:, :, :]), axis=0)
        # See https://en.wikipedia.org/wiki/Finite_difference "Second-order central"
        laplace_xyz = -np.stack((state_pos_x - 2*state_xyz + state_neg_x, 
                                 state_pos_y - 2*state_xyz + state_neg_y, 
                                 state_pos_z - 2*state_xyz + state_neg_z), axis=-1) / np.array([args['hx'], args['hy'], args['hz']])[None, None, None, :]

        assert laplace_xyz.shape == (args['Nz'], args['Ny'], args['Nx'], args['dim'])
        laplace = np.sum(laplace_xyz.reshape(-1, args['dim']), axis=-1)

        laplace = laplace.at[left_inds].set(x[left_inds], unique_indices=True)
        laplace = laplace.at[right_inds].set(x[right_inds], unique_indices=True)

        return laplace


    def precond(x):
        y = 1. / (2./args['hx'] + 2./args['hy'] + 2./args['hz']) * x
        y = y.at[left_inds].set(x[left_inds], unique_indices=True)
        y = y.at[right_inds].set(x[right_inds], unique_indices=True)
        return y     


    def operator_fn1(x):
        x = x.at[0:10].set(10*x[0:10])
        return 2*x

    return operator_fn, precond


def run():
    args['case'] = 'fd_example'

    if args['case'] == 'implicit':
        args['num_grains'] = 1000
        args['domain_x'] = 0.1
        args['domain_y'] = 0.1
        args['domain_z'] = 0.1
    elif args['case'] == 'fd_example':
        args['domain_x'] = 1.
        args['domain_y'] = 0.2
        args['domain_z'] = 0.1       

    neper = False
    if neper:
        os.system(f'''neper -T -n {args['num_grains']} -id 1 -regularization 0 -domain "cube({args['domain_x']},\
                   {args['domain_y']},{args['domain_z']})" \
                    -o post-processing/neper/{args['case']}/domain -format tess,obj,ori''')
        os.system(f"neper -T -loadtess post-processing/neper/{args['case']}/domain.tess -statcell x,y,z,vol,facelist -statface x,y,z,area")
        os.system(f"neper -M -rcl 1 -elttype hex -faset faces post-processing/neper/{args['case']}/domain.tess")

    filepath = f"post-processing/neper/{args['case']}/domain.msh"
    mesh = meshio.read(filepath)
    points = mesh.points
    cells =  mesh.cells_dict['hexahedron']
    cell_points = onp.take(points, cells, axis=0)
    centroids = onp.mean(cell_points, axis=1)

    hx = points[1, 0]
    Nx = round(args['domain_x'] / hx)
    hy = points[Nx + 1, 1]
    Ny = round(args['domain_y'] / hy)
    hz = points[(Nx + 1)*(Ny + 1), 2]
    Nz = round(args['domain_z'] / hz)
    args['Nx'] = Nx
    args['Ny'] = Ny
    args['Nz'] = Nz
    args['hx'] = hx
    args['hy'] = hy
    args['hz'] = hz
    print(f"Nx = {Nx}, Ny = {Ny}, Nz = {Nz}, hx = {hx}, hy = {hy}, hz = {hz}, num_cell = {len(cells)}")
    EPS = 1e-8


    args['hx'] = 1
    args['hy'] = 1
    args['hz'] = 1

    print(f"Total num of finite difference cells = {len(cells)}")
    assert Nx*Ny*Nz == len(cells)

    b_vec = 10*np.ones(len(cells))

    left_inds = onp.argwhere(centroids[:, 0] < hx + EPS).reshape(-1)
    right_inds = onp.argwhere(centroids[:, 0] >  args['domain_x'] - (hx + EPS)).reshape(-1)

    b_vec = b_vec.at[left_inds].set(0.)
    b_vec = b_vec.at[right_inds].set(0.)

    mesh.cell_data['rhs'] = [onp.array(b_vec, dtype=onp.float32)]

    A, precond = lap_operator(left_inds, right_inds)

    start = time.time()
    # sol = linear_solve.solve_bicgstab(A, b_vec, tol=1e-10)
    sol, info = jax.scipy.sparse.linalg.bicgstab(A, b_vec, x0=None, M=precond, tol=1e-10, atol=1e-10, maxiter=10000) # gmres, bicgstab, cg
    end = time.time()
    print(f"First solve took {end - start}")

    mesh.cell_data['sol'] = [onp.array(sol, dtype=onp.float32)]

    # print(A(sol) - b_vec)
    print(f"res l_2 = {np.linalg.norm(A(sol) - b_vec)}")  
    print(f"res l_inf = {np.max(np.absolute(A(sol) - b_vec))}")  

    # mesh.write(f"post-processing/vtk/implicit/rhs.vtu")


if __name__ == "__main__":
    run()
 
