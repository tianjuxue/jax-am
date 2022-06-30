import jraph
import jax
import jax.numpy as np
import numpy as onp
import meshio
import os
import glob
import time 
import pickle
from functools import partial
from scipy.spatial.transform import Rotation as R
from collections import namedtuple
from matplotlib import pyplot as plt
from src.arguments import args
from src.utils import get_unique_ori_colors, obj_to_vtu, walltime


# TODO: unique_oris_rgb and unique_grain_directions should be a class property, not an instance property
PolyCrystal = namedtuple('PolyCrystal', ['edges', 'ch_len', 'centroids', 'volumes', 'unique_oris_rgb', 
    'unique_grain_directions', 'cell_ori_inds', 'boundary_face_areas', 'boundary_face_centroids', 'meta_info'])

# gpus = jax.devices('gpu')
# @partial(jax.jit, static_argnums=(2,), device=gpus[-1])


@partial(jax.jit, static_argnums=(2,))
def rk4(state, t_crt, f, ode_params):
    '''
    Fourth order Runge-Kutta method
    We probably don't need this one.
    '''
    y_prev, t_prev = state
    h = t_crt - t_prev
    k1 = h * f(y_prev, t_prev, ode_params)
    k2 = h * f(y_prev + k1/2., t_prev + h/2., ode_params)
    k3 = h * f(y_prev + k2/2., t_prev + h/2., ode_params)
    k4 = h * f(y_prev + k3, t_prev + h, ode_params)
    y_crt = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (y_crt, t_crt), y_crt


@partial(jax.jit, static_argnums=(2,))
def explicit_euler(state, t_crt, f, ode_params):
    '''
    Explict Euler method
    '''
    y_prev, t_prev = state
    h = t_crt - t_prev
    y_crt = y_prev + h * f(y_prev, t_prev, ode_params)
    return (y_crt, t_crt), y_crt


@walltime
def odeint(polycrystal, mesh, get_T, stepper, f, y0, ts, ode_params):
    '''
    ODE integrator. 
    '''
    ys = [y0]
    clean_sols()
    state = (y0, ts[0])
    T = get_T(ts[0], ode_params)
    write_sols(polycrystal, mesh, y0, T, 0)
    for (i, t_crt) in enumerate(ts[1:]):
        state, y = stepper(state, t_crt, f, ode_params)
        # ys.append(y)
        T = get_T(t_crt, ode_params)
        if (i + 1) % 20 == 0:
            print(f"step {i + 1} of {len(ts[1:])}, unix timestamp = {time.time()}")
            # print(y[:10, :5])
            inspect_sol(y, y0, T)
            if not np.all(np.isfinite(y)):          
                raise ValueError(f"Found np.inf or np.nan in y - stop the program")
        write_sol_interval = args.write_sol_interval
        if (i + 1) % write_sol_interval == 0:
            write_sols(polycrystal, mesh, y, T, (i + 1) // write_sol_interval)

    write_info(polycrystal)
    return y, ys

 
def inspect_sol(y, y0, T):
    '''
    While running simulations, print out some useful information.
    '''
    eta0 = np.argmax(y0, axis=1)
    eta = np.argmax(y, axis=1)
    change_eta = np.where(eta0 == eta, 0, 1)
    change_T = np.where(T >= args.T_melt, 1, 0)
    print(f"percent of change of orientations = {np.sum(change_eta)/len(change_eta)*100}%")
    print(f"percet of T >= T_melt = {np.sum(change_T)/len(change_T)*100}%")
    print(f"max T = {np.max(T)}")
 

def clean_sols():
    '''
    Clean the data folder.
    '''
    vtk_folder = f"data/vtk/{args.case}/sols"
    numpy_folder = f"data/numpy/{args.case}/sols"
    files_vtk = glob.glob(vtk_folder + f"/*")
    files_numpy = glob.glob(numpy_folder + f"/*")
    files = files_vtk + files_numpy
    for f in files:
        os.remove(f)


def write_info(polycrystal):
    '''
    Mostly for post-processing. E.g., compute grain volume, aspect ratios, etc.
    '''
    if not args.case.startswith('gn_multi_layer'):
        onp.save(f"data/numpy/{args.case}/info/edges.npy", polycrystal.edges)
        onp.save(f"data/numpy/{args.case}/info/vols.npy", polycrystal.volumes)
        onp.save(f"data/numpy/{args.case}/info/centroids.npy", polycrystal.centroids)


def write_sols_heper(polycrystal, mesh, y, T):
   
    zeta = T.reshape(-1) < args.T_melt
    eta = y
    eta_max = onp.max(eta, axis=1)
    cell_ori_inds = onp.argmax(eta, axis=1)
    ipf_x = onp.take(polycrystal.unique_oris_rgb[0], cell_ori_inds, axis=0)
    ipf_y = onp.take(polycrystal.unique_oris_rgb[1], cell_ori_inds, axis=0)
    ipf_z = onp.take(polycrystal.unique_oris_rgb[2], cell_ori_inds, axis=0)

    # TODO: Is this better?
    ipf_x[zeta < 0.1] = 0.
    ipf_y[zeta < 0.1] = 0.
    ipf_z[zeta < 0.1] = 0.

    mesh.cell_data['T'] = [onp.array(T, dtype=onp.float32)]
    mesh.cell_data['ipf_x'] = [ipf_x]
    mesh.cell_data['ipf_y'] = [ipf_y]
    mesh.cell_data['ipf_z'] = [ipf_z]
    cell_ori_inds = onp.array(cell_ori_inds, dtype=onp.int32)
    mesh.cell_data['ori_inds'] = [cell_ori_inds]

    return T, cell_ori_inds


def write_sols(polycrystal, mesh, y, T, step):
    '''
    Use Paraview to open .vtu files for visualization of:
    1. Temeperature field (T)
    2. Liquid/Solid phase (zeta)
    3. Grain orientations (eta)
    '''
    print(f"Write sols to file...")
    T, cell_ori_inds = write_sols_heper(polycrystal, mesh, y, T)
    onp.save(f"data/numpy/{args.case}/sols/T_{step:03d}.npy", T)
    onp.save(f"data/numpy/{args.case}/sols/cell_ori_inds_{step:03d}.npy", cell_ori_inds)
    mesh.write(f"data/vtk/{args.case}/sols/u{step:03d}.vtu")
 


def polycrystal_fd(domain_name='single_layer'):
    '''
    Prepare graph information for finite difference method
    '''
    filepath = f'data/neper/{domain_name}/domain.msh'
    mesh = meshio.read(filepath)
    points = mesh.points
    cells =  mesh.cells_dict['hexahedron']
    cell_grain_inds = mesh.cell_data['gmsh:physical'][0] - 1
    onp.save(f"data/numpy/{args.case}/info/cell_grain_inds.npy", cell_grain_inds)
    assert args.num_grains == onp.max(cell_grain_inds) + 1

    unique_oris_rgb, unique_grain_directions = get_unique_ori_colors()
    grain_oris_inds = onp.random.randint(args.num_oris, size=args.num_grains)
    cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)

    Nx = round(args.domain_length / points[1, 0])
    Ny = round(args.domain_width / points[Nx + 1, 1])
    Nz = round(args.domain_height / points[(Nx + 1)*(Ny + 1), 2])
    args.Nx = Nx
    args.Ny = Ny
    args.Nz = Nz

    print(f"Total num of grains = {args.num_grains}")
    print(f"Total num of orientations = {args.num_oris}")
    print(f"Total num of finite difference cells = {len(cells)}")
    assert Nx*Ny*Nz == len(cells)

    edges = []
    for i in range(Nx):
        if i % 100 == 0:
            print(f"i = {i}")
        for j in range(Ny):
            for k in range(Nz):
                crt_ind = i + j * Nx + k * Nx * Ny
                if i != Nx - 1:
                    edges.append([crt_ind, (i + 1) + j * Nx + k * Nx * Ny])
                if j != Ny - 1:
                    edges.append([crt_ind, i + (j + 1) * Nx + k * Nx * Ny])
                if k != Nz - 1:
                    edges.append([crt_ind, i + j * Nx + (k + 1) * Nx * Ny])

    edges = onp.array(edges)
    cell_points = onp.take(points, cells, axis=0)
    centroids = onp.mean(cell_points, axis=1)
    domain_vol = args.domain_length*args.domain_width*args.domain_height
    volumes = domain_vol / (Nx*Ny*Nz) * onp.ones(len(cells))
    ch_len = (domain_vol / len(cells))**(1./3.) * onp.ones(len(edges))

    face_inds = [[0, 3, 4, 7], [1, 2, 5, 6], [0, 1, 4, 5], [2, 3, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]]
    boundary_face_centroids = onp.transpose(onp.stack([onp.mean(onp.take(cell_points, face_ind, axis=1), axis=1) 
        for face_ind in face_inds]), axes=(1, 0, 2))
    
    boundary_face_areas = []
    domain_measures = [args.domain_length, args.domain_width, args.domain_height]
    face_cell_nums = [Ny*Nz, Nx*Nz, Nx*Ny]
    for i, domain_measure in enumerate(domain_measures):
        cell_area = domain_vol/domain_measure/face_cell_nums[i]
        boundary_face_area1 = onp.where(onp.isclose(boundary_face_centroids[:, 2*i, i], 0., atol=1e-08), cell_area, 0.)
        boundary_face_area2 = onp.where(onp.isclose(boundary_face_centroids[:, 2*i + 1, i], domain_measure, atol=1e-08), cell_area, 0.)
        boundary_face_areas += [boundary_face_area1, boundary_face_area2]

    boundary_face_areas = onp.transpose(onp.stack(boundary_face_areas))

    meta_info = onp.array([0., 0., 0., args.domain_length, args.domain_width, args.domain_height])
    polycrystal = PolyCrystal(edges, ch_len, centroids, volumes, unique_oris_rgb, unique_grain_directions,
                              cell_ori_inds, boundary_face_areas, boundary_face_centroids, meta_info)

    return polycrystal, mesh


def phase_field(polycrystal):

    centroids = polycrystal.centroids
    # TODO: make this simpler
    mesh_h = polycrystal.ch_len[0]


    # TODO: consider anisotropic growth
    def update_anisotropy():

        sender_centroids = np.take(centroids, graph.senders, axis=0)
        receiver_centroids = np.take(centroids, graph.receivers, axis=0)
        edge_directions = sender_centroids - receiver_centroids
        edge_directions = np.repeat(edge_directions[:, None, :], args.num_oris, axis=1) # (num_edges, num_oris, dim)
 
        unique_grain_directions = polycrystal.unique_grain_directions # (num_directions_per_cube, num_oris, dim)

        assert edge_directions.shape == (len(graph.senders), args.num_oris, args.dim)
        cosines = np.sum(unique_grain_directions[None, :, :, :] * edge_directions[:, None, :, :], axis=-1) \
                  / (np.linalg.norm(edge_directions, axis=-1)[:, None, :])
        anlges =  np.arccos(cosines) 
        anlges = np.where(np.isfinite(anlges), anlges, 0.)
        anlges = np.where(anlges < np.pi/2., anlges, np.pi - anlges)
        anlges = np.min(anlges, axis=1)

        anisotropy_term = 1. + args.anisotropy * (np.cos(anlges)**4 + np.sin(anlges)**4) # (num_edges, num_oris)

        assert anisotropy_term.shape == (len(graph.senders), args.num_oris)
        graph.edges['anisotropy'] = anisotropy_term
        print("End of compute_anisotropy...")

    def get_T(t, ode_params):
        '''
        Analytic T from https://doi.org/10.1016/j.actamat.2021.116862
        '''
        Q, alpha = ode_params
        # Q = 25
        # alpha = 5.2

        T_ambiant = 300.

        kappa = 2.7*1e-2
        x0 = 0.2*args.domain_length

        vel = 0.6/0.0024

        X = centroids[:, 0] - x0 - vel * t
        Y = centroids[:, 1] - 0.5*args.domain_width
        Z = centroids[:, 2] - args.domain_height
        R = np.sqrt(X**2 + Y**2 + Z**2)
        T = T_ambiant + Q / (2 * np.pi * kappa) / R * np.exp(-vel / (2*alpha) * (R + X))

        # TODO
        T = np.where(T > 2000., 2000., T)

        return T[:, None]

    def local_energy_fn(eta, zeta):
        gamma = 1
        vmap_outer = jax.vmap(np.outer, in_axes=(0, 0))
        grain_energy_1 = np.sum((eta**4/4. - eta**2/2.))
        graph_energy_2 = gamma * (np.sum(vmap_outer(eta, eta)**2) - np.sum(eta**4))  
        graph_energy_3 = np.sum((1 - zeta.reshape(-1))**2 * np.sum(eta**2, axis=1).reshape(-1))
        grain_energy = args.m_g * (grain_energy_1 +  graph_energy_2 + graph_energy_3)
        return grain_energy

    local_energy_grad_fn = jax.grad(local_energy_fn, argnums=0) 

    def state_rhs(state, t, ode_params):
        eta = state
        T = get_T(t, ode_params)
        zeta = 0.5 * (1 - np.tanh(1e1*(T/args.T_melt - 1)))
        local_energy_grad = local_energy_grad_fn(eta, zeta) / args.ad_hoc
        # TODO: concatenate is slow, any workaround?
        eta_xyz = np.reshape(eta, (args.Nz, args.Ny, args.Nx, args.num_oris))
        eta_neg_x = np.concatenate((eta_xyz[:, :, :1, :], eta_xyz[:, :, :-1, :]), axis=2)
        eta_pos_x = np.concatenate((eta_xyz[:, :, 1:, :], eta_xyz[:, :, -1:, :]), axis=2)
        eta_neg_y = np.concatenate((eta_xyz[:, :1, :, :], eta_xyz[:, :-1, :, :]), axis=1)
        eta_pos_y = np.concatenate((eta_xyz[:, 1:, :, :], eta_xyz[:, -1:, :, :]), axis=1)
        eta_neg_z = np.concatenate((eta_xyz[:1, :, :, :], eta_xyz[:-1, :, :, :]), axis=0)
        eta_pos_z = np.concatenate((eta_xyz[1:, :, :, :], eta_xyz[-1:, :, :, :]), axis=0)
        # See https://en.wikipedia.org/wiki/Finite_difference "Second-order central"
        laplace_xyz = -np.stack((eta_pos_x - 2*eta_xyz + eta_neg_x, 
                                 eta_pos_y - 2*eta_xyz + eta_neg_y, 
                                 eta_pos_z - 2*eta_xyz + eta_neg_z), axis=-1) / mesh_h**2 * args.kappa_g * args.ad_hoc
        assert laplace_xyz.shape == (args.Nz, args.Ny, args.Nx, args.num_oris, args.dim)
        laplace = np.sum(laplace_xyz.reshape(-1, args.num_oris, args.dim), axis=-1)
        assert local_energy_grad.shape == laplace.shape
        Lg = args.L0 * np.exp(-args.Qg / (T*args.gas_const))
        rhs = -Lg * (local_energy_grad + laplace)
        return rhs

    return state_rhs, get_T
