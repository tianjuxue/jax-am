import jax.numpy as np
import jax
import numpy as onp
import orix
import meshio
import pickle
import time
import os
import matplotlib.pyplot as plt
from orix import plot, sampling
from orix.crystal_map import Phase
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from src.arguments import args
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R


def get_unique_ori_colors():
    onp.random.seed(1)

    ori2 = Orientation.random(args.num_oris)        

    vx = Vector3d((1, 0, 0))
    vy = Vector3d((0, 1, 0))
    vz = Vector3d((0, 0, 1))
    ipfkey_x = plot.IPFColorKeyTSL(symmetry.Oh, vx)
    rgb_x = ipfkey_x.orientation2color(ori2)
    ipfkey_y = plot.IPFColorKeyTSL(symmetry.Oh, vy)
    rgb_y = ipfkey_y.orientation2color(ori2)
    ipfkey_z = plot.IPFColorKeyTSL(symmetry.Oh, vz)
    rgb_z = ipfkey_z.orientation2color(ori2)
    rgb = onp.stack((rgb_x, rgb_y, rgb_z))

    onp.save(f"data/numpy/quat.npy", ori2.data)
    dx = onp.array([1., 0., 0.])
    dy = onp.array([0., 1., 0.])
    dz = onp.array([0., 0., 1.])
    scipy_quat = onp.concatenate((ori2.data[:, 1:], ori2.data[:, :1]), axis=1)
    r = R.from_quat(scipy_quat)
    grain_directions = onp.stack((r.apply(dx), r.apply(dy), r.apply(dz)))

    save_ipf = True
    if save_ipf:
        # Plot IPF for those orientations
        new_params = {
            "figure.facecolor": "w",
            "figure.figsize": (6, 3),
            "lines.markersize": 10,
            "font.size": 20,
            "axes.grid": True,
        }
        plt.rcParams.update(new_params)
        ori2.symmetry = symmetry.Oh
        ori2.scatter("ipf", c=rgb_x, direction=ipfkey_x.direction)
        # plt.savefig(f'data/pdf/ipf_x.pdf', bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_y, direction=ipfkey_y.direction)
        # plt.savefig(f'data/pdf/ipf_y.pdf', bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_z, direction=ipfkey_z.direction)
        # plt.savefig(f'data/pdf/ipf_z.pdf', bbox_inches='tight')

    return rgb, grain_directions


def obj_to_vtu(domain_name):
    filepath=f'data/neper/{domain_name}/domain.obj'
    file = open(filepath, 'r')
    lines = file.readlines()
    points = []
    cells_inds = []

    for i, line in enumerate(lines):
        l = line.split()
        if l[0] == 'v':
            points.append([float(l[1]), float(l[2]), float(l[3])])
        if l[0] == 'g':
            cells_inds.append([])
        if l[0] == 'f':
            cells_inds[-1].append([int(pt_ind) - 1 for pt_ind in l[1:]])

    cells = [('polyhedron', cells_inds)]
    mesh = meshio.Mesh(points, cells)
    return mesh


def walltime(func):
    def wrapper(*list_args, **keyword_args):
        start_time = time.time()
        return_values = func(*list_args, **keyword_args)
        end_time = time.time()
        time_elapsed = end_time - start_time
        platform = jax.lib.xla_bridge.get_backend().platform
        print(f"Time elapsed {time_elapsed} on platform {platform}") 
        with open(f'data/txt/walltime_{platform}_{args.case}_{args.layer:03d}.txt', 'w') as f:
            f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
        return return_values
    return wrapper


def read_path(path):
    path_info = onp.loadtxt(path)
    traveled_time =  path_info[:, 0]
    x_corners = path_info[:, 1]
    y_corners = path_info[:, 2]
    power_control = path_info[:-1, 3]
    ts, xs, ys, ps = [], [], [], []
    for i in range(len(traveled_time) - 1):
        ts_seg = onp.arange(traveled_time[i], traveled_time[i + 1], args.dt)
        xs_seg = onp.linspace(x_corners[i], x_corners[i + 1], len(ts_seg))
        ys_seg = onp.linspace(y_corners[i], y_corners[i + 1], len(ts_seg))
        ps_seg = onp.linspace(power_control[i], power_control[i], len(ts_seg))
        ts.append(ts_seg)
        xs.append(xs_seg)
        ys.append(ys_seg)
        ps.append(ps_seg)

    ts, xs, ys, ps = onp.hstack(ts), onp.hstack(xs), onp.hstack(ys), onp.hstack(ps)  
    print(f"Total number of time steps = {len(ts)}")
    return ts, xs, ys, ps
