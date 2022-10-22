import jax.numpy as np
import jax
import numpy as onp
import orix
import meshio
import pickle
import time
import os
import matplotlib.pyplot as plt
from orix import plot
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from scipy.spatial.transform import Rotation as R

onp.random.seed(1)


class Field:
    """Handles polycrystal mesh, grain orientations...
    TODO: Implement post-processing functions
    """
    def __init__(self, pf_args, ori2=None):
        self.pf_args = pf_args
        self.ori2 = ori2
        self.process_neper_mesh()
        self.get_unique_ori_colors()

    def process_neper_mesh(self):
        print(f"Processing neper mesh...")
        neper_folder = os.path.join(self.pf_args['data_dir'], "neper")

        mesh = meshio.read(os.path.join(neper_folder, f"domain.msh"))
        points = mesh.points
        cells = mesh.cells_dict['hexahedron']
        cell_grain_inds = mesh.cell_data['gmsh:physical'][0] - 1
        assert self.pf_args['num_grains'] == onp.max(cell_grain_inds) + 1, \
        f"specified number of grains = {self.pf_args['num_grains']}, actual Neper = {onp.max(cell_grain_inds) + 1}"
        mesh.cell_data['grain_inds'] = [cell_grain_inds]
        grain_oris_inds = onp.random.randint(self.pf_args['num_oris'], size=self.pf_args['num_grains'])
        cell_ori_inds = onp.take(grain_oris_inds, cell_grain_inds, axis=0)

        # TODO: Not robust
        Nx = round(self.pf_args['domain_x'] / points[1, 0])
        Ny = round(self.pf_args['domain_y'] / points[Nx + 1, 1])
        Nz = round(self.pf_args['domain_z'] / points[(Nx + 1)*(Ny + 1), 2])
        assert Nx*Ny*Nz == len(cells)
        self.pf_args['Nx'] = Nx
        self.pf_args['Ny'] = Ny
        self.pf_args['Nz'] = Nz
        print(f"Nx = {Nx}, Ny = {Ny}, Nz = {Nz}")
        print(f"Total num of grains = {self.pf_args['num_grains']}")
        print(f"Total num of orientations = {self.pf_args['num_oris']}")
        print(f"Total num of finite difference cells = {len(cells)}")

        cell_points = onp.take(points, cells, axis=0)
        centroids = onp.mean(cell_points, axis=1)
        mesh_h_xyz = (self.pf_args['domain_x']/self.pf_args['Nx'], 
                      self.pf_args['domain_y']/self.pf_args['Ny'], 
                      self.pf_args['domain_z']/self.pf_args['Nz'])

        self.mesh = mesh
        self.mesh_h_xyz = mesh_h_xyz
        self.centroids = centroids
        self.cell_ori_inds = cell_ori_inds 

        pf_vtk_mesh_folder = os.path.join(self.pf_args['data_dir'], f"vtk/pf/mesh")
        os.makedirs(pf_vtk_mesh_folder, exist_ok=True)
        mesh.write(os.path.join(pf_vtk_mesh_folder, f"fd_mesh.vtu"))

        # Optionally, create a poly mesh: obj to vtu
        file = open(os.path.join(neper_folder, "domain.obj"), 'r')
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
        poly_mesh = meshio.Mesh(points, cells)
        poly_mesh.write(os.path.join(pf_vtk_mesh_folder, f"poly_mesh.vtu"))

    def get_unique_ori_colors(self):
        """Grain orientations and IPF colors
        """
        if self.ori2 is None:
            ori2 = Orientation.random(self.pf_args['num_oris'])
        else:
            ori2 = self.ori2

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

        dx = onp.array([1., 0., 0.])
        dy = onp.array([0., 1., 0.])
        dz = onp.array([0., 0., 1.])
        scipy_quat = onp.concatenate((ori2.data[:, 1:], ori2.data[:, :1]), axis=1)
        r = R.from_quat(scipy_quat)
        grain_directions = onp.stack((r.apply(dx), r.apply(dy), r.apply(dz)))

        # Output orientations to numpy in the form of quaternion
        pf_numpy_folder = os.path.join(self.pf_args['data_dir'], "numpy/pf")
        os.makedirs(pf_numpy_folder, exist_ok=True)
        onp.save(os.path.join(pf_numpy_folder, f"quat.npy"), ori2.data)

        # Plot orientations with IPF figures
        pf_pdf_folder = os.path.join(self.pf_args['data_dir'], "pdf/pf")
        os.makedirs(pf_pdf_folder, exist_ok=True)

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
        plt.savefig(os.path.join(pf_pdf_folder, f'ipf_x.pdf'), bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_y, direction=ipfkey_y.direction)
        plt.savefig(os.path.join(pf_pdf_folder, f'ipf_y.pdf'), bbox_inches='tight')
        ori2.scatter("ipf", c=rgb_z, direction=ipfkey_z.direction)
        plt.savefig(os.path.join(pf_pdf_folder, f'ipf_z.pdf'), bbox_inches='tight')

        # Plot the IPF legend
        new_params = {
            "figure.facecolor": "w",
            "figure.figsize": (6, 3),
            "lines.markersize": 10,
            "font.size": 25,
            "axes.grid": True,
        }
        plt.rcParams.update(new_params)
        plot.IPFColorKeyTSL(symmetry.Oh).plot()
        plt.savefig(os.path.join(pf_pdf_folder, "ipf_legend.pdf"), bbox_inches='tight')

        self.unique_oris_rgb, self.unique_grain_directions = rgb, grain_directions

    def convert_to_3D_images(self):
        step = 0
        file_path = os.path.join(self.pf_args['data_dir'], f"vtk/pf/sols/u{step:03d}.vtu")
        mesh_w_data = meshio.read(file_path)
        cell_ori_inds = mesh_w_data.cell_data['ori_inds'][0] 

        # By default, numpy uses order='C'
        cell_ori_inds_3D = np.reshape(cell_ori_inds, (self.pf_args['Nz'], self.pf_args['Ny'], self.pf_args['Nx']))

        # This should also work
        # cell_ori_inds_3D = np.reshape(cell_ori_inds, (self.pf_args['Nx'], self.pf_args['Ny'], self.pf_args['Nz']), order='F')

        print(cell_ori_inds_3D.shape)
        return cell_ori_inds_3D


def walltime(data_dir=None):
    def decorate(func):
        def wrapper(*list_args, **keyword_args):
            start_time = time.time()
            return_values = func(*list_args, **keyword_args)
            end_time = time.time()
            time_elapsed = end_time - start_time
            platform = jax.lib.xla_bridge.get_backend().platform
            print(f"Time elapsed {time_elapsed} of function {func.__name__} on platform {platform}")
            if data_dir is not None:
                txt_dir = os.path.join(data_dir, f'txt')
                os.makedirs(txt_dir, exist_ok=True)
                with open(os.path.join(txt_dir, f"walltime_{platform}.txt"), 'w') as f:
                    f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
            return return_values
        return wrapper
    return decorate


def read_path(pf_args):
    """TODO: should be used by CFD
    """
    traveled_time = pf_args['laser_path']['time']
    x_corners = pf_args['laser_path']['x_pos']
    y_corners = pf_args['laser_path']['y_pos']
    power_control = pf_args['laser_path']['switch'][:-1]

    ts, xs, ys, ps = [], [], [], []
    for i in range(len(traveled_time) - 1):
        ts_seg = onp.arange(traveled_time[i], traveled_time[i + 1], pf_args['dt'])
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


def make_video(pf_args):
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite

    # TODO
    os.system(f'ffmpeg -y -framerate 10 -i {pf_args["data_dir"]}/png/tmp/u.%04d.png -pix_fmt yuv420p -vf \
               "crop=trunc(iw/2)*2:trunc(ih/2)*2" {pf_args["data_dir"]}/mp4/test.mp4')


if __name__=="__main__":
    make_video()
