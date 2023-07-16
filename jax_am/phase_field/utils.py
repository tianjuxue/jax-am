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
from sklearn.decomposition import PCA

onp.random.seed(1)


class Field:
    """Handles polycrystal mesh, grain orientations, etc.
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

def generate_new_mesh(pf_args, ori2, polycrystal, layer):
    """
    Generate one new layer to model multi-layer polycrystal
    """
    polycrystal_next = Field(pf_args, ori2)
    # 通过改变晶粒的编号信息偏移晶粒的位置，使其位于之前的晶粒上方
    polycrystal_next.mesh.points[:, 2] = polycrystal_next.mesh.points[:, 2] + pf_args['domain_z']*layer
    polycrystal_next.mesh.cells[0].data = polycrystal_next.mesh.cells[0].data + len(polycrystal_next.mesh.points)*layer
    polycrystal_next.centroids[:, 2] = polycrystal_next.centroids[:, 2] + pf_args['domain_z']*layer
    # 合并points
    polycrystal_next.mesh.points = onp.concatenate(
        (polycrystal.mesh.points, polycrystal_next.mesh.points), axis=0)
    # 合并cells
    polycrystal_next.mesh.cells[0].data = onp.concatenate(
        (polycrystal.mesh.cells[0].data, polycrystal_next.mesh.cells[0].data), axis=0)
    # 合并三个编号
    polycrystal_next.mesh.cell_data['gmsh:physical'][0] = onp.concatenate(
        (polycrystal.mesh.cell_data['gmsh:physical'][0], polycrystal_next.mesh.cell_data['gmsh:physical'][0]),
        axis=0)
    polycrystal_next.mesh.cell_data['gmsh:geometrical'][0] = onp.concatenate(
        (polycrystal.mesh.cell_data['gmsh:geometrical'][0], polycrystal_next.mesh.cell_data['gmsh:geometrical'][0]),
        axis=0)
    polycrystal_next.mesh.cell_data['grain_inds'][0] = onp.concatenate(
        (polycrystal.mesh.cell_data['grain_inds'][0], polycrystal_next.mesh.cell_data['grain_inds'][0]), axis=0)
    # 合并晶粒取向
    polycrystal_next.cell_ori_inds = onp.concatenate((polycrystal.cell_ori_inds, polycrystal_next.cell_ori_inds), axis=0)
    # 合并每个单元中心点信息
    polycrystal_next.centroids = onp.concatenate((polycrystal.centroids, polycrystal_next.centroids), axis=0)
    return polycrystal_next

def process_eta(pf_args):
    step = 13
    file_path = os.path.join(pf_args['data_dir'], f"vtk/pf/sols/u{step:03d}.vtu")
    mesh_w_data = meshio.read(file_path)
    cell_ori_inds = mesh_w_data.cell_data['ori_inds'][0] 

    # By default, numpy uses order='C'
    cell_ori_inds_3D = onp.reshape(cell_ori_inds, (pf_args['Nz'], pf_args['Ny'], pf_args['Nx']))

    # This should also work
    # cell_ori_inds_3D = onp.reshape(cell_ori_inds, (pf_args['Nx'], pf_args['Ny'], pf_args['Nz']), order='F')

    print(cell_ori_inds_3D.shape)

    T = mesh_w_data.cell_data['T'][0] 
    nonliquid = T.reshape(-1) < pf_args['T_liquidus']
    edges_in_order = compute_edges_in_order(pf_args)


    points = mesh_w_data.points
    cells = mesh_w_data.cells_dict['hexahedron']
    cell_points = onp.take(points, cells, axis=0)
    centroids = onp.mean(cell_points, axis=1)

    domain_vol = pf_args['domain_x']*pf_args['domain_y']*pf_args['domain_z']
    volumes = domain_vol / len(cells) * onp.ones(len(cells))

    grains_combined = BFS(edges_in_order, nonliquid, cell_ori_inds, pf_args)
    grain_vols, grain_centroids = get_aspect_ratio_inputs(grains_combined, volumes, centroids)
    eta_results = compute_aspect_ratios_and_vols(grain_vols, grain_centroids)

    # print(eta_results)

    return eta_results


def compute_edges_in_order(pf_args):
    Nx, Ny, Nz = pf_args['Nx'], pf_args['Ny'], pf_args['Nz']
    num_total_cells = Nx*Ny*Nz
    cell_inds = onp.arange(num_total_cells).reshape(Nz, Ny, Nx)
    edges_x = onp.stack((cell_inds[:, :, :-1], cell_inds[:, :, 1:]), axis=3).reshape(-1, 2)
    edges_y = onp.stack((cell_inds[:, :-1, :], cell_inds[:, 1:, :]), axis=3).reshape(-1, 2)
    edges_z = onp.stack((cell_inds[:-1, :, :], cell_inds[1:, :, :]), axis=3).reshape(-1, 2)
    edges = onp.vstack((edges_x, edges_y, edges_z))
    print(f"edges.shape = {edges.shape}")
    edges_in_order = [[] for _ in range(num_total_cells)]
    print(f"Re-ordering edges and face_areas...")
    for i, edge in enumerate(edges):
        node1 = edge[0]
        node2 = edge[1]
        edges_in_order[node1].append(node2)
        edges_in_order[node2].append(node1)  
    return edges_in_order


def BFS(edges_in_order, nonliquid, cell_ori_inds, pf_args, combined=True):
    num_graph_nodes = len(nonliquid)
    print(f"BFS...")
    visited = onp.zeros(num_graph_nodes)
    grains = [[] for _ in range(pf_args['num_oris'])]
    for i in range(len(visited)):
        if visited[i] == 0 and nonliquid[i]:
            oris_index = cell_ori_inds[i]
            grains[oris_index].append([])
            queue = [i]
            visited[i] = 1
            while queue:
                s = queue.pop(0) 
                grains[oris_index][-1].append(s)
                connected_nodes = edges_in_order[s]
                for cn in connected_nodes:
                    if visited[cn] == 0 and cell_ori_inds[cn] == oris_index and nonliquid[cn]:
                        queue.append(cn)
                        visited[cn] = 1

    grains_combined = []
    for i in range(len(grains)):
        grains_oris = grains[i] 
        for j in range(len(grains_oris)):
            grains_combined.append(grains_oris[j])

    if combined:
        return grains_combined
    else:
        return grains


def get_aspect_ratio_inputs(grains_combined, volumes, centroids):
    grain_vols = []
    grain_centroids = []
    for i in range(len(grains_combined)):
        grain = grains_combined[i]
        grain_vol = onp.array([volumes[g] for g in grain])
        grain_centroid = onp.take(centroids, grain, axis=0)
        assert grain_centroid.shape == (len(grain_vol), 3)
        grain_vols.append(grain_vol)
        grain_centroids.append(grain_centroid)

    return grain_vols, grain_centroids


def compute_aspect_ratios_and_vols(grain_vols, grain_centroids):
    pca = PCA(n_components=3)
    print(f"Call compute_aspect_ratios_and_vols")
    grain_sum_vols = []
    grain_sum_aspect_ratios = []

    for i in range(len(grain_vols)):
        grain_vol = grain_vols[i]
        sum_vol = onp.sum(grain_vol)
     
        if len(grain_vol) < 5:
            print(f"Grain vol too small, ignore and set aspect_ratio = 1.")
            grain_sum_aspect_ratios.append(1.)
        else:
            directions = grain_centroids[i]
            weighted_directions = directions * grain_vol[:, None]
            # weighted_directions = weighted_directions - onp.mean(weighted_directions, axis=0)[None, :]
            pca.fit(weighted_directions)
            components = pca.components_
            ev = pca.explained_variance_
            lengths = onp.sqrt(ev)
            aspect_ratio = 2*lengths[0]/(lengths[1] + lengths[2])
            grain_sum_aspect_ratios.append(aspect_ratio)

        grain_sum_vols.append(sum_vol)

    print(len(grain_sum_vols))
    print(len(grain_sum_aspect_ratios))
    return [grain_sum_vols, grain_sum_aspect_ratios]

