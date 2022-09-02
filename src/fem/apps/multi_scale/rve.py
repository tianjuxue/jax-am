import numpy as onp
import jax
import jax.numpy as np
from src.fem.generate_mesh import box_mesh
from src.fem.jax_fem import Mesh, LinearElasticity
from src.fem.solver import solver
from src.fem.utils import save_sol



def compute_periodic_inds(location_fns_A, location_fns_B, mappings, vecs, mesh):
    p_node_inds_list_A = []
    p_node_inds_list_B = []
    p_vec_inds_list = []
    for i in range(len(location_fns_A)):
        node_inds_A = np.argwhere(jax.vmap(location_fns_A[i])(mesh.points)).reshape(-1)
        node_inds_B = np.argwhere(jax.vmap(location_fns_B[i])(mesh.points)).reshape(-1)
        points_set_A = mesh.points[node_inds_A]
        points_set_B = mesh.points[node_inds_B]

        EPS = 1e-8
        node_inds_B_ordered = []
        for node_ind in node_inds_A:
            point_A = mesh.points[node_ind]
            dist = np.linalg.norm(mappings[i](point_A)[None, :] - points_set_B, axis=-1)
            node_ind_B_ordered = node_inds_B[np.argwhere(dist < EPS)].reshape(-1)
            node_inds_B_ordered.append(node_ind_B_ordered)

        node_inds_B_ordered = np.array(node_inds_B_ordered).reshape(-1)
        vec_inds = np.ones_like(node_inds_A, dtype=np.int32)*vecs[i]

        p_node_inds_list_A.append(node_inds_A)
        p_node_inds_list_B.append(node_inds_B_ordered)
        p_vec_inds_list.append(vec_inds)

        # TODO: A better way needed
        assert len(node_inds_A) == len(node_inds_B_ordered)

    return p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list


def exp():

    problem_name = "rve"
    L = 1.

    meshio_mesh = box_mesh(10, 10, 10, L, L, L)
    jax_mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], L, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def top(point):
        return np.isclose(point[2], L, atol=1e-5)

    def dirichlet_1(point):
        return 0.

    def dirichlet_2(point):
        return 0.1

    location_fns = [left, left, left, right, right, right]
    value_fns = [dirichlet_1, dirichlet_1, dirichlet_1, dirichlet_2, dirichlet_1, dirichlet_1]
    vecs = [0, 1, 2, 0, 1, 2]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    def mapping(x):
        y = x + np.array([0., 0., L])
        return y

    periodic_bc_info = compute_periodic_inds([bottom], [top], [mapping], [2], jax_mesh)

    problem = LinearElasticity(f"{problem_name}", jax_mesh, dirichlet_bc_info=dirichlet_bc_info, periodic_bc_info=periodic_bc_info)
    sol = solver(problem)

    jax_vtu_path = f"src/fem/apps/multi_scale/data/vtk/{problem_name}/sol.vtu"

    save_sol(problem, sol, jax_vtu_path)

    # p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list = periodic_bc_info
    # a = sol[p_node_inds_list_A[0], p_vec_inds_list[0]]
    # b = sol[p_node_inds_list_B[0], p_vec_inds_list[0]]
    # ap = jax_mesh.points[p_node_inds_list_A[0]]
    # bp = jax_mesh.points[p_node_inds_list_B[0]]
 

if __name__=="__main__":
    exp()

