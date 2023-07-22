import numpy as onp
import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO

# DEBUGGING ONLY:
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=3'
import scipy
from functools import partial, wraps
import sys
import time
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Union

from jax_am.common import timeit
from jax_am.fem.generate_mesh import Mesh
from jax_am.fem.basis import get_face_shape_vals_and_grads, get_shape_vals_and_grads

from jax.config import config

import logging
logger = logging.getLogger(__name__)

config.update("jax_enable_x64", True)

onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=5)


@dataclass
class FEM:
    """
    Solving second-order elliptic PDE problems whose FEM weak form is
    (f(u_grad), v_grad) * dx - (traction, v) * ds - (body_force, v) * dx = 0,
    where u and v are trial and test functions, respectively, and f is a general function.
    This covers
        - Poisson's problem
        - Heat equation
        - Linear elasticity
        - Hyper-elasticity
        - Plasticity
        ...

    Attributes
    ----------
    mesh : Mesh object
        The mesh object stores points (coordinates) and cells (connectivity).
    vec : int
        The number of vector variable components of the solution.
        E.g., a 3D displacement field has u_x, u_y and u_z components, so vec=3
    dim : int
        The dimension of the problem.
    ele_type : str
        Element type
    dirichlet_bc_info : [location_fns, vecs, value_fns]
        location_fns : List[Callable]
            Callable : a function that inputs a point and returns if the point satisfies the location condition
        vecs: List[int]
            integer value must be in the range of 0 to vec - 1,
            specifying which component of the (vector) variable to apply Dirichlet condition to
        value_fns : List[Callable]
            Callable : a function that inputs a point and returns the Dirichlet value
    periodic_bc_info : [location_fns_A, location_fns_B, mappings, vecs]
        location_fns_A : List[Callable]
            Callable : location function for boundary A
        location_fns_B : List[Callable]
            Callable : location function for boundary B
        mappings : List[Callable]
            Callable: function mapping a point from boundary A to boundary B
        vecs: List[int]
            which component of the (vector) variable to apply periodic condition to
    neumann_bc_info : [location_fns, value_fns]
        location_fns : List[Callable]
            Callable : location function for Neumann boundary
        value_fns : List[Callable]
            Callable : a function that inputs a point and returns the Neumann value
    cauchy_bc_info : [location_fns, value_fns]
        location_fns : List[Callable]
            Callable : location function for Cauchy boundary
        value_fns : List[Callable]
            Callable : a function that inputs the solution and returns the Cauchy boundary value
    source_info: Callable
        A function that inputs a point and returns the body force at this point
    additional_info : Any
        Other information that the FEM solver should know
    """
    mesh: Mesh
    vec: int
    dim: int
    ele_type: str = 'HEX8'
    dirichlet_bc_info: Optional[List[Union[List[Callable], List[int],
                                           List[Callable]]]] = None
    periodic_bc_info: Optional[List[Union[List[Callable], List[Callable],
                                          List[Callable], List[int]]]] = None
    neumann_bc_info: Optional[List[Union[List[Callable],
                                         List[Callable]]]] = None
    cauchy_bc_info: Optional[List[Union[List[Callable],
                                        List[Callable]]]] = None
    source_info: Callable = None
    additional_info: Any = ()

    def __post_init__(self):
        self.points = self.mesh.points
        self.cells = self.mesh.cells
        self.num_cells = len(self.cells)
        self.num_total_nodes = len(self.points)
        self.num_total_dofs = self.num_total_nodes * self.vec

        start = time.time()
        logger.info(f"Computing shape function values, gradients, etc.")

        self.shape_vals, self.shape_grads_ref, self.quad_weights = get_shape_vals_and_grads(
            self.ele_type)
        self.face_shape_vals, self.face_shape_grads_ref, self.face_quad_weights, self.face_normals, self.face_inds \
        = get_face_shape_vals_and_grads(self.ele_type)
        self.num_quads = self.shape_vals.shape[0]
        self.num_nodes = self.shape_vals.shape[1]
        self.num_faces = self.face_shape_vals.shape[0]
        self.shape_grads, self.JxW = self.get_shape_grads()

        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(
            self.dirichlet_bc_info)
        self.p_node_inds_list_A, self.p_node_inds_list_B, self.p_vec_inds_list = self.periodic_boundary_conditions(
        )

        # (num_cells, num_quads, num_nodes, 1, dim)
        self.v_grads_JxW = self.shape_grads[:, :, :,
                                            None, :] * self.JxW[:, :, None,
                                                                None, None]

        end = time.time()
        compute_time = end - start

        self.internal_vars = {}
        self.compute_Neumann_boundary_inds()

        logger.info(f"Done pre-computations, took {compute_time} [s]")
        logger.info(
            f"Solving a problem with {len(self.cells)} cells, {self.num_total_nodes}x{self.vec} = {self.num_total_dofs} dofs."
        )

        self.custom_init(*self.additional_info)

    def custom_init(self):
        """Child class should override if more things need to be done in initialization
        """
        pass

    def get_shape_grads(self):
        """Compute shape function gradient value
        The gradient is w.r.t physical coordinates.
        See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
        Page 147, Eq. (3.9.3)

        Returns
        -------
        shape_grads_physical : onp.ndarray
            (num_cells, num_quads, num_nodes, dim)
        JxW : onp.ndarray
            (num_cells, num_quads)
        """
        assert self.shape_grads_ref.shape == (self.num_quads, self.num_nodes,
                                              self.dim)
        physical_coos = onp.take(self.points, self.cells,
                                 axis=0)  # (num_cells, num_nodes, dim)
        # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
        jacobian_dx_deta = onp.sum(physical_coos[:, None, :, :, None] *
                                   self.shape_grads_ref[None, :, :, None, :],
                                   axis=2,
                                   keepdims=True)
        jacobian_det = onp.linalg.det(
            jacobian_dx_deta)[:, :, 0]  # (num_cells, num_quads)
        jacobian_deta_dx = onp.linalg.inv(jacobian_dx_deta)
        # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim)
        # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
        shape_grads_physical = (self.shape_grads_ref[None, :, :, None, :]
                                @ jacobian_deta_dx)[:, :, :, 0, :]
        JxW = jacobian_det * self.quad_weights[None, :]
        return shape_grads_physical, JxW

    def get_face_shape_grads(self, boundary_inds):
        """Face shape function gradients and JxW (for surface integral)
        Nanson's formula is used to map physical surface ingetral to reference domain
        Reference: https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            (num_selected_faces, 2)

        Returns
        -------
        face_shape_grads_physical : onp.ndarray
            (num_selected_faces, num_face_quads, num_nodes, dim)
        nanson_scale : onp.ndarray
            (num_selected_faces, num_face_quads)
        """
        physical_coos = onp.take(self.points, self.cells,
                                 axis=0)  # (num_cells, num_nodes, dim)
        selected_coos = physical_coos[
            boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_f_shape_grads_ref = self.face_shape_grads_ref[
            boundary_inds[:,
                          1]]  # (num_selected_faces, num_face_quads, num_nodes, dim)
        selected_f_normals = self.face_normals[
            boundary_inds[:, 1]]  # (num_selected_faces, dim)

        # (num_selected_faces, 1, num_nodes, dim, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        # (num_selected_faces, num_face_quads, num_nodes, dim, dim) -> (num_selected_faces, num_face_quads, dim, dim)
        jacobian_dx_deta = onp.sum(
            selected_coos[:, None, :, :, None] *
            selected_f_shape_grads_ref[:, :, :, None, :],
            axis=2)
        jacobian_det = onp.linalg.det(
            jacobian_dx_deta)  # (num_selected_faces, num_face_quads)
        jacobian_deta_dx = onp.linalg.inv(
            jacobian_dx_deta)  # (num_selected_faces, num_face_quads, dim, dim)

        # (1, num_face_quads, num_nodes, 1, dim) @ (num_selected_faces, num_face_quads, 1, dim, dim)
        # (num_selected_faces, num_face_quads, num_nodes, 1, dim) -> (num_selected_faces, num_face_quads, num_nodes, dim)
        face_shape_grads_physical = (
            selected_f_shape_grads_ref[:, :, :, None, :]
            @ jacobian_deta_dx[:, :, None, :, :])[:, :, :, 0, :]

        # (num_selected_faces, 1, 1, dim) @ (num_selected_faces, num_face_quads, dim, dim)
        # (num_selected_faces, num_face_quads, 1, dim) -> (num_selected_faces, num_face_quads)
        nanson_scale = onp.linalg.norm(
            (selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :,
                                                                      0, :],
            axis=-1)
        selected_weights = self.face_quad_weights[
            boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads)
        nanson_scale = nanson_scale * jacobian_det * selected_weights
        return face_shape_grads_physical, nanson_scale

    def get_physical_quad_points(self):
        """Compute physical quadrature points

        Returns
        -------
        physical_quad_points : onp.ndarray
            (num_cells, num_quads, dim)
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)
        # (1, num_quads, num_nodes, 1) * (num_cells, 1, num_nodes, dim) -> (num_cells, num_quads, dim)
        physical_quad_points = onp.sum(self.shape_vals[None, :, :, None] *
                                       physical_coos[:, None, :, :],
                                       axis=2)
        return physical_quad_points

    def get_physical_surface_quad_points(self, boundary_inds):
        """Compute physical quadrature points on the surface

        Parameters
        ----------
        boundary_inds : List[onp.ndarray]
            ndarray shape: (num_selected_faces, 2)

        Returns
        -------
        physical_surface_quad_points : ndarray
            (num_selected_faces, num_face_quads, dim)
        """
        physical_coos = onp.take(self.points, self.cells, axis=0)
        selected_coos = physical_coos[
            boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, dim)
        selected_face_shape_vals = self.face_shape_vals[
            boundary_inds[:,
                          1]]  # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, num_face_quads, num_nodes, 1) * (num_selected_faces, 1, num_nodes, dim) -> (num_selected_faces, num_face_quads, dim)
        physical_surface_quad_points = onp.sum(
            selected_face_shape_vals[:, :, :, None] *
            selected_coos[:, None, :, :],
            axis=2)
        return physical_surface_quad_points

    def Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Indices and values for Dirichlet B.C.

        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]

        Returns
        -------
        node_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to num_total_nodes - 1
        vec_inds_List : List[onp.ndarray]
            The ndarray ranges from 0 to to vec - 1
        vals_List : List[ndarray]
            Dirichlet values to be assigned
        """
        node_inds_list = []
        vec_inds_list = []
        vals_list = []

        if dirichlet_bc_info is not None:
            # Unpack dirichlet_bc_info
            location_fns, vecs, value_fns = dirichlet_bc_info
            if len(location_fns) != len(vecs) != len(value_fns):
                raise ValueError("All lists in dirichlet_bc_info must be the"
                                 " same length.")

            # For each location function, find the indices of the nodes that
            # satisfy the location function, vector indices, and the
            # corresponding Dirichlet values

            for i, (location_fn, vec, value_fn) in enumerate(zip(location_fns, vecs, value_fns)):
                node_inds = onp.argwhere(jax.vmap(location_fn)(self.points)).reshape(-1)
                vec_inds = onp.ones_like(node_inds, dtype=onp.int32) * vec
                values = jax.vmap(value_fn)(self.points[node_inds].reshape(-1, self.dim)).reshape(-1)

                node_inds_list.append(node_inds)
                vec_inds_list.append(vec_inds)
                vals_list.append(values)

        return node_inds_list, vec_inds_list, vals_list

    def update_Dirichlet_boundary_conditions(self, dirichlet_bc_info):
        """Reset Dirichlet boundary conditions.
        Useful when a time-dependent problem is solved, and at each iteration the boundary condition needs to be updated.

        Parameters
        ----------
        dirichlet_bc_info : [location_fns, vecs, value_fns]
        """
        self.node_inds_list, self.vec_inds_list, self.vals_list = self.Dirichlet_boundary_conditions(
            dirichlet_bc_info)

    def periodic_boundary_conditions(self):
        p_node_inds_list_A = []
        p_node_inds_list_B = []
        p_vec_inds_list = []
        if self.periodic_bc_info is not None:
            location_fns_A, location_fns_B, mappings, vecs = self.periodic_bc_info
            for i in range(len(location_fns_A)):
                node_inds_A = onp.argwhere(
                    jax.vmap(location_fns_A[i])(self.mesh.points)).reshape(-1)
                node_inds_B = onp.argwhere(
                    jax.vmap(location_fns_B[i])(self.mesh.points)).reshape(-1)
                points_set_A = self.mesh.points[node_inds_A]
                points_set_B = self.mesh.points[node_inds_B]

                EPS = 1e-5
                node_inds_B_ordered = []
                for node_ind in node_inds_A:
                    point_A = self.mesh.points[node_ind]
                    dist = onp.linalg.norm(mappings[i](point_A)[None, :] -
                                           points_set_B,
                                           axis=-1)
                    node_ind_B_ordered = node_inds_B[onp.argwhere(
                        dist < EPS)].reshape(-1)
                    node_inds_B_ordered.append(node_ind_B_ordered)

                node_inds_B_ordered = onp.array(node_inds_B_ordered).reshape(
                    -1)
                vec_inds = onp.ones_like(node_inds_A,
                                         dtype=onp.int32) * vecs[i]

                p_node_inds_list_A.append(node_inds_A)
                p_node_inds_list_B.append(node_inds_B_ordered)
                p_vec_inds_list.append(vec_inds)
                assert len(node_inds_A) == len(node_inds_B_ordered)

        return p_node_inds_list_A, p_node_inds_list_B, p_vec_inds_list

    def get_boundary_conditions_inds(self, location_fns):
        """Given location functions, compute which faces satisfy the condition.

        Parameters
        ----------
        location_fns : List[Callable]
            Callable: a function that inputs a point (ndarray) and returns if the point satisfies the location condition
                      e.g., lambda x: np.isclose(x[0], 0.)

        Returns
        -------
        boundary_inds_list : List[onp.ndarray]
            (num_selected_faces, 2)
            boundary_inds_list[k][i, 0] returns the global cell index of the ith selected face of boundary subset k
            boundary_inds_list[k][i, 1] returns the local face index of the ith selected face of boundary subset k
        """
        # (num_cells, num_nodes, dim)
        cell_points = onp.take(self.points, self.cells, axis=0)
        # (num_cells, num_faces, num_face_nodes, dim)
        cell_face_points = onp.take(cell_points, self.face_inds, axis=1)
        boundary_inds_list = []

        for i in range(len(location_fns)):
            vmap_location_fn = jax.vmap(location_fns[i])

            def on_boundary(cell_points):
                boundary_flag = vmap_location_fn(cell_points)
                return onp.all(boundary_flag)

            vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
            boundary_flags = vvmap_on_boundary(cell_face_points)
            # (num_selected_faces, 2)
            boundary_inds = onp.argwhere(boundary_flags)
            boundary_inds_list.append(boundary_inds)
        return boundary_inds_list

    def compute_Neumann_integral_vars(self, **internal_vars):
        """In the weak form, we have the Neumann integral: (traction, v) * ds, and this function computes this.

        Returns
        -------
        integral: np.DeviceArray
            (num_total_nodes, vec)
        """
        integral = np.zeros((self.num_total_nodes, self.vec))
        if self.neumann_bc_info is not None:
            for i, boundary_inds in enumerate(self.neumann_boundary_inds_list):
                if 'neumann' in internal_vars.keys():
                    int_vars = internal_vars['neumann'][i]
                else:
                    int_vars = ()
                # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
                subset_quad_points = self.get_physical_surface_quad_points(
                    boundary_inds)
                # int_vars = [x[i] for x in internal_vars]
                traction = jax.vmap(jax.vmap(self.neumann_value_fns[i]))(
                    subset_quad_points,
                    *int_vars)  # (num_selected_faces, num_face_quads, vec)
                assert len(traction.shape) == 3
                _, nanson_scale = self.get_face_shape_grads(
                    boundary_inds)  # (num_selected_faces, num_face_quads)
                # (num_faces, num_face_quads, num_nodes) ->  (num_selected_faces, num_face_quads, num_nodes)
                v_vals = np.take(self.face_shape_vals,
                                 boundary_inds[:, 1],
                                 axis=0)
                v_vals = np.repeat(
                    v_vals[:, :, :, None], self.vec, axis=-1
                )  # (num_selected_faces, num_face_quads, num_nodes, vec)
                subset_cells = np.take(
                    self.cells, boundary_inds[:, 0],
                    axis=0)  # (num_selected_faces, num_nodes)
                # (num_selected_faces, num_nodes, vec) -> (num_selected_faces*num_nodes, vec)
                int_vals = np.sum(v_vals * traction[:, :, None, :] *
                                  nanson_scale[:, :, None, None],
                                  axis=1).reshape(-1, self.vec)
                integral = integral.at[subset_cells.reshape(-1)].add(int_vals)
        return integral

    def compute_Neumann_boundary_inds(self):
        """Child class should override if internal variables exist
        """
        if self.neumann_bc_info is not None:
            self.neumann_location_fns, self.neumann_value_fns = self.neumann_bc_info
            if self.neumann_location_fns is not None:
                self.neumann_boundary_inds_list = self.get_boundary_conditions_inds(
                    self.neumann_location_fns)

    def compute_body_force_by_fn(self):
        """In the weak form, we have (body_force, v) * dx, and this function computes this

        Returns
        -------
        body_force: np.DeviceArray
            (num_total_nodes, vec)
        """
        rhs = np.zeros((self.num_total_nodes, self.vec))
        if self.source_info is not None:
            body_force_fn = self.source_info
            physical_quad_points = self.get_physical_quad_points(
            )  # (num_cells, num_quads, dim)
            body_force = jax.vmap(jax.vmap(body_force_fn))(
                physical_quad_points)  # (num_cells, num_quads, vec)
            assert len(body_force.shape) == 3
            v_vals = np.repeat(self.shape_vals[None, :, :, None],
                               self.num_cells,
                               axis=0)  # (num_cells, num_quads, num_nodes, 1)
            v_vals = np.repeat(
                v_vals, self.vec,
                axis=-1)  # (num_cells, num_quads, num_nodes, vec)
            # (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
            rhs_vals = np.sum(v_vals * body_force[:, :, None, :] *
                              self.JxW[:, :, None, None],
                              axis=1).reshape(-1, self.vec)
            rhs = rhs.at[self.cells.reshape(-1)].add(rhs_vals)
        return rhs

    def compute_body_force_by_sol(self, sol, mass_map):
        """In the weak form, we have (old_solution, v) * dx, and this function computes this

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)
        mass_map : Callable
            Transformation on sol

        Returns
        -------
        body_force : np.DeviceArray
            (num_total_nodes, vec)
        """
        mass_kernel = self.get_mass_kernel(mass_map)
        cells_sol = sol[self.cells]  # (num_cells, num_nodes, vec)
        val = jax.vmap(mass_kernel)(cells_sol,
                                    self.JxW)  # (num_cells, num_nodes, vec)
        val = val.reshape(-1, self.vec)  # (num_cells*num_nodes, vec)
        body_force = np.zeros_like(sol)
        body_force = body_force.at[self.cells.reshape(-1)].add(val)
        return body_force

    def get_laplace_kernel(self, tensor_map):

        def laplace_kernel(cell_sol, cell_shape_grads, cell_v_grads_JxW,
                           *cell_internal_vars):
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :,
                                                                    None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)

            # Here, we reshape the u_grads to be (num_quads, vec, dim) - to enable vmapping
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)  # (num_quads, vec, dim)

            # (num_quads, vec, dim)
            u_physics = jax.vmap(tensor_map, )(u_grads_reshape, *cell_internal_vars).reshape(u_grads.shape)

            # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW,
                         axis=(0, -1))
            return val

        return laplace_kernel

    def get_mass_kernel(self, mass_map):

        def mass_kernel(cell_sol, cell_JxW, *cell_internal_vars):
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
            u = np.sum(cell_sol[None, :, :] * self.shape_vals[:, :, None],
                       axis=1)
            u_physics = jax.vmap(mass_map)(
                u, *cell_internal_vars)  # (num_quads, vec)
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * self.shape_vals[:, :, None] *
                         cell_JxW[:, None, None],
                         axis=0)
            return val

        return mass_kernel

    def get_cauchy_kernel(self, cauchy_map):

        def cauchy_kernel(cell_sol, face_shape_vals, face_nanson_scale):
            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None],
                       axis=1)
            u_physics = jax.vmap(cauchy_map)(u)  # (num_face_quads, vec)
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :] * face_shape_vals[:, :, None] *
                         face_nanson_scale[:, None, None],
                         axis=0)
            return val

        return cauchy_kernel

    def unpack_kernels_vars(self, **internal_vars):
        if 'mass' in internal_vars.keys():
            mass_internal_vars = internal_vars['mass']
        else:
            mass_internal_vars = ()

        if 'laplace' in internal_vars.keys():
            laplace_internal_vars = internal_vars['laplace']
        else:
            laplace_internal_vars = ()

        return [mass_internal_vars, laplace_internal_vars]

    @timeit
    def split_and_compute_cell(self,
                               cells_sol,
                               np_version,
                               jac_flag,
                               **internal_vars):

        """
        This function splits the computation for each cell and performs these
        computations in batches to optimize memory usage. The computations it
        performs involve applying a kernel function to a set of inputs,
        potentially deriving the Jacobian of the function as well. The kernel
        function to use is determined based on the internal variables provided.

        Parameters: cells_sol : ndarray
            A solution array with data related to cells.

        np_version : module
            The numpy version to use for computation, which can be either
            jax.numpy or numpy. jax.numpy allows for automatic differentiation
            but uses GPU memory, while numpy saves GPU memory but doesn't
            support automatic differentiation.

        jac_flag : bool
            A flag indicating whether to compute the Jacobian. If set to True,
            the function computes the Jacobian and returns it along with the
            computed values.

        internal_vars : dict
            A dictionary of internal variables. These are unpacked and used in
            the computations.

        Returns: values : ndarray
            The results of the computation for each cell.

        jacs : ndarray (only if jac_flag is True)
            The Jacobians computed for each cell.

        The function handles the computation in batches to avoid out-of-memory
        errors. The number of cells can be large, and computing them all at
        once can exceed available memory. Therefore, the function splits the
        computation into smaller batches and performs each computation
        separately.
        """

        # Custom functions for calculating jacobian and value in one pass.
        # JAX does not support this operation internally - authors chose not to
        # implement this themselves because in most cases, the cost of
        # calculating the Jacobian is going to dominate the cost of evaluating
        # the function, so it can be feasible to evaluate the function twice.
        # For discussion see: https://github.com/google/jax/pull/762

        def duplicate_output(has_aux):
            """
            Decorator for duplicating the output of a pipeline function. This operation
            enables the calculation of value and jacobians in a single function call.
            The approach is based on the following discussion:

            Parameters
            ----------
            has_aux : bool
                Whether the function has auxiliary outputs.

            Returns
            -------
            Callable[..., Dict[str, Any]]
                The decorated pipeline function.
            """

            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    if has_aux:
                        out, aux = func(*args, **kwargs)
                        return out, (out, aux)
                    else:
                        out = func(*args, **kwargs)
                        return (out, out)

                return wrapper
            return decorator

        # Note: this approach is slightly neater, because it allows handling
        # of arbitrary input PyTrees as opposed to the previous approach
        def value_and_jacrev(f, x):
            d = duplicate_output(has_aux=False)(f)
            jac, val = jax.jacrev(d, has_aux=True)(x)

            return val, jac.reshape(self.num_nodes,
                                    self.vec,
                                    self.num_nodes,
                                    self.vec)

        def value_and_jacfwd(f, x):
            d = duplicate_output(has_aux=False)(f)
            jac, val = jax.jacfwd(d, has_aux=True)(x)
            return val, jac.reshape(self.num_nodes,
                                    self.vec,
                                    self.num_nodes,
                                    self.vec)

        def get_kernel_fn_cell():
            """
            Selects the 'kernel' function based on the attributes of the class.
            The kernel function is the function that is applied to the cell
            to compute the values and Jacobians.
            """
            def kernel(cell_sol,
                       cell_shape_grads,
                       cell_JxW,
                       cell_v_grads_JxW,
                       cell_mass_internal_vars,
                       cell_laplace_internal_vars):

                if hasattr(self, 'get_mass_map'):
                    mass_kernel = self.get_mass_kernel(self.get_mass_map())
                    mass_val = mass_kernel(cell_sol, cell_JxW,
                                           *cell_mass_internal_vars)
                else:
                    mass_val = 0.

                if hasattr(self, 'get_tensor_map'):
                    laplace_kernel = self.get_laplace_kernel(
                        self.get_tensor_map())
                    laplace_val = laplace_kernel(cell_sol, cell_shape_grads,
                                                 cell_v_grads_JxW,
                                                 *cell_laplace_internal_vars)
                else:
                    laplace_val = 0.

                return laplace_val + mass_val

            def kernel_jac(cell_sol, *args):
                def kernel_partial(cell_sol):
                    return kernel(cell_sol, *args)
                return value_and_jacfwd(kernel_partial, cell_sol)

            return kernel, kernel_jac

        # Select a kernel function based on the internal variables
        kernel, kernel_jac = get_kernel_fn_cell()
        fn = kernel_jac if jac_flag else kernel

        # Unpack the internal variables and pass them to the kernel function
        kernel_vars = self.unpack_kernels_vars(**internal_vars)

        #=======================================================================
        # Experimental xmap implementation
        # In the future, we may want to use xmap instead of vmap to perform
        # the 'chunked' vmap. Since xmap is experimental, the code below
        # is not fully operational (bug related to scan), and therefore it is
        # commented out.

        # num_chunks = 20
        # chunked_fn = xmap(fn,
        #                   in_axes=['batch', ...],
        #                   out_axes=['batch', ...],
        #                   axis_resources={'batch': SerialLoop(num_chunks), })

        #======================================================================

        # TODO: Does not work when num_cells is not divisible by n_devices
        n_devices = jax.device_count()
        logger.debug(f"Using {n_devices} devices in split_and_compute_cell")
        # Total number of chunks - this is dictated by the memory
        n_chunks_total = 20

        # Having a number of chunks divisible by the number of devices is
        # good for load balancing.

        # Ensure n_chunks_total is at least n_devices and a multiple of n_devices
        n_chunks_total = max(n_devices, (n_chunks_total // n_devices) * n_devices)

        logger.debug(f"Total num of chunks is {n_chunks_total}")

        input_data = [cells_sol,
                      self.shape_grads,
                      self.JxW,
                      self.v_grads_JxW,
                      *kernel_vars]

        n_cells = len(cells_sol)


        # Pad the data to be divisible by the number of devices
        padding_size = (-n_cells % n_devices) % n_devices

        logger.debug(f"Padding size is {padding_size}")

        target_size = n_cells + padding_size

        logger.debug(f"Size after padding is {target_size}")

        n_cells_per_device = target_size // n_devices

        logger.debug(f"Num cells per device (after padding) is {n_cells_per_device}")

        if n_devices > 1:
            # Pad the data to be divisible by the number of devices
            padding_size = (-n_cells % n_devices) % n_devices
            target_size = n_cells + padding_size
            n_cells_per_device = target_size // n_devices

            def _pad_and_reshape(x):
                # Pad the arrays with zeros
                pad_width = [(0, 0)] * np.ndim(x)
                pad_width[0] = (0, padding_size)
                x_padded = np.pad(x, pad_width)
                device_shape = (n_devices, n_cells_per_device)
                logger.debug(f"Device shape is {device_shape}")
                x_reshaped = x_padded.reshape(device_shape + x_padded.shape[1:])
                return x_reshaped

            def _remove_padding(x):
                # Reshape to the original shape
                x = x.reshape(-1, *x.shape[2:])
                # Compute how much to slice off
                slice_end = -padding_size if padding_size else None
                logger.debug(f"Before unpadding, shape is {x.shape}")
                x_unpadded = x[:slice_end]
                logger.debug(f"After unpadding, shape is {x_unpadded.shape}")

                return x_unpadded

            # Pad and reshape to distribute across devices
            input_data = jax.tree_map(_pad_and_reshape, input_data)

        def _extract_data_chunk(input_data, chunk_id, chunk_size, num_chunks):
            start = chunk_id * chunk_size
            if chunk_id < num_chunks - 1:
                end = (chunk_id + 1) * chunk_size
            else:  # For the last chunk, take all remaining elements
                end = None
            data_chunk = jax.tree_map(lambda x: x[start:end], input_data)
            return data_chunk

        def chunked_vmap(f, num_chunks):
            def chunked_fn(input_data):
                # Check the size of the first argument
                n_elements = input_data[0].shape[0]
                chunk_size = n_elements // num_chunks

                values = []
                jacs = []
                for chunk_id in range(num_chunks):
                    # Extract chunk
                    data_chunk = _extract_data_chunk(input_data, chunk_id, chunk_size, num_chunks)

                    # Apply original function to the chunk
                    if jac_flag:
                        value, jac = jax.vmap(f)(*data_chunk)
                        logger.debug(f"values shape is {value.shape}")
                        logger.debug(f"jacs shape is {jac.shape}")
                        values.append(value)
                        jacs.append(jac)
                    else:
                        value = jax.vmap(f)(*data_chunk)
                        values.append(value)
                if jac_flag:
                    vals = jax.lax.concatenate(values, 0)
                    jacs = jax.lax.concatenate(jacs, 0)
                    return vals, jacs
                else:
                    return jax.lax.concatenate(value, 0)
            return chunked_fn

        n_chunks_per_device = n_chunks_total // n_devices
        chunked_vmap_fn = chunked_vmap(fn, n_chunks_per_device)
        apply_fn = jax.pmap(chunked_vmap_fn) if n_devices > 1 else chunked_vmap_fn
        # PyTree of inputs to the kernel function

        if jac_flag:
            values, jacs = apply_fn(input_data)
            logger.debug(f"Shape coming out from apply_fn is {values.shape}")
            if n_devices > 1:
                values = _remove_padding(values)
                jacs = _remove_padding(jacs)

            logger.debug(f"Reference values shape is {values.shape}")
            return values, jacs
        else:
            values = apply_fn(input_data)
            if n_devices > 1:
                values = _remove_padding(values)

            logger.debug(f"Reference values shape is {values.shape}")
            return values

#===============================================================================
    def compute_face(self, cells_sol, np_version, jac_flag):

        def get_kernel_fn_face(cauchy_map):

            def kernel(cell_sol, face_shape_vals, face_nanson_scale):
                cauchy_kernel = self.get_cauchy_kernel(cauchy_map)
                val = cauchy_kernel(cell_sol, face_shape_vals,
                                    face_nanson_scale)
                return val

            def kernel_jac(cell_sol, *args):
                return jax.jacfwd(kernel)(cell_sol, *args)

            return kernel, kernel_jac

        # TODO: Better to move the following to __init__ function?
        location_fns, value_fns = self.cauchy_bc_info
        boundary_inds_list = self.get_boundary_conditions_inds(location_fns)
        values = []
        selected_cells = []
        for i, boundary_inds in enumerate(boundary_inds_list):
            selected_cell_sols = cells_sol[
                boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, vec))
            selected_face_shape_vals = self.face_shape_vals[
                boundary_inds[:, 1]]  # (num_selected_faces, num_face_quads, num_nodes)
            _, nanson_scale = self.get_face_shape_grads(
                boundary_inds)  # (num_selected_faces, num_face_quads)
            kernel, kernel_jac = get_kernel_fn_face(value_fns[i])
            fn = kernel_jac if jac_flag else kernel

            # TODO: Nesting vmap and pmap
            vmap_fn = jax.jit(jax.vmap(fn))
            # vmap_fn = jax.pmap(jax.vmap(fn))
            val = vmap_fn(selected_cell_sols,
                          selected_face_shape_vals,
                          nanson_scale)
            values.append(val)
            selected_cells.append(self.cells[boundary_inds[:, 0]])

        values = np_version.vstack(values)
        selected_cells = onp.vstack(selected_cells)

        assert len(values) == len(selected_cells)

        return values, selected_cells

    def convert_from_dof_to_quad(self, sol):
        """Obtain quad values from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)

        Returns
        -------
        u : np.DeviceArray
            (num_cells, num_quads, vec)
        """
        # (num_total_nodes, vec) -> (num_cells, num_nodes, vec)
        cells_sol = sol[self.cells]
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, num_nodes, vec) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] *
                   self.shape_vals[None, :, :, None],
                   axis=2)
        return u

    def convert_neumann_from_dof(self, sol, index):
        """Obtain surface solution from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)
        index : int

        Returns
        -------
        u : np.DeviceArray
            (num_selected_faces, num_face_quads, vec)
        """
        cells_old_sol = sol[self.cells]  # (num_cells, num_nodes, vec)
        boundary_inds = self.neumann_boundary_inds_list[index]
        selected_cell_sols = cells_old_sol[
            boundary_inds[:, 0]]  # (num_selected_faces, num_nodes, vec))
        selected_face_shape_vals = self.face_shape_vals[
            boundary_inds[:,
                          1]]  # (num_selected_faces, num_face_quads, num_nodes)
        # (num_selected_faces, 1, num_nodes, vec) * (num_selected_faces, num_face_quads, num_nodes, 1) -> (num_selected_faces, num_face_quads, vec)
        u = np.sum(selected_cell_sols[:, None, :, :] *
                   selected_face_shape_vals[:, :, :, None],
                   axis=2)
        return u

    def sol_to_grad(self, sol):
        """Obtain solution gradient from nodal solution

        Parameters
        ----------
        sol : np.DeviceArray
            (num_total_nodes, vec)
        index : int

        Returns
        -------
        u_grads : np.DeviceArray
            (num_cells, num_quads, vec, dim)
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = np.take(sol, self.cells,
                          axis=0)[:, None, :, :,
                                  None] * self.shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2)  # (num_cells, num_quads, vec, dim)
        return u_grads

    def compute_residual_vars_helper(self, sol, weak_form, **internal_vars):
        res = np.zeros((self.num_total_nodes, self.vec))
        weak_form = weak_form.reshape(-1,
                                      self.vec)  # (num_cells*num_nodes, vec)
        res = res.at[self.cells.reshape(-1)].add(weak_form)

        if self.cauchy_bc_info is not None:
            cells_sol = sol[self.cells]
            values, selected_cells = self.compute_face(cells_sol, np, False)
            values = values.reshape(-1, self.vec)
            res = res.at[selected_cells.reshape(-1)].add(values)

        self.body_force = self.compute_body_force_by_fn()
        if 'body' in internal_vars.keys():
            self.body_force = self.compute_body_force_by_sol(
                internal_vars['body'], self.get_body_map())

        self.neumann = self.compute_Neumann_integral_vars(**internal_vars)

        res = res - self.body_force - self.neumann
        return res

    def compute_residual_vars(self, sol, **internal_vars):
        logger.info(f"Computing cell residual...")
        cells_sol = sol[self.cells]  # (num_cells, num_nodes, vec)
        weak_form = self.split_and_compute_cell(
            cells_sol, np, False,
            **internal_vars)  # (num_cells, num_nodes, vec)
        return self.compute_residual_vars_helper(sol, weak_form,
                                                 **internal_vars)

    def compute_newton_vars(self, sol, **internal_vars):
        logger.info(f"Computing cell Jacobian and cell residual...")
        cells_sol = sol[self.cells]  # (num_cells, num_nodes, vec)
        # (num_cells, num_nodes, vec), (num_cells, num_nodes, vec, num_nodes, vec)
        weak_form, cells_jac = self.split_and_compute_cell(
            cells_sol, onp, True, **internal_vars)

        V = cells_jac.reshape(-1)
        inds = (self.vec * self.cells[:, :, None] +
                onp.arange(self.vec)[None, None, :]).reshape(
                    len(self.cells), -1)

        I = onp.repeat(inds[:, :, None], self.num_nodes * self.vec,
                       axis=2).reshape(-1)
        J = onp.repeat(inds[:, None, :], self.num_nodes * self.vec,
                       axis=1).reshape(-1)

        # TODO: Replace this with a sparse matrix representation - less memory
        # intensive
        # self.I = I
        # self.J = J
        # self.V = V


        if self.cauchy_bc_info is not None:
            D_face, selected_cells = self.compute_face(cells_sol, onp, True)
            V_face = D_face.reshape(-1)
            inds_face = (self.vec * selected_cells[:, :, None] +
                         onp.arange(self.vec)[None, None, :]).reshape(
                             len(selected_cells), -1)
            I_face = onp.repeat(inds_face[:, :, None],
                                self.num_nodes * self.vec,
                                axis=2).reshape(-1)
            J_face = onp.repeat(inds_face[:, None, :],
                                self.num_nodes * self.vec,
                                axis=1).reshape(-1)

            # TODO: Same as above
            I = onp.hstack((I, I_face))
            J = onp.hstack((J, J_face))
            V = onp.hstack((V, V_face))

        self.A_sp_scipy = scipy.sparse.csr_array((V, (I, J)),
                                           shape=(self.num_total_dofs, self.num_total_dofs))

        return self.compute_residual_vars_helper(sol, weak_form,
                                                 **internal_vars)

    def compute_residual(self, sol):
        return self.compute_residual_vars(sol, **self.internal_vars)

    def newton_update(self, sol):
        return self.compute_newton_vars(sol, **self.internal_vars)

    def set_params(self, params):
        """Used for solving inverse problems.
        """
        raise NotImplementedError("Child class must implement this function!")

    def print_BC_info(self):
        """Print boundary condition information for debugging purposes.
        """
        if hasattr(self, 'neumann_boundary_inds_list'):
            print(f"\n\n### Neumann B.C. is specified")
            for i in range(len(self.neumann_boundary_inds_list)):
                print(f"\nNeumann Boundary part {i + 1} information:")
                print(self.neumann_boundary_inds_list[i])
                print(f"Array.shape = (num_selected_faces, 2) = {self.neumann_boundary_inds_list[i].shape}")
                print(f"Interpretation:")
                print(f"    Array[i, 0] returns the global cell index of the ith selected face")
                print(
                    f"    Array[i, 1] returns the local face index of the ith selected face"
                )
        else:
            print(f"\n\n### No Neumann B.C. found.")

        if len(self.node_inds_list) != 0:
            print(f"\n\n### Dirichlet B.C. is specified")
            for i in range(len(self.node_inds_list)):
                print(f"\nDirichlet Boundary part {i + 1} information:")
                bc_array = onp.stack([
                    self.node_inds_list[i], self.vec_inds_list[i],
                    self.vals_list[i]
                ]).T
                print(bc_array)
                print(
                    f"Array.shape = (num_selected_dofs, 3) = {bc_array.shape}")
                print(f"Interpretation:")
                print(
                    f"    Array[i, 0] returns the node index of the ith selected dof"
                )
                print(
                    f"    Array[i, 1] returns the vec index of the ith selected dof"
                )
                print(
                    f"    Array[i, 2] returns the value assigned to ith selected dof"
                )
        else:
            print(f"\n\n### No Dirichlet B.C. found.")
