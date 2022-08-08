import numpy as onp
import jax
import jax.numpy as np
import os
import sys
import time
import meshio
import matplotlib.pyplot as plt
from functools import partial
import gc
from src.fem.generate_mesh import box_mesh, cylinder_mesh

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from jax.config import config
config.update("jax_enable_x64", True)

onp.random.seed(0)
onp.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True, precision=5)


global_args = {}
global_args['dim'] = 3
global_args['num_quads'] = 8
global_args['num_nodes'] = 8
global_args['num_faces'] = 6


# TODO
def safe_sqrt(x):  
    safe_x = np.where(x > 0., x, 1e-10)
    return np.sqrt(safe_x)


class Mesh():
    """A custom mesh manager might be better than just use third-party packages like meshio?
    """
    def __init__(self, points, cells):
        # TODO: Assert that cells must have correct orders 
        self.points = points
        self.cells = cells


class FEM:
    def __init__(self, mesh, dirichlet_bc_info, neumann_bc_info=None, source_info=None):
        self.mesh = mesh
        self.dirichlet_bc_info = dirichlet_bc_info
        self.neumann_bc_info = neumann_bc_info
        self.source_info = source_info
        self.fem_pre_computations()

    def fem_pre_computations(self):
        """Many quantities can be pre-computed and stored for better performance.
        """
        def get_shape_val_functions():
            """Hard-coded first order shape functions in the parent domain.
            Important: f1-f8 order must match "self.cells" by gmsh file!
            """
            f1 = lambda x: -1./8.*(x[0] - 1)*(x[1] - 1)*(x[2] - 1)
            f2 = lambda x: 1./8.*(x[0] + 1)*(x[1] - 1)*(x[2] - 1)
            f3 = lambda x: -1./8.*(x[0] + 1)*(x[1] + 1)*(x[2] - 1) 
            f4 = lambda x: 1./8.*(x[0] - 1)*(x[1] + 1)*(x[2] - 1)
            f5 = lambda x: 1./8.*(x[0] - 1)*(x[1] - 1)*(x[2] + 1)
            f6 = lambda x: -1./8.*(x[0] + 1)*(x[1] - 1)*(x[2] + 1)
            f7 = lambda x: 1./8.*(x[0] + 1)*(x[1] + 1)*(x[2] + 1)
            f8 = lambda x: -1./8.*(x[0] - 1)*(x[1] + 1)*(x[2] + 1)
            return [f1, f2, f3, f4, f5, f6, f7, f8]

        def get_shape_grad_functions():
            """Shape gradient functions
            """
            shape_fns = get_shape_val_functions()
            return [jax.grad(f) for f in shape_fns]

        def get_quad_points():
            """Pre-compute quadrature points

            Returns
            -------
            shape_vals: ndarray
                (8, 3) = (num_quads, dim)  
            """
            quad_degree = 2
            quad_points = []
            for i in range(quad_degree):
                for j in range(quad_degree):
                    for k in range(quad_degree):
                       quad_points.append([(2*(k % 2) - 1) * np.sqrt(1./3.), 
                                           (2*(j % 2) - 1) * np.sqrt(1./3.), 
                                           (2*(i % 2) - 1) * np.sqrt(1./3.)])
            quad_points = np.array(quad_points) # (quad_degree^dim, dim)
            return quad_points

        def get_face_quad_points():
            """Pre-compute face quadrature points

            Returns
            -------
            face_quad_points: ndarray
                (6, 4, 3) = (num_faces, num_face_quads, dim)  
            face_normals: ndarray
                (6, 3) = (num_faces, dim)  
            """
            face_quad_degree = 2
            face_quad_points = []
            face_normals = []
            face_extremes = np.array([-1., 1.])
            for d in range(global_args['dim']):
                for s in face_extremes:
                    s_quad_points = []
                    for i in range(face_quad_degree):
                        for j in range(face_quad_degree):
                            items = np.array([s, (2*(j % 2) - 1) * np.sqrt(1./3.), (2*(i % 2) - 1) * np.sqrt(1./3.)])
                            s_quad_points.append(list(np.roll(items, d)))            
                    face_quad_points.append(s_quad_points)
                    face_normals.append(list(np.roll(np.array([s, 0., 0.]), d)))
            face_quad_points = np.array(face_quad_points)
            face_normals = np.array(face_normals)
            return face_quad_points, face_normals

        def get_shape_vals():
            """Pre-compute shape function values

            Returns
            -------
            shape_vals: ndarray
               (8, 8) = (num_quads, num_nodes)  
            """
            shape_val_fns = get_shape_val_functions()
            quad_points = get_quad_points()
            shape_vals = []
            for quad_point in quad_points:
                physical_shape_vals = []
                for shape_val_fn in shape_val_fns:
                    physical_shape_val = shape_val_fn(quad_point) 
                    physical_shape_vals.append(physical_shape_val)
         
                shape_vals.append(physical_shape_vals)

            shape_vals = np.array(shape_vals)
            assert shape_vals.shape == (global_args['num_quads'], global_args['num_nodes'])
            return shape_vals

        @jax.jit
        def get_shape_grads():
            """Pre-compute shape function gradient value
            The gradient is w.r.t physical coordinates.
            See Hughes, Thomas JR. The finite element method: linear static and dynamic finite element analysis. Courier Corporation, 2012.
            Page 147, Eq. (3.9.3)

            Returns
            -------
            shape_grads_physical: ndarray
                (num_cells, num_quads, num_nodes, dim)  
            JxW: ndarray
                (num_cells, num_quads)
            """
            shape_grad_fns = get_shape_grad_functions()
            quad_points = get_quad_points()
            shape_grads_reference = []
            for quad_point in quad_points:
                shape_grads_ref = []
                for shape_grad_fn in shape_grad_fns:
                    shape_grad = shape_grad_fn(quad_point)
                    shape_grads_ref.append(shape_grad)
                shape_grads_reference.append(shape_grads_ref)
            shape_grads_reference = np.array(shape_grads_reference) # (num_quads, num_nodes, dim)
            assert shape_grads_reference.shape == (global_args['num_quads'], global_args['num_nodes'], global_args['dim'])

            physical_coos = np.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)

            # (num_cells, num_quads, num_nodes, dim, dim) -> (num_cells, num_quads, 1, dim, dim)
            jacobian_dx_deta = np.sum(physical_coos[:, None, :, :, None] * shape_grads_reference[None, :, :, None, :], axis=2, keepdims=True)
            jacobian_det = np.linalg.det(jacobian_dx_deta)[:, :, 0] # (num_cells, num_quads)
            jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta)
            # (1, num_quads, num_nodes, 1, dim) @ (num_cells, num_quads, 1, dim, dim) 
            # (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, dim)
            shape_grads_physical = (shape_grads_reference[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]

            # For first order FEM with 8 quad points, those quad weights are all equal to one
            quad_weights = 1.
            JxW = jacobian_det * quad_weights
            return shape_grads_physical, JxW

        def get_face_shape_vals():
            """Pre-compute face shape function values

            Returns
            -------
            face_shape_vals: ndarray
               (6, 4, 8) = (num_faces, num_face_quads, num_nodes)  
            """
            shape_val_fns = get_shape_val_functions()
            face_quad_points, _ = get_face_quad_points()
            face_shape_vals = []
            for f_quad_points in face_quad_points:
                f_shape_vals = []
                for quad_point in f_quad_points:
                    physical_shape_vals = []
                    for shape_val_fn in shape_val_fns:
                        physical_shape_val = shape_val_fn(quad_point) 
                        physical_shape_vals.append(physical_shape_val)
                    f_shape_vals.append(physical_shape_vals)
                face_shape_vals.append(f_shape_vals)
            face_shape_vals = np.array(face_shape_vals)
            return face_shape_vals


        def get_face_inds():
            """Hard-coded reference node points.
            Important: order must match "self.cells" by gmsh file!

            TODO: move this function to a more suitable place

            Returns
            ------- 
            face_inds: ndarray
                (6, 4) = (num_faces, num_face_quads)
            """
            # TODO: Hard-coded
            node_points = np.array([[-1., -1., -1.],
                                    [1., -1, -1.],
                                    [1., 1., -1.],
                                    [-1., 1., -1.],
                                    [-1., -1., 1.],
                                    [1., -1, 1.],
                                    [1., 1., 1.],
                                    [-1., 1., 1.]])
            face_inds = []
            face_extremes = np.array([-1., 1.])
            for d in range(global_args['dim']):
                for s in face_extremes:
                    face_inds.append(np.argwhere(np.isclose(node_points[:, d], s)).reshape(-1))
            face_inds = np.array(face_inds)
            return face_inds

        self.points = self.mesh.points
        self.cells = self.mesh.cells

        global_args['num_cells'] = len(self.cells)
        global_args['num_total_vertices'] = len(self.mesh.points)
        self.shape_vals = get_shape_vals()
        self.shape_grads, self.JxW = get_shape_grads()

        # TODO
        self.face_inds = get_face_inds()
        self.face_shape_vals = get_face_shape_vals()
        self.shape_grad_fns = get_shape_grad_functions()
        self.face_quad_points, self.face_normals = get_face_quad_points()


    # @partial(jax.jit, static_argnums=(0))
    def get_face_Nanson_scale(self, boundary_inds):
        """Pre-compute face JxW (for surface integral)
        Nanson's formula is used to map physical surface ingetral to reference domain
        Reference: https://en.wikiversity.org/wiki/Continuum_mechanics/Volume_change_and_area_change

        Returns
        ------- 
        face_shape_grads_physical: ndarray
            (num_selected_faces, num_face_quads, num_nodes, dim)
        nanson_scale: ndarray
            (num_selected_faces, num_face_quads)
        """

        # shape_grad_fns = get_shape_grad_functions()
        # face_quad_points, face_normals = get_face_quad_points() # _, (num_faces, dim)

        face_shape_grads_reference = []
        for f_quad_points in self.face_quad_points:
            f_shape_grads_ref = []
            for f_quad_point in f_quad_points:
                f_shape_grads = []
                for shape_grad_fn in self.shape_grad_fns:
                    f_shape_grad = shape_grad_fn(f_quad_point)
                    f_shape_grads.append(f_shape_grad)
                f_shape_grads_ref.append(f_shape_grads)
            face_shape_grads_reference.append(f_shape_grads_ref)

        face_shape_grads_reference = np.array(face_shape_grads_reference) # (num_faces, num_face_quads, num_nodes, dim)

        physical_coos = np.take(self.points, self.cells, axis=0) # (num_cells, num_nodes, dim)

        selected_coos = physical_coos[boundary_inds[:, 0]] # (num_selected_faces, num_nodes, dim)
        selected_f_shape_grads_ref = face_shape_grads_reference[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes, dim)
        selected_f_normals = self.face_normals[boundary_inds[:, 1]] # (num_selected_faces, dim)

        # (num_selected_faces, 1, num_nodes, dim, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        # (num_selected_faces, num_face_quads, num_nodes, dim, dim) -> (num_selected_faces, num_face_quads, dim, dim)
        jacobian_dx_deta = np.sum(selected_coos[:, None, :, :, None] * selected_f_shape_grads_ref[:, :, :, None, :], axis=2)
        jacobian_det = np.linalg.det(jacobian_dx_deta) # (num_selected_faces, num_face_quads)
        jacobian_deta_dx = np.linalg.inv(jacobian_dx_deta) # (num_selected_faces, num_face_quads, dim, dim)

        # (1, num_face_quads, num_nodes, 1, dim) @ (num_selected_faces, num_face_quads, 1, dim, dim)
        # (num_selected_faces, num_face_quads, num_nodes, 1, dim) -> (num_selected_faces, num_face_quads, num_nodes, dim)
        face_shape_grads_physical = (selected_f_shape_grads_ref[:, :, :, None, :] @ jacobian_deta_dx[:, :, None, :, :])[:, :, :, 0, :]


        # (num_selected_faces, 1, 1, dim) @ (num_selected_faces, num_face_quads, dim, dim)
        # (num_selected_faces, num_face_quads, 1, dim) -> (num_selected_faces, num_face_quads)
        nanson_scale = np.linalg.norm((selected_f_normals[:, None, None, :] @ jacobian_deta_dx)[:, :, 0, :], axis=-1)
        quad_weights = 1.
        nanson_scale = nanson_scale * jacobian_det * quad_weights
        return face_shape_grads_physical, nanson_scale


    def get_physical_quad_points(self):
        """Compute physical quadrature points
 
        Returns
        ------- 
        physical_quad_points: ndarray
            (num_cells, num_quads, dim) 
        """
        physical_coos = np.take(self.points, self.cells, axis=0)
        # (1, num_quads, num_nodes, 1) * (num_cells, 1, num_nodes, dim) -> (num_cells, num_quads, dim) 
        physical_quad_points = np.sum(self.shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2)
        return physical_quad_points


    def get_physical_surface_quad_points(self, boundary_inds):
        """Compute physical quadrature points on the surface
 
        Returns
        ------- 
        physical_surface_quad_points: ndarray
            (num_selected_faces, num_face_quads, dim) 
        """
        physical_coos = np.take(self.points, self.cells, axis=0)
        selected_coos = physical_coos[boundary_inds[:, 0]] # (num_selected_faces, num_nodes, dim)
        selected_face_shape_vals = self.face_shape_vals[boundary_inds[:, 1]] # (num_selected_faces, num_face_quads, num_nodes)  
        # (num_selected_faces, num_face_quads, num_nodes, 1) * (num_selected_faces, 1, num_nodes, dim) -> (num_selected_faces, num_face_quads, dim) 
        physical_surface_quad_points = np.sum(selected_face_shape_vals[:, :, :, None] * selected_coos[:, None, :, :], axis=2)
        return physical_surface_quad_points


    def Dirichlet_boundary_conditions(self):
        """
        """
        location_fns, vecs, value_fns = self.dirichlet_bc_info
        # TODO: add assertion for vecs, vecs must only contain 0 or 1 or 2, and must be integer
        assert len(location_fns) == len(value_fns) and len(value_fns) == len(vecs)
        inds_node_list = []
        vec_inds_list = []
        vals_list = []
        for i in range(len(location_fns)):
            node_inds = np.argwhere(jax.vmap(location_fns[i])(self.mesh.points)).reshape(-1)
            vec_inds = np.ones_like(node_inds, dtype=np.int32)*vecs[i]
            values = jax.vmap(value_fns[i])(self.mesh.points[node_inds])
            inds_node_list.append(node_inds)
            vec_inds_list.append(vec_inds)
            vals_list.append(values)
        return inds_node_list, vec_inds_list, vals_list


    def Neuman_boundary_conditions_inds(self, location_fns):
        """
        """
        # location_fns, _ = self.neumann_bc_info
        cell_points = np.take(self.points, self.cells, axis=0)
        cell_face_points = np.take(cell_points, self.face_inds, axis=1) # (num_cells, num_faces, num_face_nodes, dim)
        boundary_inds_list = []
        for i in range(len(location_fns)):
            vmap_location_fn = jax.vmap(location_fns[i])
            def on_boundary(cell_points):
                boundary_flag = vmap_location_fn(cell_points)
                return np.all(boundary_flag)
            vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
            boundary_flags = vvmap_on_boundary(cell_face_points)
            boundary_inds = np.argwhere(boundary_flags) # (num_selected_faces, 2)
            boundary_inds_list.append(boundary_inds)
        return boundary_inds_list


    def Neuman_boundary_conditions_vals(self, value_fns, boundary_inds_list):
        cell_points = np.take(self.points, self.cells, axis=0)
        cell_face_points = np.take(cell_points, self.face_inds, axis=1) # (num_cells, num_faces, num_face_nodes, dim)
        traction_list = []
        for i in range(len(value_fns)):
            boundary_inds = boundary_inds_list[i]
            # (num_cells, num_faces, num_face_quads, dim) -> (num_selected_faces, num_face_quads, dim)
            subset_quad_points = self.get_physical_surface_quad_points(boundary_inds)
            traction = jax.vmap(jax.vmap(value_fns[i]))(subset_quad_points) # (num_selected_faces, num_face_quads, vec)
            assert len(traction.shape) == 3
            traction_list.append(traction)
        return traction_list


    def surface_integral(self, location_fn, physics_fn, dofs):
        """For post-processing only
        """
        boundary_inds = self.Neuman_boundary_conditions_inds([location_fn])[0]
        face_shape_grads_physical, nanson_scale = self.get_face_Nanson_scale(boundary_inds)
        # (num_selected_faces, 1, num_nodes, vec, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
        u_grads_face = dofs[self.cells][boundary_inds[:, 0]][:, None, :, :, None] * face_shape_grads_physical[:, :, :, None, :]
        u_grads_face = np.sum(u_grads_face, axis=2) # (num_selected_faces, num_face_quads, vec, dim)
        tmp = np.ones_like(u_grads_face)
        # (num_selected_faces, num_nodes, vec) * (num_selected_faces, num_face_quads, 1)
        int_val = np.sum(tmp[:, :, :, 0] * nanson_scale[:, :, None], axis=(0, 1))
        return int_val



class Laplace(FEM):
    def __init__(self, mesh, dirichlet_bc_info, neumann_bc_info=None, source_info=None):
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info) 
        # Some pre-computations   
        self.rhs = self.compute_rhs()
        self.neumann = self.compute_Neumann_integral()
        # (num_cells, num_quads, num_nodes, 1, dim)
        self.v_grads_JxW = self.shape_grads[:, :, :, None, :] * self.JxW[:, :, None, None, None]

    def compute_residual(self, dofs):
        """The function takes a lot of memory - Thinking about ways for memory saving...
        E.g., (num_cells, num_quads, num_nodes, vec, dim) takes 4.6G memory for num_cells=1,000,000

        Parameters
        ----------
        dofs: ndarray
            (num_nodes, vec) 
        """
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(dofs, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)  
        u_physics = self.compute_physics(dofs, u_grads) # (num_cells, num_quads, vec, dim)  
        # (num_cells, num_quads, num_nodes, vec, dim) -> (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
        weak_form = np.sum(u_physics[:, :, None, :, :] * self.v_grads_JxW, axis=(1, -1)).reshape(-1, self.vec) 
        res = np.zeros_like(dofs)
        res = res.at[self.cells.reshape(-1)].add(weak_form)
        return res - self.rhs - self.neumann

    def compute_physics(self, dofs, u_grads):
        """Default
        """
        return u_grads

    def compute_rhs(self):
        """Default
        """
        rhs = np.zeros((global_args['num_total_vertices'], self.vec))
        if self.source_info is not None:
            body_force_fn = self.source_info
            physical_quad_points = self.get_physical_quad_points() # (num_cells, num_quads, dim) 
            body_force = jax.vmap(jax.vmap(body_force_fn))(physical_quad_points) # (num_cells, num_quads, vec) 
            assert len(body_force.shape) == 3
            v_vals = np.repeat(self.shape_vals[None, :, :, None], global_args['num_cells'], axis=0) # (num_cells, num_quads, num_nodes, 1)
            v_vals = np.repeat(v_vals, self.vec, axis=-1) # (num_cells, num_quads, num_nodes, vec)
            # (num_cells, num_nodes, vec) -> (num_cells*num_nodes, vec)
            rhs_vals = np.sum(v_vals * body_force[:, :, None, :] * self.JxW[:, :, None, None], axis=1).reshape(-1, self.vec) 
            rhs = rhs.at[self.cells.reshape(-1)].add(rhs_vals) 
        return rhs

    def compute_Neumann_integral(self):
        integral = np.zeros((global_args['num_total_vertices'], self.vec))
        if self.neumann_bc_info is not None:
            location_fns, value_fns = self.neumann_bc_info
            integral = np.zeros((global_args['num_total_vertices'], self.vec))
            boundary_inds_list = self.Neuman_boundary_conditions_inds(location_fns)
            traction_list = self.Neuman_boundary_conditions_vals(value_fns, boundary_inds_list)
            for i, boundary_inds in enumerate(boundary_inds_list):
                traction = traction_list[i]

                # # (num_cells, num_faces, num_face_quads) -> (num_selected_faces, num_face_quads)
                # scale = self.face_scale[boundary_inds[:, 0], boundary_inds[:, 1]]

                _, nanson_scale = self.get_face_Nanson_scale(boundary_inds) # (num_selected_faces, num_face_quads)

                # (num_faces, num_face_quads, num_nodes) ->  (num_selected_faces, num_face_quads, num_nodes)
                v_vals = np.take(self.face_shape_vals, boundary_inds[:, 1], axis=0)
                v_vals = np.repeat(v_vals[:, :, :, None], self.vec, axis=-1) # (num_selected_faces, num_face_quads, num_nodes, vec)
                subset_cells = np.take(self.cells, boundary_inds[:, 0], axis=0) # (num_selected_faces, num_nodes)
                # (num_selected_faces, num_nodes, vec) -> (num_selected_faces*num_nodes, vec)
                int_vals = np.sum(v_vals * traction[:, :, None, :] * nanson_scale[:, :, None, None], axis=1).reshape(-1, self.vec) 
                integral = integral.at[subset_cells.reshape(-1)].add(int_vals)   
        return integral


class LinearPoisson(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 1
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)
        # self.name = 'linear_poisson'


class NonelinearPoisson(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 1
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)
        # self.name = 'nonlinear_poisson'

    def compute_physics(self, dofs, u_grads):
        """

        Parameters
        ----------
        u_grads: ndarray
            (num_cells, num_quads, vec, dim)
        """
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, num_nodes, vec)
        u_vals = np.take(dofs, self.cells, axis=0)[:, None, :, :] * self.shape_vals[None, :, :, None] 
        u_vals = np.sum(u_vals, axis=2) # (num_cells, num_quads, vec)
        q = (1 + u_vals**2)[:, :, :, :, None] # (num_cells, num_quads, vec, 1)
        return q * u_grads


class LinearElasticity(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)

    def compute_physics(self, dofs, u_grads):
        """

        Parameters
        ----------
        u_grads: ndarray
            (num_cells, num_quads, vec, dim)
        """

        # Remark: This is faster than double vmap by reducing ~30% computing time
        u_grads_reshape = u_grads.reshape(-1, self.vec, global_args['dim'])

        def stress(u_grad):
            E = 100.
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(global_args['dim']) + 2*mu*epsilon
            return sigma

        vmap_stress = jax.vmap(stress)
        sigmas = vmap_stress(u_grads_reshape).reshape(u_grads.shape)
        return sigmas


class HyperElasticity(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)


class Plasticity(Laplace):
    def __init__(self, name, mesh, dirichlet_bc_info, neumann_bc_info=None, source_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)
        self.epsilon_old = np.zeros((len(self.cells)*global_args['num_quads'], self.vec, global_args['dim']))
        self.sigma_old = np.zeros_like(self.epsilon_old)


    def stress_strain_fns(self):

        def strain(u_grad):
            epsilon = 0.5*(u_grad + u_grad.T)
            return epsilon

        def stress(u_grad, sigma_old, epsilon_old):
            E = 70e3
            nu = 0.3
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            sig0 = 250.

            epsilon_crt = strain(u_grad)

            epsilon_inc = epsilon_crt - epsilon_old

            sigma_trial = lmbda*np.trace(epsilon_inc)*np.eye(global_args['dim']) + 2*mu*epsilon_inc + sigma_old

            s = sigma_trial - 1./global_args['dim']*np.trace(sigma_trial)*np.eye(global_args['dim'])

            # s_norm = np.sqrt(3./2.*np.sum(s*s))

            s_norm = safe_sqrt(3./2.*np.sum(s*s))

            f_yield = s_norm - sig0

            f_yield_plus = np.where(f_yield > 0., f_yield, 0.)

            # TODO
            sigma = sigma_trial - f_yield_plus*s/(s_norm + 1e-10)

            # sigma = sigma_trial
 
            return sigma
            # return f_yield_plus

        vmap_stress = jax.vmap(stress)
        vmap_strain = jax.vmap(strain)

        return vmap_stress, vmap_strain


    def compute_physics(self, dofs, u_grads):
        """
        Reference: https://comet-fenics.readthedocs.io/en/latest/demo/2D_plasticity/vonMises_plasticity.py.html

        Parameters
        ----------
        u_grads: ndarray
            (num_cells, num_quads, vec, dim)
        """
        u_grads_reshape = u_grads.reshape(-1, self.vec, global_args['dim'])
        vmap_stress, _ = self.stress_strain_fns()
        sigmas = vmap_stress(u_grads_reshape, self.sigma_old, self.epsilon_old).reshape(u_grads.shape)
        return sigmas



    def update_stress_strain(self, dofs):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(dofs, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)  

        u_grads_reshape = u_grads.reshape(-1, self.vec, global_args['dim'])
        vmap_stress, vmap_strain = self.stress_strain_fns()
        sigmas = vmap_stress(u_grads_reshape, self.sigma_old, self.epsilon_old).reshape(u_grads.shape)
        epsilons = vmap_strain(u_grads_reshape)

        self.sigma_old = sigmas
        self.epsilons = epsilons



def save_sol(problem, sol, sol_file):
    out_mesh = meshio.Mesh(points=problem.points, cells={'hexahedron': problem.cells})
    out_mesh.point_data['sol'] = onp.array(sol.reshape((global_args['num_total_vertices'], problem.vec)), dtype=onp.float32)
    # out_mesh.write(f"post-processing/vtk/fem/jax_{self.name}.vtu")

    out_mesh.write(sol_file)


def solver(problem, initial_guess=None):
    def operator_to_matrix(operator_fn):
        J = jax.jacfwd(operator_fn)(np.zeros(global_args['num_total_vertices']*problem.vec))
        return J

    def apply_bc(res_fn, node_inds_list, vec_inds_list, vals_list):
        def A_fn(dofs):
            """Apply Dirichlet boundary conditions
            """
            dofs = dofs.reshape(global_args['num_total_vertices'], problem.vec)
            res = res_fn(dofs)
            for i in range(len(node_inds_list)):
                res = res.at[node_inds_list[i], vec_inds_list[i]].set(dofs[node_inds_list[i], vec_inds_list[i]], unique_indices=True)
                res = res.at[node_inds_list[i], vec_inds_list[i]].add(-vals_list[i])
            return res.reshape(-1)
        
        return A_fn

    def get_A_fn_linear_fn(sol):
        def A_fn_linear_fn(inc):
            primals, tangents = jax.jvp(A_fn, (sol,), (inc,))
            return tangents
        return A_fn_linear_fn

    def get_A_fn_linear_fn_JFNK(sol):
        def A_fn_linear_fn(inc):
            EPS = 1e-3
            return (A_fn(sol + EPS*inc) - A_fn(sol))/EPS
        return A_fn_linear_fn

    res_fn = problem.compute_residual
    node_inds_list, vec_inds_list, vals_list = problem.Dirichlet_boundary_conditions()
    A_fn = apply_bc(res_fn, node_inds_list, vec_inds_list, vals_list)

    print("Done pre-computing and start timing")
    start = time.time()




    if initial_guess is not None:
        sol = initial_guess.reshape((global_args['num_total_vertices'], problem.vec))
    else:
        sol = np.zeros((global_args['num_total_vertices'], problem.vec))




    for i in range(len(node_inds_list)):
        sol = sol.at[node_inds_list[i], vec_inds_list[i]].set(vals_list[i])
    sol = sol.reshape(-1)

    tol = 1e-6  
    step = 0
    b = -A_fn(sol)
    res_val = np.linalg.norm(b)
    print(f"step = {step}, res l_2 = {res_val}") 
    while res_val > tol:
        A_fn_linear = get_A_fn_linear_fn(sol)
        debug = False
        if debug:
            # Check onditional number of the matrix
            A_dense = operator_to_matrix(A_fn_linear)
            print(np.linalg.cond(A_dense))
            print(np.max(A_dense))
            # print(A_dense)

        inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000) # bicgstab

        # print(np.max(inc))
        # exit()

        sol = sol + inc
        b = -A_fn(sol)
        res_val = np.linalg.norm(b)
        step += 1
        print(f"step = {step}, res l_2 = {res_val}") 

    end = time.time()
    solve_time = end - start
    print(f"Solve took {solve_time} [s], finished in {step} steps")
    print(f"max of sol = {np.max(sol)}")
    print(f"min of sol = {np.min(sol)}")

    return sol


def debug():
    mesh = cylinder_mesh()
    cells = mesh.cells_dict['hexahedron'] 
    points = mesh.points

    # 0, 14 useless
    points = np.vstack((points[1:14], points[15:]))
    cells = onp.where(cells > 14, cells - 2, cells - 1)

    mesh = Mesh(points, cells)

    H = 10.
    R = 5.

    def top(point):
        return np.isclose(point[2], H, atol=1e-5)

    def bottom(point):
        return np.isclose(point[2], 0., atol=1e-5)

    def zero_val(point):
        return 0.

    def nonzero_val(point):
        return 1.

    location_fns = [bottom, bottom, bottom, top, top, top]
    value_fns = [zero_val, zero_val, zero_val, zero_val, zero_val, nonzero_val]
    vecs = [0, 1, 2, 0, 1, 2]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    # problem = LinearElasticity('plasticity_cylinder', mesh, dirichlet_bc_info)
    # sol = solver(problem)

    problem = Plasticity('plasticity_cylinder', mesh, dirichlet_bc_info)
    sol = solver(problem)


def linear_elasticity():
    meshio_mesh = box_mesh(100, 100, 100)
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict['hexahedron'])
    del meshio_mesh
    gc.collect()

    def left(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right(point):
        return np.isclose(point[0], 1., atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    def dirichlet_val(point):
        return 0.1

    def neumann_val(point):
        return np.array([10., 0., 0.])

    def body_force(point):
        return np.array([point[0], 2.*point[1], 3.*point[2]])

    dirichlet_bc_info = [[left, left, left], [0, 1, 2], [dirichlet_val, dirichlet_val, dirichlet_val]]
    neumann_bc_info = [[right], [neumann_val]]

    dirichlet_bc_info = [[left, left, left, right, right, right], 
                         [0, 1, 2, 0, 1, 2], 
                         [zero_dirichlet_val, zero_dirichlet_val, zero_dirichlet_val, 
                          dirichlet_val, zero_dirichlet_val, zero_dirichlet_val]]
    neumann_bc_info = None
    body_force = None

    problem = LinearElasticity('linear_elasticity', mesh, dirichlet_bc_info, neumann_bc_info, body_force)
    sol = solver(problem)


if __name__ == "__main__":
    debug()
    # linear_elasticity()

