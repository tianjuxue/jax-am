import numpy as onp
import jax
import jax.numpy as np
from src.fem.jax_fem import Mesh, Laplace
from src.fem.apps.multi_scale.arguments import args
from src.fem.apps.multi_scale.trainer import get_nn_batch_forward
from src.fem.apps.multi_scale.utils import tensor_to_flat


class HyperElasticity(Laplace):
    """Three modes: rve, dns, nn
    """
    def __init__(self, name, mesh, mode=None, dirichlet_bc_info=None, neumann_bc_info=None, source_info=None, periodic_bc_info=None):
        self.name = name
        self.vec = 3
        super().__init__(mesh, dirichlet_bc_info, neumann_bc_info, source_info)
        self.mode = mode
        if self.mode == 'rve':
            self.H_bar = None
            self.physical_quad_points = self.get_physical_quad_points()
            self.E, self.nu = self.compute_moduli()
            self.periodic_bc_info = periodic_bc_info
            self.p_node_inds_list_A, self.p_node_inds_list_B, self.p_vec_inds_list = self.periodic_bc_info
        elif self.mode == 'dns':
            self.physical_quad_points = self.get_physical_quad_points()
            self.E, self.nu = self.compute_moduli()
        elif self.mode == 'nn':
            # hyperparam = 'default'
            hyperparam = 'MLP2'
            self.nn_batch_forward = get_nn_batch_forward(hyperparam)
        else:
            raise NotImplementedError(f"mode = {self.mode} is not defined.")

    def compute_residual(self, sol):
        if self.mode == 'rve' or self.mode == 'dns' :
            return self.compute_residual_vars(sol, self.E, self.nu)
        elif self.mode == 'nn':
            return self.compute_residual_vars(sol)
        else:
            raise NotImplementedError(f"compute_residual Only support rve, dns or nn.")

    def get_tensor_map(self):
        stress_map, _ = self.get_maps()
        return stress_map


    def newton_update(self, sol):
        if self.mode == 'dns':
            return self.newton_vars(sol, self.E, self.nu)
        elif self.mode == 'nn':
            return self.newton_vars(sol)
        else:
            raise NotImplementedError(f"newton_update Only support dns or nn.")

    def get_maps(self):
        if self.mode == 'rve':
            return self.maps_rve()
        elif self.mode == 'dns':
            return self.maps_dns()
        elif self.mode == 'nn':
            return self.maps_nn()
        else:
            raise NotImplementedError(f"get_maps Only support rve, dns or nn.")

    def stress_strain_fns(self):
        if self.mode == 'rve':  
            stress, psi = self.maps_rve()
            vmap_stress = lambda x: jax.vmap(stress)(x, self.E, self.nu)
            vmap_energy = lambda x: jax.vmap(psi)(x + self.H_bar[None, :, :], self.E, self.nu)
        elif self.mode == 'dns':
            stress, psi = self.maps_dns()
            vmap_stress = lambda x: jax.vmap(stress)(x, self.E, self.nu)
            vmap_energy = lambda x: jax.vmap(psi)(x, self.E, self.nu)
        elif self.mode == 'nn':
            stress, psi = self.maps_nn()
            vmap_stress = jax.vmap(stress)
            vmap_energy = jax.vmap(psi)
        else:
            raise NotImplementedError(f"get_maps Only support rve, dns or nn.")
        return vmap_stress, vmap_energy


    def maps_rve(self):
        def psi(F, E, nu):
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2. 
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, E, nu):
            I = np.eye(self.dim)
            F = u_grad + I + self.H_bar
            P = P_fn(F, E, nu)
            return P

        return first_PK_stress, psi

    def maps_dns(self):
        def psi(F, E, nu):
            mu = E/(2.*(1. + nu))
            kappa = E/(3.*(1. - 2.*nu))
            J = np.linalg.det(F)
            Jinv = J**(-2./3.)
            I1 = np.trace(F.T @ F)
            energy = (mu/2.)*(Jinv*I1 - 3.) + (kappa/2.) * (J - 1.)**2. 
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad, E, nu):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F, E, nu)
            return P

        return first_PK_stress, psi

    def maps_nn(self):
        def psi(F):
            C = F.T @ F
            C_flat = tensor_to_flat(C)
            energy = self.nn_batch_forward(C_flat[None, :])[0]
            return energy
        P_fn = jax.grad(psi)

        def first_PK_stress(u_grad):
            I = np.eye(self.dim)
            F = u_grad + I
            P = P_fn(F)
            return P
 
        return first_PK_stress, psi

    def compute_energy(self, sol):
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim) 
        u_grads = np.take(sol, self.cells, axis=0)[:, None, :, :, None] * self.shape_grads[:, :, :, None, :] 
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim) 
        F_reshape = u_grads.reshape(-1, self.vec, self.dim) + np.eye(self.dim)[None, :, :]
        _, vmap_energy  = self.stress_strain_fns()
        psi = vmap_energy(F_reshape).reshape(u_grads.shape[:2]) # (num_cells, num_quads)
        energy = np.sum(psi * self.JxW)
        return energy

    def fluc_to_disp(self, sol_fluc):
        sol_disp = (self.H_bar @ self.points.T).T + sol_fluc
        return sol_disp

    def compute_moduli(self):
        def moduli_map(point):
            inclusion = False
            for i in range(args.units_x):
                for j in range(args.units_y):
                    for k in range(args.units_z):
                        center = np.array([(i + 0.5)*args.L, (j + 0.5)*args.L, (k + 0.5)*args.L])
                        hit = np.max(np.absolute(point - center)) < args.L*args.ratio
                        inclusion = np.logical_or(inclusion, hit)
            E = np.where(inclusion, args.E_in, args.E_out)
            nu = np.where(inclusion, args.nu_in, args.nu_out)
            return np.array([E, nu])

        E, nu = jax.vmap(jax.vmap(moduli_map))(self.physical_quad_points).reshape(-1, 2).T
        return E, nu


    def compute_traction(self, location_fn, sol):
        """Not working.
        """
        def traction_fn(u_grads):
            # (num_selected_faces, num_face_quads, vec, dim) -> (num_selected_faces*num_face_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(-1, self.vec, self.dim)
            vmap_stress, _ = self.stress_strain_fns()
            sigmas = vmap_stress(u_grads_reshape).reshape(u_grads.shape)
            # TODO: a more general normals with shape (num_selected_faces, num_face_quads, dim, 1) should be supplied
            # (num_selected_faces, num_face_quads, vec, dim) @ (1, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
            normals = np.array([0., 0., 1.]).reshape((self.dim, 1))
            traction = (sigmas @ normals[None, None, :, :])[:, :, :, 0]
            return traction

        traction_integral_val = self.surface_integral(location_fn, traction_fn, sol)
        return traction_integral_val
