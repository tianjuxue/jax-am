import numpy as onp
import jax.numpy as np
import jax
import meshio
import os
import time
import glob
 
from jax_am.common import make_video, box_mesh

onp.random.seed(0)

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

case_name = 'surface_tension'
data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
os.makedirs(vtk_dir, exist_ok=True)

# TODO: phi near obstacle should never be used


def simulation():
    def to_id(idx, idy, idz):
        return idx * Ny * Nz + idy * Nz + idz

    def to_id_xyz(lattice_id):
        id_z = lattice_id % Nz
        lattice_id = lattice_id // Nz
        id_y = lattice_id % Ny
        id_x = lattice_id // Ny    
        return id_x, id_y, id_z 

    def extract_7x3x3_grid(lattice_id, values):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        grid_index = np.ix_(np.array([(id_x - 3) % Nx, (id_x - 2) % Nx, (id_x - 1) % Nx, id_x, (id_x + 1) % Nx, (id_x + 2) % Nx, (id_x + 3) % Nx]), 
                            np.array([(id_y - 1) % Ny, id_y, (id_y + 1) % Ny]),
                            np.array([(id_z - 1) % Nz, id_z, (id_z + 1) % Nz]))
        grid_values = values[grid_index]
        return grid_values 

    def extract_3x7x3_grid(lattice_id, values):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        grid_index = np.ix_(np.array([(id_x - 1) % Nx, id_x, (id_x + 1) % Nx]),  
                            np.array([(id_y - 3) % Ny, (id_y - 2) % Ny, (id_y - 1) % Ny, id_y, (id_y + 1) % Ny, (id_y + 2) % Ny, (id_y + 3) % Ny]),
                            np.array([(id_z - 1) % Nz, id_z, (id_z + 1) % Nz]))
        grid_values = values[grid_index]
        return grid_values 

    def extract_3x3x7_grid(lattice_id, values):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        grid_index = np.ix_(np.array([(id_x - 1) % Nx, id_x, (id_x + 1) % Nx]), 
                            np.array([(id_y - 1) % Ny, id_y, (id_y + 1) % Ny]),
                            np.array([(id_z - 3) % Nz, (id_z - 2) % Nz, (id_z - 1) % Nz, id_z, (id_z + 1) % Nz, (id_z + 2) % Nz, (id_z + 3) % Nz]))
        grid_values = values[grid_index]
        return grid_values  

    def extract_3x3x3_grid(lattice_id, values):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        grid_index = np.ix_(np.array([(id_x - 1) % Nx, id_x, (id_x + 1) % Nx]), 
                            np.array([(id_y - 1) % Ny, id_y, (id_y + 1) % Ny]),
                            np.array([(id_z - 1) % Nz, id_z, (id_z + 1) % Nz]))
        grid_values = values[grid_index]
        return grid_values        

    def extract_local(lattice_id, values):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        return values[vels[0] + id_x, vels[1] + id_y, vels[2] + id_z]

    def extract_income(lattice_id, values):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        return values[vels[0] + id_x, vels[1] + id_y, vels[2] + id_z, rev]

    def extract_self(lattice_id, values):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        return values[id_x, id_y, id_z]

    def shape_wrapper(f):
        def shape_wrapper(*args):
            return jax.tree_util.tree_map(lambda x: x.reshape((Nx, Ny, Nz) + x.shape[1:]), f(*args))
        return shape_wrapper


    def initialize_phase_surface_tension(lattice_id, cell_centroids):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        flag_x = np.logical_and(cell_centroids[lattice_id, 0] > 0.2 * domain_x, cell_centroids[lattice_id, 0] < 0.8 * domain_x)
        flag_y = np.logical_and(cell_centroids[lattice_id, 1] > 0.2 * domain_y, cell_centroids[lattice_id, 1] < 0.8 * domain_y)
        flag_z = np.logical_and(cell_centroids[lattice_id, 2] > 0.2 * domain_z, cell_centroids[lattice_id, 2] < 0.8 * domain_z)


        # flag_x = np.logical_and(cell_centroids[lattice_id, 0] > 0.4 * domain_x, cell_centroids[lattice_id, 0] < 0.6 * domain_x)
        # flag_y = np.logical_and(cell_centroids[lattice_id, 1] > 0.4 * domain_y, cell_centroids[lattice_id, 1] < 0.6 * domain_y)
        # flag_z = np.logical_and(cell_centroids[lattice_id, 2] > 0.3 * domain_z, cell_centroids[lattice_id, 2] < 0.8 * domain_z)

        flag = np.logical_and(np.logical_and(flag_x, flag_y), flag_z)

        # flag =  cell_centroids[lattice_id, 2] < 0.5 * domain_z

        tmp = np.where(flag, LIQUID, GAS)
        wall_x = np.logical_or(id_x == 0, id_x == Nx - 1)
        wall_y = np.logical_or(id_y == 0, id_y == Ny - 1)
        wall_z = np.logical_or(id_z == 0, id_z == Nz - 1)
        wall = np.logical_or(wall_x, np.logical_or(wall_y, wall_z))
        return np.where(wall, WALL, tmp)

    initialize_phase_surface_tension_vmap = shape_wrapper(jax.vmap(initialize_phase_surface_tension, in_axes=(0, None)))


    def initialize_phase_free_fall(lattice_id, cell_centroids):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        flag_x = cell_centroids[lattice_id, 0] < 0.4 * domain_x
        flag_y = cell_centroids[lattice_id, 1] < 0.4 * domain_y
        flag_z = cell_centroids[lattice_id, 2] < 0.8 * domain_z
        flag = np.logical_and(np.logical_and(flag_x, flag_y), flag_z)
        tmp = np.where(flag, LIQUID, GAS)
        wall_x = np.logical_or(id_x == 0, id_x == Nx - 1)
        wall_y = np.logical_or(id_y == 0, id_y == Ny - 1)
        wall_z = np.logical_or(id_z == 0, id_z == Nz - 1)
        wall = np.logical_or(wall_x, np.logical_or(wall_y, wall_z))
        return np.where(wall, WALL, tmp)

    initialize_phase_free_fall_vmap = shape_wrapper(jax.vmap(initialize_phase_free_fall, in_axes=(0, None)))



    def generate_powder():
        def check_overlap(circles, cand_c):
            for c in circles:
                if (c[0] - cand_c[0])**2 + (c[1] - cand_c[1])**2 < (c[2] + cand_c[2])**2:
                    return True
            return False

        circles = []
        num_circles = 90
        # num_circles = 0
        
        r_mean = 0.08*domain_y
        for i in range(num_circles):
            while True:
                r = onp.random.normal(loc=r_mean, scale=0.2*r_mean)
                x = onp.random.uniform(low=r, high=domain_x - r)
                y = onp.random.uniform(low=r, high=domain_y - r)
                cand_c = [x, y, r]
                if not check_overlap(circles, cand_c):
                    circles.append(cand_c)
                    break
        return circles


    def initialize_phase_powder(lattice_id, cell_centroids, circles):
        id_x, id_y, id_z = to_id_xyz(lattice_id)

        plate_z = 0.5 * domain_z
        flag =  cell_centroids[lattice_id, 2] < plate_z

        # # r = 0.1*domain_y
        # r = 0.18*domain_y
        # center1 = np.array([0.8*domain_y, 0.5*domain_y, 0.6*domain_y])
        # center2 = np.array([1.2*domain_y, 0.5*domain_y, 0.6*domain_y])

        # flag = np.logical_or(np.linalg.norm(cell_centroids[lattice_id] - center1) < r, flag)
        # flag = np.logical_or(np.linalg.norm(cell_centroids[lattice_id] - center2) < r, flag)

 
        for circle in circles:
            x, y, r = circle
            z = plate_z + r
            flag = np.logical_or(np.linalg.norm(cell_centroids[lattice_id] - np.array([x, y, z])) < r, flag)


        tmp = np.where(flag, LIQUID, GAS)
        wall_x = np.logical_or(id_x == 0, id_x == Nx - 1)
        wall_y = np.logical_or(id_y == 0, id_y == Ny - 1)
        wall_z = np.logical_or(id_z == 0, id_z == Nz - 1)
        wall = np.logical_or(wall_x, np.logical_or(wall_y, wall_z))
        return np.where(wall, WALL, tmp)

    initialize_phase_powder_vmap = shape_wrapper(jax.vmap(initialize_phase_powder, in_axes=(0, None, None)))


    def equilibrium_f(q, rho, u):
        vel_dot_u = np.sum(vels[:, q] * u)
        u_sq = np.sum(u * u)
        return weights[q] * rho * (1. + vel_dot_u/cs_sq + vel_dot_u**2./(2.*cs_sq**2.) - u_sq/(2.*cs_sq))

    equilibrium_f_vmap = jax.vmap(equilibrium_f, in_axes=(0, None, None))


    def equilibrium_h(q, enthalpy, T, u):
        u_sq = np.sum(u * u)
        vel_dot_u = np.sum(vels[:, q] * u)
        result = weights[q]*heat_capacity*T*(1. + vel_dot_u/cs_sq + vel_dot_u**2./(2.*cs_sq**2.) - u_sq/(2.*cs_sq))
        result0 = enthalpy - heat_capacity*T + weights[0]*heat_capacity*T*(1 - u_sq/(2.*cs_sq))
        return np.where(q == 0, result0, result)

    equilibrium_h_vmap = jax.vmap(equilibrium_h, in_axes=(0, None, None, None))


    def f_forcing(q, u, volume_force):
        ei = vels[:, q]
        return (1. - 1./(2.*tau_viscosity_nu)) * weights[q] * np.sum(((ei - u)/cs_sq + np.sum(ei * u)/cs_sq**2 * ei) * volume_force)

    f_forcing_vmap = jax.vmap(f_forcing, in_axes=(0, None, None))


    def h_forcing_vmap(volume_power, rho):
        return volume_power / rho * weights


    @jax.jit
    def compute_rho(f_distribute):
        rho = np.sum(f_distribute, axis=-1)
        return rho

    @jax.jit
    def compute_enthalpy(h_distribute):
        enthalpy = np.sum(h_distribute, axis=-1)
        return enthalpy

    @jax.jit
    def compute_T(enthalpy):
        T = np.where(enthalpy < enthalpy_s, enthalpy/heat_capacity, 
            np.where(enthalpy < enthalpy_l, T_solidus + (enthalpy - enthalpy_s)/(enthalpy_l - enthalpy_s)*(T_liquidus - T_solidus), 
                                            T_liquidus + (enthalpy - enthalpy_l)/heat_capacity))
        return T

    @jax.jit
    def compute_vof(rho, phase, mass):
        vof = np.where(phase == LIQUID, rho, mass)
        vof = np.where(phase == GAS, 0., vof)
        vof = np.where(phase == WALL, rho0, vof)
        return vof

    def compute_curvature(lattice_ids, vof):
        def get_phi(lattice_id, vof):
            vof_grid = extract_3x3x3_grid(lattice_id, vof) # (3, 3, 3)
            c_id = 1
            phi_x = (vof_grid[c_id + 1, c_id, c_id] - vof_grid[c_id - 1, c_id, c_id])/(2.*h)
            phi_y = (vof_grid[c_id, c_id + 1, c_id] - vof_grid[c_id, c_id - 1, c_id])/(2.*h)
            phi_z = (vof_grid[c_id, c_id, c_id + 1] - vof_grid[c_id, c_id, c_id - 1])/(2.*h)
            return np.array([phi_x, phi_y, phi_z])

        get_phi_vmap = jax.jit(jax.vmap(get_phi, in_axes=(0, None)))

        def kappa_7x3x3(lattice_id, vof):
            vof_grid = extract_7x3x3_grid(lattice_id, vof) # (7, 3, 3)
            return np.sum(vof_grid, axis=0)

        hf_7x3x3_vmap = jax.jit(jax.vmap(kappa_7x3x3, in_axes=(0, None)))

        def kappa_3x7x3(lattice_id, vof):
            vof_grid = extract_3x7x3_grid(lattice_id, vof) # (3, 7, 3)
            return np.sum(vof_grid, axis=1)

        hf_3x7x3_vmap = jax.jit(jax.vmap(kappa_3x7x3, in_axes=(0, None)))

        def kappa_3x3x7(lattice_id, vof):
            vof_grid = extract_3x3x7_grid(lattice_id, vof) # (3, 3, 7)
            return np.sum(vof_grid, axis=2)

        hf_3x3x7_vmap = jax.jit(jax.vmap(kappa_3x3x7, in_axes=(0, None)))

        def curvature(hgt_func):
            c_id = 1
            Hx = (hgt_func[..., c_id + 1, c_id] - hgt_func[..., c_id - 1, c_id])/(2.*h)
            Hy = (hgt_func[..., c_id, c_id + 1] - hgt_func[..., c_id, c_id - 1])/(2.*h)
            Hxx = (hgt_func[..., c_id + 1, c_id] - 2.*hgt_func[..., c_id, c_id] + hgt_func[..., c_id - 1, c_id])/h**2
            Hyy = (hgt_func[..., c_id, c_id + 1] - 2.*hgt_func[..., c_id, c_id] + hgt_func[..., c_id, c_id - 1])/h**2
            Hxy = (hgt_func[..., c_id + 1, c_id + 1] - hgt_func[..., c_id - 1, c_id + 1] - hgt_func[..., c_id + 1, c_id - 1] 
                 + hgt_func[..., c_id - 1, c_id - 1])/(4*h)
            kappa = -(Hxx + Hyy + Hxx*Hy**2. + Hyy*Hx**2. - 2.*Hxy*Hx*Hy) / (1 + Hx**2. + Hy**2.)**(3./2.)
            kappa = np.where(np.isfinite(kappa), kappa, 0.)
            return kappa

        kappa_1 = curvature(hf_7x3x3_vmap(lattice_ids, vof))
        kappa_2 = curvature(hf_3x7x3_vmap(lattice_ids, vof))
        kappa_3 = curvature(hf_3x3x7_vmap(lattice_ids, vof))

        phi = get_phi_vmap(lattice_ids, vof)
            
        kappa = np.where(np.logical_and(np.absolute(phi[..., 0]) >= np.absolute(phi[..., 1]), np.absolute(phi[..., 0]) >= np.absolute(phi[..., 2])), kappa_1, 
               np.where(np.logical_and(np.absolute(phi[..., 1]) >= np.absolute(phi[..., 0]), np.absolute(phi[..., 1]) >= np.absolute(phi[..., 2])), kappa_2, kappa_3))
        
        return phi, kappa

    compute_curvature = jax.jit(shape_wrapper(compute_curvature))

    def compute_T_grad(lattice_ids, T, phase):
        # TODO: duplicated code with VOF
        def get_T_grad(lattice_id, T, phase):
            T_grid = extract_3x3x3_grid(lattice_id, T) # (3, 3, 3)
            phase_grid = extract_3x3x3_grid(lattice_id, phase)
            c_id = 1
            T_grid = np.where(np.logical_or(phase_grid == GAS, phase_grid == WALL), T_grid[c_id, c_id, c_id], T_grid)
            T_grad_x = (T_grid[c_id + 1, c_id, c_id] - T_grid[c_id - 1, c_id, c_id])/(2.*h)
            T_grad_y = (T_grid[c_id, c_id + 1, c_id] - T_grid[c_id, c_id - 1, c_id])/(2.*h)
            T_grad_z = (T_grid[c_id, c_id, c_id + 1] - T_grid[c_id, c_id, c_id - 1])/(2.*h)
            return np.array([T_grad_x, T_grad_y, T_grad_z])
        get_T_grad_vmap = jax.jit(jax.vmap(get_T_grad, in_axes=(0, None, None)))
        T_grad = get_T_grad_vmap(lattice_ids, T, phase)
        return T_grad
 
    compute_T_grad = jax.jit(shape_wrapper(compute_T_grad))

    @jax.jit
    def compute_f_source_term(rho, vof, phi, kappa, T, T_grad):
        gravity_force = rho[:, :, :, None] * g[None, None, None, :]
        st_force = st_coeff * kappa[:, :, :, None] * phi
        normal = phi / np.linalg.norm(phi, axis=-1)[:, :, :, None]
        normal = np.where(np.isfinite(normal), normal, 0.)
        Marangoni_force = st_grad_coeff * (T_grad - np.sum(normal*T_grad, axis=-1)[:, :, :, None]*normal) * \
                          np.linalg.norm(phi, axis=-1)[:, :, :, None]*2.*vof[:, :, :, None]

        recoil_pressure = 0.54*p_atm*np.exp(latent_evap*(T - T_evap)/(gas_const*T*T_evap))[:, :, :, None] * phi
        
        source_term = gravity_force + st_force + Marangoni_force + recoil_pressure
        return source_term


    @jax.jit
    def compute_u(f_distribute, rho, T, f_source_term):
        rho = np.where(rho == 0., 1., rho)
        u = (np.sum(f_distribute[:, :, :, :, None] * vels.T[None, None, None, :, :], axis=-2) + dt*m*f_source_term) / \
            rho[:, :, :, None] 
        u = np.where((T < T_solidus)[:, :, :, None], 0., u)
        return u


    def compute_h_src(lattice_id, T, vof, phi, cell_centroids, crt_t):
        T_self = extract_self(lattice_id, T)
        vof_self = extract_self(lattice_id, vof)
        phi_self = extract_self(lattice_id, phi)
        centroid = cell_centroids[lattice_id]

        q_rad = SB_const*emissivity*(T0**4 - T_self**4)
        q_conv = h_conv*(T0 - T_self)
        q_loss = np.linalg.norm(phi_self) * (q_conv + q_rad) * 2.*vof_self

        x, y, z = centroid
        laser_x = 1./6.*domain_x + scanning_vel*crt_t
        laser_y = 1./2.*domain_y
        laser_z = 1./2.*domain_z

        # d = 1./4.*domain_z
        # q_laser = 2*laser_power*absorbed_fraction/(np.pi*beam_size**2)*np.exp(-2.*((x-laser_x)**2 + (y-laser_y)**2)/beam_size**2)
        # q_laser_body = q_laser/d * np.where(np.absolute(z - laser_z) < d, 1., 0.)  
        # heat_source = q_loss + q_laser_body

        q_laser = 2*laser_power*absorbed_fraction/(np.pi*beam_size**2)*np.exp(-2.*((x-laser_x)**2 + (y-laser_y)**2)/beam_size**2)

        # heat_source = np.linalg.norm(phi_self) * q_laser * 2.*vof_self + q_loss # TODO: 2.*vof_self?

        tmp = (-phi_self * np.array([0., 0., 1.]))[2]
        tmp = np.where(tmp > 0., tmp, 0.)
        heat_source = tmp * q_laser * 2.*vof_self + q_loss

        return heat_source

    compute_h_source_term = jax.jit(shape_wrapper(jax.vmap(compute_h_src, in_axes=(0, None, None, None, None, None))))



    def collide_f(lattice_id, f_distribute, rho, T, u, phase, f_source_term):
        """Returns f_distribute
        """
        f_distribute_self = extract_self(lattice_id, f_distribute) 
        rho_self = extract_self(lattice_id, rho)
        u_self = extract_self(lattice_id, u)
        phase_local = extract_local(lattice_id, phase)
        T_self = extract_self(lattice_id, T)
        f_source_term_self = extract_self(lattice_id, f_source_term)

        def gas_or_wall():
            return np.zeros_like(f_distribute_self)

        def nongas():
            f_equil = equilibrium_f_vmap(np.arange(Ns), rho_self, u_self)  
            forcing = f_forcing_vmap(np.arange(Ns), u_self, f_source_term_self)

            new_f_dist = np.where(T_self < T_solidus, weights*rho_self,
                1./tau_viscosity_nu*(f_equil - f_distribute_self) + f_distribute_self + forcing*dt)

            # new_f_dist = 1./tau_viscosity_nu*(f_equil - f_distribute_self) + f_distribute_self + forcing*dt

            return new_f_dist

        return jax.lax.cond(np.logical_or(phase_local[0] == GAS, phase_local[0] == WALL), gas_or_wall, nongas)

    collide_f_vmap = jax.jit(shape_wrapper(jax.vmap(collide_f, in_axes=(0, None, None, None, None, None, None))))


    def collide_h(lattice_id, h_distribute, enthalpy, T, rho, u, phase, h_source_term):
        """Returns h_distribute
        """
        h_distribute_self = extract_self(lattice_id, h_distribute) 
        enthalpy_self = extract_self(lattice_id, enthalpy)
        T_self = extract_self(lattice_id, T)
        rho_self = extract_self(lattice_id, rho)
        u_self = extract_self(lattice_id, u)
        phase_local = extract_local(lattice_id, phase)

        h_source_term_self = extract_self(lattice_id, h_source_term)

        def gas_or_wall():
            return np.zeros_like(h_distribute_self)

        def nongas():
            h_equil = equilibrium_h_vmap(np.arange(Ns), enthalpy_self, T_self, u_self)
            heat_source = h_forcing_vmap(h_source_term_self, rho_self)   

            tau_diffusivity = np.where(T_self < T_solidus, tau_diffusivity_s, tau_diffusivity_l)
            new_h_dist = 1./tau_diffusivity*(h_equil - h_distribute_self) + h_distribute_self + heat_source*dt

            # new_h_dist = 1./tau_diffusivity_s*(h_equil - h_distribute_self) + h_distribute_self  

            return new_h_dist

        return jax.lax.cond(np.logical_or(phase_local[0] == GAS, phase_local[0] == WALL), gas_or_wall, nongas)

    collide_h_vmap = jax.jit(shape_wrapper(jax.vmap(collide_h, in_axes=(0, None, None, None, None, None, None, None))))


    def update_f(lattice_id, f_distribute, rho, u, phase, mass, vof):
        """Returns f_distribute, mass
        """
        f_distribute_self = extract_self(lattice_id, f_distribute) 
        f_distribute_income = extract_income(lattice_id, f_distribute)
        rho_self = extract_self(lattice_id, rho)
        u_self = extract_self(lattice_id, u)
        phase_local = extract_local(lattice_id, phase)
        vof_local = extract_local(lattice_id, vof)

        def gas_or_wall():
            return np.zeros_like(f_distribute_self), 0.

        def nongas():
            def stream_wall(q):
                return f_distribute_self[rev[q]]

            def stream_gas(q):
                return equilibrium_f(rev[q], rho_g, u_self) + equilibrium_f(q, rho_g, u_self) - f_distribute_self[rev[q]]

            def stream_liquid_or_lg(q):
                return f_distribute_income[rev[q]]

            def compute_stream(q):  
                return jax.lax.cond(phase_local[rev[q]] == GAS, stream_gas, 
                    lambda q: jax.lax.cond(phase_local[rev[q]] == WALL, stream_wall, stream_liquid_or_lg, q), q)

            streamed_f_dist = jax.vmap(compute_stream)(np.arange(Ns)) # (Ns,)

            def liquid():
                return streamed_f_dist, rho_self

            def lg():
                def mass_change_liquid(q):
                    f_in = f_distribute_income[q] 
                    f_out = f_distribute_self[q]
                    return f_in - f_out

                def mass_change_lg(q):
                    f_in = f_distribute_income[q]
                    f_out = f_distribute_self[q]
                    vof_in = vof_local[q]
                    vof_out = vof_local[0]
                    return (f_in - f_out) * (vof_in + vof_out) / 2.

                def mass_change_gas_or_wall(q):
                    return 0.

                def mass_change(q):
                    return jax.lax.switch(phase_local[q], [mass_change_liquid, mass_change_lg, mass_change_gas_or_wall, mass_change_gas_or_wall], q)

                delta_m = np.sum(jax.vmap(mass_change)(np.arange(Ns)))
                mass_self = extract_self(lattice_id, mass)
                return streamed_f_dist, delta_m + mass_self

            return jax.lax.cond(phase_local[0] == LIQUID, liquid, lg)

        return jax.lax.cond(np.logical_or(phase_local[0] == GAS, phase_local[0] == WALL), gas_or_wall, nongas)

    update_f_vmap = jax.jit(shape_wrapper(jax.vmap(update_f, in_axes=(0, None, None, None, None, None, None))))



    def update_h(lattice_id, h_distribute, u, phase):
        """Returns h_distribute
        """
        h_distribute_self = extract_self(lattice_id, h_distribute) 
        h_distribute_income = extract_income(lattice_id, h_distribute)
        u_self = extract_self(lattice_id, u)
        phase_local = extract_local(lattice_id, phase)

        def gas_or_wall():
            return np.zeros_like(h_distribute_self)

        def nongas():
            def stream_wall(q):
                return equilibrium_h(q, T0*heat_capacity, T0, np.zeros(3))

            def stream_gas(q):
                return h_distribute_self[rev[q]]
 
            def stream_liquid_or_lg(q):
                return h_distribute_income[rev[q]]

            def compute_stream(q):  
                return jax.lax.cond(phase_local[rev[q]] == GAS, stream_gas, 
                    lambda q: jax.lax.cond(phase_local[rev[q]] == WALL, stream_wall, stream_liquid_or_lg, q), q)

            streamed_h_dist = jax.vmap(compute_stream)(np.arange(Ns)) # (Ns,)

            return streamed_h_dist

        return jax.lax.cond(np.logical_or(phase_local[0] == GAS, phase_local[0] == WALL), gas_or_wall, nongas)

    update_h_vmap = jax.jit(shape_wrapper(jax.vmap(update_h, in_axes=(0, None, None, None))))



    def reini_lg_to_liquid(lattice_id, f_distribute, phase, mass):
        """Returns phase
        """
        f_distribute_self = extract_self(lattice_id, f_distribute) # (Ns,)
        rho_self = np.sum(f_distribute_self) # (,)
        phase_self = extract_self(lattice_id, phase) # (,)
        mass_self = extract_self(lattice_id, mass) # (,)
        flag = np.logical_and(phase_self == LG, mass_self > (1+theta)*rho_self)
        return jax.lax.cond(flag, lambda: LIQUID, lambda: phase_self)
 
    reini_lg_to_liquid_vmap = jax.jit(shape_wrapper(jax.vmap(reini_lg_to_liquid, in_axes=(0, None, None, None))))

 
    def reini_gas_to_lg(lattice_id, f_distribute, h_distribute, rho, u, enthalpy, T, phase, mass):
        """Returns f_distribute, h_distribute, phase, mass
        """
        phase_local = extract_local(lattice_id, phase) # (Ns,)
        flag = np.logical_and(phase_local[0] == GAS, np.any(phase_local[1:] == LIQUID))
        def convert():
            rho_local = extract_local(lattice_id, rho) # (Ns,)
            u_local = extract_local(lattice_id, u) # (Ns, dim)
            enthalpy_local = extract_local(lattice_id, enthalpy) # (Ns,)
            T_local = extract_local(lattice_id, T) # (Ns,)
            nb_liquid_flag = np.logical_or(phase_local == LIQUID, phase_local == LG)
            rho_avg =  np.sum(nb_liquid_flag * rho_local) / np.sum(nb_liquid_flag)
            u_avg = np.sum(nb_liquid_flag[:, None] * u_local, axis=0) / np.sum(nb_liquid_flag)
            enthalpy_avg = np.sum(nb_liquid_flag * enthalpy_local) / np.sum(nb_liquid_flag)
            T_avg = np.sum(nb_liquid_flag * T_local) / np.sum(nb_liquid_flag)
            f_equil = equilibrium_f_vmap(np.arange(Ns), rho_avg, u_avg) # (Ns,

            h_equil = equilibrium_h_vmap(np.arange(Ns), enthalpy_avg, T_avg, u_avg)
            # h_equil = heat_capacity*T0*weights
            # h_equil = enthalpy_avg*weights

            return f_equil, h_equil, LG, 0.
        def nonconvert():
            f_distribute_self = extract_self(lattice_id, f_distribute)
            h_distribute_self = extract_self(lattice_id, h_distribute)
            mass_self = extract_self(lattice_id, mass)
            return f_distribute_self, h_distribute_self, phase_local[0], mass_self
        return jax.lax.cond(flag, convert, nonconvert)

    reini_gas_to_lg_vmap = jax.jit(shape_wrapper(jax.vmap(reini_gas_to_lg, in_axes=(0, None, None, None, None, None, None, None, None))))


    def reini_lg_to_gas(lattice_id, f_distribute, phase, mass):
        """Returns phase
        """
        f_distribute_self = extract_self(lattice_id, f_distribute) # (Ns,)
        rho_self = np.sum(f_distribute_self) # (,)
        phase_self = extract_self(lattice_id, phase) # (,)
        mass_self = extract_self(lattice_id, mass) # (,)
        flag = np.logical_and(phase_self == LG, mass_self < (0-theta)*rho_self)
        return jax.lax.cond(flag, lambda: GAS, lambda: phase_self)
 
    reini_lg_to_gas_vmap = jax.jit(shape_wrapper(jax.vmap(reini_lg_to_gas, in_axes=(0, None, None, None))))


    def reini_liquid_to_lg(lattice_id, f_distribute, phase, mass):
        """Returns phase, mass
        """
        f_distribute_self = extract_self(lattice_id, f_distribute) # (Ns,)
        rho_self = np.sum(f_distribute_self) # (,)
        phase_local = extract_local(lattice_id, phase) # (Ns,)
        mass_self = extract_self(lattice_id, mass)
        flag = np.logical_and(phase_local[0] == LIQUID, np.any(phase_local[1:] == GAS))
        def convert():
            return LG, rho_self
        def nonconvert():
            return phase_local[0], mass_self
        return jax.lax.cond(flag, convert, nonconvert)

    reini_liquid_to_lg_vmap = jax.jit(shape_wrapper(jax.vmap(reini_liquid_to_lg, in_axes=(0, None, None, None))))


    def adhoc_step(lattice_id, f_distribute, phase, mass):
        """Returns phase
        """
        f_distribute_self = extract_self(lattice_id, f_distribute) # (Ns,)
        rho_self = np.sum(f_distribute_self) # (,)
        phase_local = extract_local(lattice_id, phase) 
        mass_self = extract_self(lattice_id, mass)
        gas_nb_flag = np.all(np.logical_or(phase_local[1:] == WALL, np.logical_or(phase_local[1:] == GAS, phase_local[1:] == LG)))
        gas_flag = np.logical_and(phase_local[0] == LG, gas_nb_flag)
        liquid_nb_flag = np.all(np.logical_or(phase_local[1:] == WALL, np.logical_or(phase_local[1:] == LIQUID, phase_local[1:] == LG)))
        liquid_flag = np.logical_and(phase_local[0] == LG, liquid_nb_flag)
        return jax.lax.cond(gas_flag, lambda: GAS, lambda: jax.lax.cond(liquid_flag, lambda: LIQUID, lambda: phase_local[0]))

    adhoc_step_vmap = jax.jit(shape_wrapper(jax.vmap(adhoc_step, in_axes=(0, None, None, None))))


    def refresh_for_output(lattice_id, f_distribute, h_distribute, phase, mass):
        """Returns f_distribute, h_distribute, phase, mass
        """
        f_distribute_self = extract_self(lattice_id, f_distribute)
        h_distribute_self = extract_self(lattice_id, h_distribute)
        phase_self = extract_self(lattice_id, phase) # (Ns,)
        mass_self = extract_self(lattice_id, mass)
        def refresh():
            return np.zeros_like(f_distribute_self), np.zeros_like(h_distribute_self), phase_self, 0.
        def no_refresh():
            return f_distribute_self, h_distribute_self, phase_self, mass_self
        return jax.lax.cond(np.logical_or(phase_self == GAS, phase_self == WALL), refresh, no_refresh)

    refresh_for_output_vmap = jax.jit(shape_wrapper(jax.vmap(refresh_for_output, in_axes=(0, None, None, None, None))))


    def compute_total_mass(lattice_id, f_distribute, phase, mass):
        phase_self = extract_self(lattice_id, phase)
        def liquid():
            f_distribute_self = extract_self(lattice_id, f_distribute) # (Ns,)
            rho_self = np.sum(f_distribute_self) # (,)
            return rho_self
        def lg():
            mass_self = extract_self(lattice_id, mass)
            return mass_self
        def gas():
            return 0.
        return jax.lax.cond(phase_self == LIQUID, liquid, lambda: jax.lax.cond(phase_self == LG, lg, gas))

    compute_total_mass_vmap = jax.jit(shape_wrapper(jax.vmap(compute_total_mass, in_axes=(0, None, None, None))))


    def output_result(meshio_mesh, f_distribute, h_distribute, phase, mass, kappa, melted, step):
        rho = np.sum(f_distribute, axis=-1) # (Nx, Ny, Nz)
        rho = np.where(rho == 0., 1., rho)
        u = np.sum(f_distribute[:, :, :, :, None] * vels.T[None, None, None, :, :], axis=-2) / rho[:, :, :, None]
        u = np.where(np.isfinite(u), u, 0.)
        u = np.where((phase == LIQUID)[..., None], u, 0.)
        T = compute_T(compute_enthalpy(h_distribute))

        rho = rho * C_density
        u = u * C_length/C_time
        T = T * C_temperature
        max_x, max_y, max_z = to_id_xyz(np.argmax(T))
        print(f"max T = {np.max(T)} at ({max_x}, {max_y}, {max_z}) of ({Nx}, {Ny}, {Nz})")
        meshio_mesh.cell_data['phase'] = [onp.array(phase, dtype=onp.float32)]
        meshio_mesh.cell_data['mass'] = [onp.array(mass, dtype=onp.float32)]
        meshio_mesh.cell_data['rho'] = [onp.array(rho, dtype=onp.float32)]
        meshio_mesh.cell_data['kappa'] = [onp.array(kappa, dtype=onp.float32)]
        meshio_mesh.cell_data['vel'] = [onp.array(u.reshape(-1, 3) , dtype=onp.float32)]
        meshio_mesh.cell_data['T'] = [onp.array(T, dtype=onp.float32)]
        meshio_mesh.cell_data['melted'] = [onp.array(melted, dtype=onp.float32)]
        # meshio_mesh.cell_data['debug'] = [onp.array(dmass_output, dtype=onp.float32)]
        meshio_mesh.write(os.path.join(vtk_dir, f'sol_{step:04d}.vtu'))


    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    # 800x200x200 works
    # Nx, Ny, Nz = 300, 100, 100 
    # Nx, Ny, Nz = 150, 50, 50 
    Nx, Ny, Nz = 250, 50, 40 
    # Nx, Ny, Nz = 100, 100, 100 

    domain_x, domain_y, domain_z = Nx, Ny, Nz
    LIQUID, LG, GAS, WALL = 0, 1, 2, 3
    meshio_mesh = box_mesh(Nx, Ny, Nz, domain_x, domain_y, domain_z)
    points = meshio_mesh.points
    cells = meshio_mesh.cells_dict['hexahedron']
    cell_centroids = np.mean(points[cells], axis=1)

    vels = np.array([[0, 1, -1, 0,  0, 0,  0, 1, -1 , 1, -1, 1, -1, -1,  1, 0,  0,  0,  0],
                     [0, 0,  0, 1, -1, 0,  0, 1, -1, -1,  1, 0,  0,  0,  0, 1, -1,  1, -1],
                     [0, 0,  0, 0,  0, 1, -1, 0,  0,  0,  0, 1, -1,  1, -1, 1, -1, -1,  1]])
    rev = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
    weights = np.array([1./3., 1./18., 1./18., 1./18., 1./18., 1./18., 1./18., 1./36., 1./36., 1./36.,
                        1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36.])

    m = 0.5
    Ns = 19
    cs_sq = 1./3.
    theta = 1e-3

    h = 1.
    h_real = 6e-6 # [m]
    C_length = h_real/h

    dt = 1.
    dt_real = 2.*1e-7 # [s], 1e-7 for surface tension
    # dt_real = 0.5*1e-7 # [s], 1e-7 for surface tension

    C_time = dt_real/dt

    rho0 = 1.
    rho0_real = 8440. # [kg/m^3]
    C_density = rho0_real/rho0

    T0 = 1.
    T0_real = 300.
    C_temperature = T0_real/T0

    M0 = 1.
    M0_real = 1.
    C_molar = M0_real/M0

    C_mass = C_density*C_length**3
    C_force = C_mass*C_length/(C_time**2)
    C_energy = C_force*C_length
    C_pressure = C_force/C_length**2

    gravity_real = 9.8 # [m/s^2]
    viscosity_mu_real = 0.007 # [kg/(m*s)]
    # viscosity_mu_real = 0.001 # [kg/(m*s)] 
    st_coeff_real = 1.8 # [N/m]
    st_grad_coeff_real = -2e-5 # [N/(m*K)]

    p_atm_real = 101325 # [Pa]
    molar_mass_real = 6.14e-2 # [kg/mol]
    gas_const_real = 8.314 # [J/(K*mol)]

    laser_power_real = 100 # [W]
    beam_size_real = 25.*1e-6 # [m]
    absorbed_fraction_real = 0.4
    scanning_vel_real = 0.4 # [m/s]
    heat_capacity_volume_real = 5.2e6 # [J/(m^3*K)]
    heat_capacity_real = heat_capacity_volume_real/rho0_real # [J/(kg*K)]
    thermal_diffusivitity_l_real = 30./heat_capacity_volume_real # [m^2/s]  
    thermal_diffusivitity_s_real = 9.8/heat_capacity_volume_real # [m^2/s]
    SB_const_real = 5.67e-8 # [kg*s^-3*K^-4]
    emissivity_real = 0.3
    h_conv_real = 100 # [kg*s^-3*K^-1]
    latent_heat_real = 2.16e9/rho0_real # [J/kg]
    latent_evap_real = 379.e3 # [J/mol] 
    T_liquidus_real = 1623 # [K]
    T_solidus_real = 1563 # [K]
    T_evap_real = 3188 # [K]
    enthalpy_s_real = heat_capacity_real*T_solidus_real # [J/kg]
    enthalpy_l_real = heat_capacity_real*T_liquidus_real + latent_heat_real # [J/kg]

    gravity = gravity_real/(C_length/C_time**2)
    viscosity_mu = viscosity_mu_real/(C_mass/(C_length*C_time))
    st_coeff = st_coeff_real/(C_force/C_length)
    st_grad_coeff = st_grad_coeff_real/(C_force/(C_length*C_temperature))

    p_atm = p_atm_real/C_pressure
    molar_mass = molar_mass_real/(C_mass/C_molar)
    gas_const = gas_const_real/(C_energy/(C_temperature*C_molar))


    laser_power = laser_power_real/(C_energy/C_time)
    beam_size = beam_size_real/C_length
    absorbed_fraction = absorbed_fraction_real
    scanning_vel = scanning_vel_real/(C_length/C_time)
    heat_capacity = heat_capacity_real/(C_energy/(C_mass*C_temperature))
    thermal_diffusivitity_l = thermal_diffusivitity_l_real/(C_length**2/C_time)
    thermal_diffusivitity_s = thermal_diffusivitity_s_real/(C_length**2/C_time)
    SB_const = SB_const_real/(C_mass/C_time**3/C_temperature**4)
    emissivity = emissivity_real
    h_conv = h_conv_real/(C_mass/C_time**3/C_temperature)
    latent_heat = latent_heat_real/(C_energy/C_mass)
    latent_evap = latent_evap_real/(C_energy/C_molar)
    T_liquidus = T_liquidus_real/C_temperature
    T_solidus = T_solidus_real/C_temperature
    T_evap = T_evap_real/C_temperature
    enthalpy_s = enthalpy_s_real/(C_energy/C_mass)
    enthalpy_l = enthalpy_l_real/(C_energy/C_mass)

    viscosity_nu = viscosity_mu/rho0
    tau_viscosity_nu = viscosity_nu/(cs_sq*dt) + 0.5
    tau_diffusivity_l = thermal_diffusivitity_l/(cs_sq*dt) + 0.5
    tau_diffusivity_s = thermal_diffusivitity_s/(cs_sq*dt) + 0.5
    rho_g = rho0
    enthalpy_g = T0*heat_capacity
    g = np.array([0., 0., -gravity])

    # tau_viscosity_nu = 0.6
    # st_coeff = 0.05

    # assert tau_viscosity_nu < 1., f"Warning: tau_viscosity_nu = {tau_viscosity_nu} is out of range [0.5, 1] - may cause numerical instability"
    print(f"Relaxation parameter tau_viscosity_nu = {tau_viscosity_nu}, tau_diffusivity_s = {tau_diffusivity_s}, surface tensiont coeff = {st_coeff}")
    print(f"Lattice = ({Nx}, {Ny}, {Nz}), size = {h_real*1e6} micro m")
    print(f"TODO - Reynolds number")

    lattice_ids = np.arange(Nx*Ny*Nz)

    # phase = initialize_phase_surface_tension_vmap(lattice_ids, cell_centroids)
    # phase = initialize_phase_free_fall_vmap(lattice_ids, cell_centroids)
    phase = initialize_phase_powder_vmap(lattice_ids, cell_centroids, generate_powder())


    h_distribute = np.tile(weights, (Nx, Ny, Nz, 1)) * T0*heat_capacity
    f_distribute = np.tile(weights, (Nx, Ny, Nz, 1)) * rho0
    mass = np.sum(f_distribute, axis=-1)
    rho = compute_rho(f_distribute)
    u = np.zeros((Nx, Ny, Nz, 3))
    enthalpy = compute_enthalpy(h_distribute)
    T = compute_T(enthalpy)

    f_distribute, h_distribute, phase, mass = reini_gas_to_lg_vmap(lattice_ids, f_distribute, h_distribute, rho, u, enthalpy, T, phase, mass)
    mass = np.where(phase == LG, 0.5*np.sum(f_distribute, axis=-1), mass)

    total_mass = np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass))

    melted = np.zeros_like(mass)
    output_result(meshio_mesh, f_distribute, h_distribute, phase, mass, np.zeros_like(mass), melted, 0)

    start_time = time.time()
    for i in range(15000):
    # for i in range(2000):
        # print(f"Step {i}")
        # print(f"Initial mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass))}")
        crt_t = (i + 1)*dt

        rho = compute_rho(f_distribute)
        enthalpy = compute_enthalpy(h_distribute)
        T = compute_T(enthalpy)
        vof = compute_vof(rho, phase, mass)
        phi, kappa = compute_curvature(lattice_ids, vof)
        T_grad = compute_T_grad(lattice_ids, T, phase)

        f_source_term = compute_f_source_term(rho, vof, phi, kappa, T, T_grad)
        h_source_term = compute_h_source_term(lattice_ids, T, vof, phi, cell_centroids, crt_t)
        u = compute_u(f_distribute, rho, T, f_source_term)

        f_distribute = collide_f_vmap(lattice_ids, f_distribute, rho, T, u, phase, f_source_term)
        h_distribute = collide_h_vmap(lattice_ids, h_distribute, enthalpy, T, rho, u, phase, h_source_term)

        f_distribute, mass = update_f_vmap(lattice_ids, f_distribute, rho, u, phase, mass, vof)
        h_distribute = update_h_vmap(lattice_ids, h_distribute, u, phase)

        # print(f"max kappa = {np.max(kappa)}, min kappa = {np.min(kappa)}")
        # print(f"After update, mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass))}")
        
        phase = reini_lg_to_liquid_vmap(lattice_ids, f_distribute, phase, mass)
        # print(f"After lg_to_liquid, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass))}")

        # rho, u = compute_rho_u(f_distribute)
        # rho = compute_rho(f_distribute)
        # u = compute_u(f_distribute, rho, f_source_term)
        f_distribute, h_distribute, phase, mass = reini_gas_to_lg_vmap(lattice_ids, f_distribute, h_distribute, rho, u, enthalpy, T, phase, mass)
        # print(f"After gas_to_lg, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass))}")

        phase = reini_lg_to_gas_vmap(lattice_ids, f_distribute, phase, mass)
        # print(f"After lg_to_gas, mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass))}")

        phase, mass = reini_liquid_to_lg_vmap(lattice_ids, f_distribute, phase, mass)
        # print(f"After liquid_to_lg, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass))}")

        phase = adhoc_step_vmap(lattice_ids, f_distribute, phase, mass)

        calculated_mass = np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass))
        mass = np.where(phase == LG, mass + (total_mass - calculated_mass)/np.sum(phase == LG), mass)

        f_distribute, h_distribute, phase, mass = refresh_for_output_vmap(lattice_ids, f_distribute, h_distribute, phase, mass)
        # print(f"After refresh, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass))}")
        # print(f"max f_distribute = {np.max(f_distribute)}, max mass = {np.max(mass)}")
        # print(f"min f_distribute = {np.min(f_distribute)}, min mass = {np.min(mass)}")

        melted = np.where(T > T_solidus, 1., melted)
        inverval = 250
        if (i + 1) % inverval == 0:
            print(f"Step {i + 1}")
            output_result(meshio_mesh, f_distribute, h_distribute, phase, mass, kappa, melted, (i + 1) // inverval)

    end_time = time.time()
    print(f"Total wall time = {end_time - start_time}")


if __name__== "__main__":
    simulation()
    # make_video(data_dir)
