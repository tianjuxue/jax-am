import numpy as onp
import jax.numpy as np
import jax
import meshio
import os
import glob

from jax_am.common import make_video, box_mesh

jax.config.update("jax_enable_x64", True)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

case_name = 'surface_tension'
data_dir = os.path.join(os.path.dirname(__file__), 'data')
vtk_dir = os.path.join(data_dir, f'vtk/{case_name}')
os.makedirs(vtk_dir, exist_ok=True)


def simulation():
    def to_id(idx, idy, idz):
        return idx * Ny * Nz + idy * Nz + idz

    def to_id_xyz(lattice_id):
        id_z = lattice_id % Nz
        lattice_id = lattice_id // Nz
        id_y = lattice_id % Ny
        id_x = lattice_id // Ny    
        return id_x, id_y, id_z 

    def extract_7x7x7_grid(lattice_id, values):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        grid_index = np.ix_(np.array([(id_x - 3) % Nx, (id_x - 2) % Nx, (id_x - 1) % Nx, id_x, (id_x + 1) % Nx, (id_x + 2) % Nx, (id_x + 3) % Nx]), 
                            np.array([(id_y - 3) % Ny, (id_y - 2) % Ny, (id_y - 1) % Ny, id_y, (id_y + 1) % Ny, (id_y + 2) % Ny, (id_y + 3) % Ny]),
                            np.array([(id_z - 3) % Nz, (id_z - 2) % Nz, (id_z - 1) % Nz, id_z, (id_z + 1) % Nz, (id_z + 2) % Nz, (id_z + 3) % Nz]))
        grid_values = values[grid_index]
        return grid_values   

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
        grid_values = extract_3x3x3_grid(lattice_id, values)
        # TODO: ugly
        local_values = grid_values[vels[0] + 1, vels[1] + 1, vels[2] + 1]
        return local_values

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
        flag = np.logical_and(np.logical_and(flag_x, flag_y), flag_z)
        tmp = np.where(flag, LIQUID, GAS)
        wall_x = np.logical_or(id_x == 0, id_x == Nx - 1)
        wall_y = np.logical_or(id_y == 0, id_y == Ny - 1)
        wall_z = np.logical_or(id_z == 0, id_z == Nz - 1)
        wall = np.logical_or(wall_x, np.logical_or(wall_y, wall_z))
        return np.where(wall, WALL, tmp)

    initialize_phase_surface_tension_vmap = shape_wrapper(jax.vmap(initialize_phase_surface_tension, in_axes=(0, None)))



    def initialize_phase_free_fall(lattice_id, cell_centroids):
        id_x, id_y, id_z = to_id_xyz(lattice_id)
        flag_x = cell_centroids[lattice_id, 0] < 0.8 * domain_x
        flag_y = cell_centroids[lattice_id, 1] < 0.8 * domain_y
        flag_z = cell_centroids[lattice_id, 2] < 0.8 * domain_z
        flag = np.logical_and(np.logical_and(flag_x, flag_y), flag_z)
        tmp = np.where(flag, LIQUID, GAS)
        wall_x = np.logical_or(id_x == 0, id_x == Nx - 1)
        wall_y = np.logical_or(id_y == 0, id_y == Ny - 1)
        wall_z = np.logical_or(id_z == 0, id_z == Nz - 1)
        wall = np.logical_or(wall_x, np.logical_or(wall_y, wall_z))
        return np.where(wall, WALL, tmp)

    initialize_phase_free_fall_vmap = shape_wrapper(jax.vmap(initialize_phase_free_fall, in_axes=(0, None)))





    def equilibrium(q, rho, u):
        vel_dot_u = np.sum(vels[:, q] * u)
        u_sq = np.sum(u * u)
        return weights[q] * rho * (1. + vel_dot_u/cs_sq + vel_dot_u**2./(2.*cs_sq**2.) - u_sq/(2.*cs_sq))

    equilibrium_vmap = jax.vmap(equilibrium, in_axes=(0, None, None))

    def forcing(q, u, force):
        ei = vels[:, q]
        return (1. - 1./(2.*tau)) * weights[q] * np.sum(((ei - u)/cs_sq + np.sum(ei * u)/cs_sq**2 * ei) * force)

    forcing_vmap = jax.vmap(forcing, in_axes=(0, None, None))

    def update(lattice_id, f_distribute, phase, mass):
        """Returns f_distribute, mass, kappa
        """
        f_distribute_local = extract_local(lattice_id, f_distribute) # (Ns, Ns)
        rho_local = np.sum(f_distribute_local, axis=-1) # (Ns,)
        u_local = (np.sum(f_distribute_local[:, :, None] * vels.T[None, :, :], axis=1) + m*g[None, :]*rho_local[:, None]) / rho_local[:, None] # (Ns, dim)
        phase_local = extract_local(lattice_id, phase) # (Ns,)
        mass_local = extract_local(lattice_id, mass)
        vof_local = np.where(phase_local == LIQUID, rho_local, mass_local)
        vof_local = np.where(phase_local == GAS, 0., vof_local)

        def gas_or_wall():
            return np.zeros_like(f_distribute_local[0]), 0., 0.

        def nongas():
            def stream_wall(q):
                return f_distribute_local[0, rev[q]]

            def stream_gas(q):
                return equilibrium(rev[q], rho_g, u_local[0]) + equilibrium(q, rho_g, u_local[0]) - f_distribute_local[0, rev[q]]

            def stream_liquid_or_lg(q):
                return f_distribute_local[rev[q], q]

            def compute_stream(q):  
                return jax.lax.cond(phase_local[rev[q]] == GAS, stream_gas, 
                    lambda q: jax.lax.cond(phase_local[rev[q]] == WALL, stream_wall, stream_liquid_or_lg, q), q)

            streamed_f_dist = jax.vmap(compute_stream)(np.arange(Ns)) # (Ns,)
            streamed_rho = np.sum(streamed_f_dist) # (,)
            streamed_u = (np.sum(streamed_f_dist[:, None] * vels.T[:, :], axis=0) + m*g*streamed_rho)/ streamed_rho # (dim,)


            def get_vof_grid(extract_grid_fn):
                f_distribute_grid = extract_grid_fn(lattice_id, f_distribute) 
                rho_grid = np.sum(f_distribute_grid, axis=-1) 
                phase_grid = extract_grid_fn(lattice_id, phase) 
                mass_grid = extract_grid_fn(lattice_id, mass) 
                vof_grid = np.where(phase_grid == LIQUID, rho_grid, mass_grid)
                vof_grid = np.where(phase_grid == GAS, 0., vof_grid)
                return vof_grid
  
            def curvature(hgt_func):
                c_id = 1
                Hx = (hgt_func[c_id + 1, c_id] - hgt_func[c_id - 1, c_id])/(2.*h)
                Hy = (hgt_func[c_id, c_id + 1] - hgt_func[c_id, c_id - 1])/(2.*h)
                Hxx = (hgt_func[c_id + 1, c_id] - 2.*hgt_func[c_id, c_id] + hgt_func[c_id - 1, c_id])/h**2
                Hyy = (hgt_func[c_id, c_id + 1] - 2.*hgt_func[c_id, c_id] + hgt_func[c_id, c_id - 1])/h**2
                Hxy = (hgt_func[c_id + 1, c_id + 1] - hgt_func[c_id - 1, c_id + 1] - hgt_func[c_id + 1, c_id - 1] + hgt_func[c_id - 1, c_id - 1])/(4*h)
                kappa = -(Hxx + Hyy + Hxx*Hy**2. + Hyy*Hx**2. - 2.*Hxy*Hx*Hy) / (1 + Hx**2. + Hy**2.)**(3./2.)
                kappa = np.where(np.isfinite(kappa), kappa, 0.)
                return kappa

            def x_big():
                vof_grid = get_vof_grid(extract_7x3x3_grid)       
                hgt_func = np.sum(vof_grid, axis=0)
                kappa = curvature(hgt_func)
                return kappa

            def y_big():
                vof_grid = get_vof_grid(extract_3x7x3_grid)       
                hgt_func = np.sum(vof_grid, axis=1)
                kappa = curvature(hgt_func)
                return kappa

            def z_big():
                vof_grid = get_vof_grid(extract_3x3x7_grid)       
                hgt_func = np.sum(vof_grid, axis=2)
                kappa = curvature(hgt_func)
                return kappa

            vof_grid = get_vof_grid(extract_3x3x3_grid)
            c_id = 1
            phi_x = (vof_grid[c_id + 1, c_id, c_id] - vof_grid[c_id - 1, c_id, c_id])/(2.*h)
            phi_y = (vof_grid[c_id, c_id + 1, c_id] - vof_grid[c_id, c_id - 1, c_id])/(2.*h)
            phi_z = (vof_grid[c_id, c_id, c_id + 1] - vof_grid[c_id, c_id, c_id - 1])/(2.*h)
            kappa = jax.lax.cond(np.logical_and(np.absolute(phi_x) >= np.absolute(phi_y), np.absolute(phi_x) >= np.absolute(phi_z)), x_big, 
                lambda: jax.lax.cond(np.logical_and(np.absolute(phi_y) >= np.absolute(phi_x), np.absolute(phi_y) >= np.absolute(phi_z)), y_big, z_big))

            # kappa_limit = 3.
            # kappa = np.where(kappa > kappa_limit, kappa_limit, kappa)
            # kappa = np.where(kappa < -kappa_limit, -kappa_limit, kappa)

            phi_vec = np.array([phi_x, phi_y, phi_z])
            surface_force = forcing_vmap(np.arange(Ns), streamed_u, st_coeff*kappa*phi_vec)

            f_equil = equilibrium_vmap(np.arange(Ns), streamed_rho, streamed_u) # (Ns,)
            body_force = forcing_vmap(np.arange(Ns), streamed_u, streamed_rho*g) # (Ns,)

            def liquid():
                new_f_dist = 1./tau*(f_equil - streamed_f_dist) + streamed_f_dist + body_force + surface_force
                return new_f_dist, rho_local[0], kappa

            def lg():
                def mass_change_liquid(q):
                    f_in = f_distribute_local[q, rev[q]]
                    f_out = f_distribute_local[0, q]
                    return f_in - f_out

                def mass_change_lg(q):
                    f_in = f_distribute_local[q, rev[q]]
                    f_out = f_distribute_local[0, q]
                    vof_in = vof_local[q]
                    vof_out = vof_local[0]
                    return (f_in - f_out) * (vof_in + vof_out) / 2.

                def mass_change_gas_or_wall(q):
                    return 0.

                def mass_change(q):
                    return jax.lax.switch(phase_local[q], [mass_change_liquid, mass_change_lg, mass_change_gas_or_wall, mass_change_gas_or_wall], q)

                delta_m = np.sum(jax.vmap(mass_change)(np.arange(Ns)))
                new_f_dist = 1./tau*(f_equil - streamed_f_dist) + streamed_f_dist + body_force + surface_force
                mass_self = extract_self(lattice_id, mass)
                return new_f_dist, delta_m + mass_self, kappa

            return jax.lax.cond(phase_local[0] == LIQUID, liquid, lg)

        return jax.lax.cond(np.logical_or(phase_local[0] == GAS, phase_local[0] == WALL), gas_or_wall, nongas)

    update_vmap = jax.jit(shape_wrapper(jax.vmap(update, in_axes=(0, None, None, None))))


    def reini_lg_to_liquid(lattice_id, f_distribute, phase, mass):
        """Returns phase, dmass
        """
        f_distribute_self = extract_self(lattice_id, f_distribute) # (Ns,)
        rho_self = np.sum(f_distribute_self) # (,)
        phase_self = extract_self(lattice_id, phase) # (,)
        mass_self = extract_self(lattice_id, mass) # (,)
        flag = np.logical_and(phase_self == LG, mass_self > (1+theta)*rho_self)
        return jax.lax.cond(flag, lambda: (LIQUID, mass_self - rho_self), lambda: (phase_self, 0.))
 
    reini_lg_to_liquid_vmap = jax.jit(shape_wrapper(jax.vmap(reini_lg_to_liquid, in_axes=(0, None, None, None))))


    def reini_gas_to_lg(lattice_id, f_distribute, phase, mass):
        """Returns f_distribute, phase, mass
        """
        f_distribute_local = extract_local(lattice_id, f_distribute) # (Ns, Ns)
        phase_local = extract_local(lattice_id, phase) # (Ns,)
        flag = np.logical_and(phase_local[0] == GAS, np.any(phase_local[1:] == LIQUID))
        def convert():
            rho_local = np.sum(f_distribute_local, axis=-1) # (Ns,)
            rho_local = np.where(rho_local == 0., 1., rho_local)
            # TODO
            u_local = (np.sum(f_distribute_local[:, :, None] * vels.T[None, :, :], axis=1) + m*g[None, :]*rho_local[:, None]) / rho_local[:, None] # (Ns, dim) 
            nb_liquid_flag = (phase_local == LIQUID)
            rho_avg =  np.sum(nb_liquid_flag * rho_local) / np.sum(nb_liquid_flag)
            u_avg = np.sum(nb_liquid_flag[:, None] * u_local, axis=0) / np.sum(nb_liquid_flag)
            f_equil = equilibrium_vmap(np.arange(Ns), rho_avg, u_avg) # (Ns,)
            return f_equil, LG, 0.
        def nonconvert():
            mass_self = extract_self(lattice_id, mass)
            return f_distribute_local[0], phase_local[0], mass_self
        return jax.lax.cond(flag, convert, nonconvert)

    reini_gas_to_lg_vmap = jax.jit(shape_wrapper(jax.vmap(reini_gas_to_lg, in_axes=(0, None, None, None))))


    def reini_lg_to_gas(lattice_id, f_distribute, phase, mass, dmass):
        """Returns phase, dmass
        """
        f_distribute_self = extract_self(lattice_id, f_distribute) # (Ns,)
        rho_self = np.sum(f_distribute_self) # (,)
        phase_self = extract_self(lattice_id, phase) # (,)
        mass_self = extract_self(lattice_id, mass) # (,)
        dmass_self = extract_self(lattice_id, dmass)
        flag = np.logical_and(phase_self == LG, mass_self < (0-theta)*rho_self)
        return jax.lax.cond(flag, lambda: (GAS, mass_self), lambda: (phase_self, dmass_self))
 
    reini_lg_to_gas_vmap = jax.jit(shape_wrapper(jax.vmap(reini_lg_to_gas, in_axes=(0, None, None, None, None))))


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


    def reallocate_mass_step1(lattice_id, phase, dmass):
        """Returns dmass_avg
        """
        phase_local = extract_local(lattice_id, phase) # (Ns,)
        dmass_self = extract_self(lattice_id, dmass)
        def alct():
            return dmass_self / np.sum(phase_local == LG)
        def no_alct():
            return 0.
        return jax.lax.cond(dmass_self != 0., alct, no_alct)

    reallocate_mass_step1_vmap = jax.jit(shape_wrapper(jax.vmap(reallocate_mass_step1, in_axes=(0, None, None))))


    def reallocate_mass_step2(lattice_id, phase, mass, dmass_avg):
        """Returns mass, dmass
        """
        phase_self = extract_self(lattice_id, phase) # (Ns,)
        mass_self = extract_self(lattice_id, mass)
        dmass_avg_local = extract_local(lattice_id, dmass_avg)
        def add():
            return mass_self + np.sum(dmass_avg_local)
        def no_add():
            return mass_self
        return jax.lax.cond(phase_self == LG, add, no_add), 0.

    reallocate_mass_step2_vmap = jax.jit(shape_wrapper(jax.vmap(reallocate_mass_step2, in_axes=(0, None, None, None))))


    def adhoc_step1(lattice_id, f_distribute, phase, mass):
        """Returns phase, adhoc_mass
        """
        f_distribute_self = extract_self(lattice_id, f_distribute) # (Ns,)
        rho_self = np.sum(f_distribute_self) # (,)
        phase_local = extract_local(lattice_id, phase) 
        mass_self = extract_self(lattice_id, mass)
        gas_nb_flag = np.all(np.logical_or(phase_local[1:] == WALL, np.logical_or(phase_local[1:] == GAS, phase_local[1:] == LG)))
        gas_flag = np.logical_and(phase_local[0] == LG, gas_nb_flag)
        liquid_nb_flag = np.all(np.logical_or(phase_local[1:] == WALL, np.logical_or(phase_local[1:] == LIQUID, phase_local[1:] == LG)))
        liquid_flag = np.logical_and(phase_local[0] == LG, liquid_nb_flag)
        return jax.lax.cond(gas_flag, lambda:(GAS, mass_self), lambda: jax.lax.cond(liquid_flag, lambda:(LIQUID, mass_self - rho_self), lambda:(phase_local[0], 0.)))

    adhoc_step1_vmap = jax.jit(shape_wrapper(jax.vmap(adhoc_step1, in_axes=(0, None, None, None))))


    def adhoc_step2(lattice_id, phase, mass, ad_mass):
        """Returns mass
        """
        phase_self = extract_self(lattice_id, phase) 
        mass_self = extract_self(lattice_id, mass)
        return np.where(phase_self == LG, mass_self + ad_mass, mass_self)

    adhoc_step2_vmap = jax.jit(shape_wrapper(jax.vmap(adhoc_step2, in_axes=(0, None, None, None))))


    def refresh_for_output(lattice_id, f_distribute, phase, mass):
        """Returns f_distribute, phase, mass
        """
        f_distribute_self = extract_self(lattice_id, f_distribute)
        phase_self = extract_self(lattice_id, phase) # (Ns,)
        mass_self = extract_self(lattice_id, mass)
        def refresh():
            return np.zeros_like(f_distribute_self), phase_self, 0.
        def no_refresh():
            return f_distribute_self, phase_self, mass_self
        return jax.lax.cond(np.logical_or(phase_self == GAS, phase_self == WALL), refresh, no_refresh)

    refresh_for_output_vmap = jax.jit(shape_wrapper(jax.vmap(refresh_for_output, in_axes=(0, None, None, None))))


    def compute_total_mass(lattice_id, f_distribute, phase, mass, dmass):
        phase_self = extract_self(lattice_id, phase)
        dmass_self = extract_self(lattice_id, dmass)
        def liquid():
            f_distribute_self = extract_self(lattice_id, f_distribute) # (Ns,)
            rho_self = np.sum(f_distribute_self) # (,)
            return rho_self
        def lg():
            mass_self = extract_self(lattice_id, mass)
            return mass_self
        def gas():
            return 0.
        return jax.lax.cond(phase_self == LIQUID, liquid, lambda: jax.lax.cond(phase_self == LG, lg, gas)) + dmass_self

    compute_total_mass_vmap = jax.jit(shape_wrapper(jax.vmap(compute_total_mass, in_axes=(0, None, None, None, None))))


    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    Nx, Ny, Nz = 100, 100, 100 
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
    h = 1.
    dt = 1.
    m = 0.5
    Ns = 19
    rho_g = 1.
    rho0 = 1.
    cs_sq = 1./3.
    tau = 0.75
    theta = 1e-3
    g = np.array([0., 0., -0.0005])
    st_coeff = 0.05
    # st_coeff = 0.
    # g = np.array([0., 0., 0.])
    lattice_ids = np.arange(Nx*Ny*Nz)

    phase = initialize_phase_surface_tension_vmap(lattice_ids, cell_centroids)
    # phase = initialize_phase_free_fall_vmap(lattice_ids, cell_centroids)


    # f_distribute = np.ones((Nx, Ny, Nz, Ns)) * rho0 / Ns
    f_distribute = np.tile(weights, (Nx, Ny, Nz, 1)) * rho0
    mass = np.sum(f_distribute, axis=-1)
    
    f_distribute, phase, mass = reini_gas_to_lg_vmap(lattice_ids, f_distribute, phase, mass)
    mass = np.where(phase == LG, 0.5*np.sum(f_distribute, axis=-1), mass)
    dmass = np.zeros_like(mass)

    print(f"max f_distribute = {np.max(f_distribute)}, max mass = {np.max(mass)}")

    total_mass = np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))

    for i in range(5001):
        print(f"Step {i}")
        print(f"Initial mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))}")
        f_distribute, mass, kappa = update_vmap(lattice_ids, f_distribute, phase, mass)
        print(f"After update, mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))}")
        print(f"max kappa = {np.max(kappa)}, min kappa = {np.min(kappa)}")

        phase, dmass = reini_lg_to_liquid_vmap(lattice_ids, f_distribute, phase, mass)
        print(f"After lg_to_liquid, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))}")

        f_distribute, phase, mass = reini_gas_to_lg_vmap(lattice_ids, f_distribute, phase, mass)
        print(f"After gas_to_lg, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))}")

        phase, dmass = reini_lg_to_gas_vmap(lattice_ids, f_distribute, phase, mass, dmass)
        print(f"After lg_to_gas, mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))}")

        phase, mass = reini_liquid_to_lg_vmap(lattice_ids, f_distribute, phase, mass)
        print(f"After liquid_to_lg, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))}")

        # dmass_output = np.array(dmass)
        # dmass_output = np.where(dmass_output > 0., 1., dmass_output)
        # dmass_output = np.where(dmass_output < 0., -1., dmass_output)

        # mass = np.where(phase == LG, mass + np.sum(dmass)/np.sum(phase == LG), mass)
        dmass = np.zeros_like(dmass)

        # dmass_avg = reallocate_mass_step1_vmap(lattice_ids, phase, dmass)
        # mass, dmass = reallocate_mass_step2_vmap(lattice_ids, phase, mass, dmass_avg)
        # print(f"After reallocate_mass, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))}")

        phase, adhoc_mass = adhoc_step1_vmap(lattice_ids, f_distribute, phase, mass)
        # mass = np.where(phase == LG, mass + np.sum(adhoc_mass)/np.sum(phase == LG), mass)
        # mass = adhoc_step2_vmap(lattice_ids, phase, mass, np.sum(adhoc_mass)/np.sum(phase == LG))
        # print(f"After adhoc, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))}")

        calculated_mass = np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, np.zeros_like(mass)))
        mass = np.where(phase == LG, mass + (total_mass - calculated_mass)/np.sum(phase == LG), mass)

        f_distribute, phase, mass = refresh_for_output_vmap(lattice_ids, f_distribute, phase, mass)
        print(f"After refresh, total mass = {np.sum(compute_total_mass_vmap(lattice_ids, f_distribute, phase, mass, dmass))}")
        # print(f"max f_distribute = {np.max(f_distribute)}, max mass = {np.max(mass)}")
        # print(f"min f_distribute = {np.min(f_distribute)}, min mass = {np.min(mass)}")

        if not np.all(np.isfinite(mass)):
            print(f"mass nan")
            break
        
        if not np.all(np.isfinite(f_distribute)):
            print(f"f_distribute nan")
            break

        if i % 50 == 0:
            rho = np.sum(f_distribute, axis=-1) # (Nx, Ny, Nz)
            u = np.sum(f_distribute[:, :, :, :, None] * vels.T[None, None, None, :, :], axis=-2) / rho[:, :, :, None]
            u = np.where(np.isfinite(u), u, 0.)
            meshio_mesh.cell_data['phase'] = [onp.array(phase, dtype=onp.float32)]
            meshio_mesh.cell_data['mass'] = [onp.array(mass, dtype=onp.float32)]
            meshio_mesh.cell_data['rho'] = [onp.array(rho, dtype=onp.float32)]
            meshio_mesh.cell_data['kappa'] = [onp.array(kappa, dtype=onp.float32)]
            meshio_mesh.cell_data['vel'] = [onp.array(u.reshape(-1, 3) , dtype=onp.float32)]
            # meshio_mesh.cell_data['debug'] = [onp.array(dmass_output, dtype=onp.float32)]
            meshio_mesh.write(os.path.join(vtk_dir, f'sol_{i:04d}.vtu'))


if __name__== "__main__":
    # simulation()
    make_video(data_dir)

