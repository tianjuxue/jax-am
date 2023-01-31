import jax
import jax.numpy as np
import numpy as onp
import scipy
from jax.experimental.sparse import BCOO
from functools import partial
import os
import time
import glob
jax.config.update("jax_enable_x64", True)


 # calculate ghost_cell values
def ghost_cell(BC, value, dx=None, neighbor=None):
    if BC == 'Dirchlet' or BC == 0:
        return 2 * value - neighbor
    if BC == 'Neumann' or BC == 1:
        return neighbor + dx * value
    if BC == 'Marangoni_Z' or BC == 2:
        return np.concatenate(
            ((neighbor[:, :, :, [0]] + dx * value[:, :, :, [0]]),
             (neighbor[:, :, :, [1]] + dx * value[:, :, :, [1]]),
             (2 * value[:, :, :, [2]] - neighbor[:, :, :, [2]])),
            axis=-1)

def get_GC_values(f, BCs, dX):
    return [
        ghost_cell(BCs[0][0], BCs[1][0], -dX[0], f[[0], :, :]),
        ghost_cell(BCs[0][1], BCs[1][1], dX[0], f[[-1], :, :]),
        ghost_cell(BCs[0][2], BCs[1][2], -dX[1], f[:, [0], :]),
        ghost_cell(BCs[0][3], BCs[1][3], dX[1], f[:, [-1], :]),
        ghost_cell(BCs[0][4], BCs[1][4], -dX[2], f[:, :, [0]]),
        ghost_cell(BCs[0][5], BCs[1][5], dX[2], f[:, :, [-1]])
    ]

def laplace(f,f_gc,miu,dX,BCs=None):
    if BCs == None:
        BCs = [[1, 1, 1, 1, 1, 1], [0., 0., 0., 0., 0., 0.]]
        f_gc = get_GC_values(f, BCs, dX)
    if np.isscalar(miu):
        dif = ((np.diff(f,axis=0,append=f_gc[1]) - np.diff(f,axis=0,prepend=f_gc[0]))/dX[0]*dX[1]*dX[2] +
               (np.diff(f,axis=1,append=f_gc[3]) - np.diff(f,axis=1,prepend=f_gc[2]))/dX[1]*dX[2]*dX[0] +
               (np.diff(f,axis=2,append=f_gc[5]) - np.diff(f,axis=2,prepend=f_gc[4]))/dX[2]*dX[0]*dX[1])*miu
    else:
        dif = ((np.diff(f,axis=0,append=f_gc[1])*miu[0][1:,:,:] 
                    - np.diff(f,axis=0,prepend=f_gc[0])*miu[0][:-1,:,:])/dX[0]*dX[1]*dX[2] +
               (np.diff(f,axis=1,append=f_gc[3])*miu[1][:,1:,:] 
                    - np.diff(f,axis=1,prepend=f_gc[2])*miu[1][:,:-1,:])/dX[1]*dX[2]*dX[0] +
               (np.diff(f,axis=2,append=f_gc[5])*miu[2][:,:,1:] 
                    - np.diff(f,axis=2,prepend=f_gc[4])*miu[2][:,:,:-1])/dX[2]*dX[0]*dX[1])
    return dif

def div(f, vel_f, dX, theta=0.125, BCs=None):
    if BCs == None:
        BCs = [[1, 1, 1, 1, 1, 1], [0., 0., 0., 0., 0., 0.]]
    f_gc = get_GC_values(f, BCs, dX)
    u_f, v_f, w_f = vel_f
    fx = np.concatenate((f_gc[0], f_gc[0], f, f_gc[1], f_gc[1]), axis=0)
    fy = np.concatenate((f_gc[2], f_gc[2], f, f_gc[3], f_gc[3]), axis=1)
    fz = np.concatenate((f_gc[4], f_gc[4], f, f_gc[5], f_gc[5]), axis=2)

    f_face_x = (theta * (fx[1:-2, :, :] / 2 + fx[2:-1, :, :] / 2) +
                (1 - theta) * (3 * fx[1:-2, :, :] / 2 - fx[:-3, :, :] / 2) *
                (u_f >= 0) + (1 - theta) *
                (3 * fx[2:-1, :, :] / 2 - fx[3:, :, :] / 2) * (u_f < 0))
    f_face_y = (theta * (fy[:, 1:-2, :] / 2 + fy[:, 2:-1, :] / 2) +
                (1 - theta) * (3 * fy[:, 1:-2, :] / 2 - fy[:, :-3, :] / 2) *
                (v_f >= 0) + (1 - theta) *
                (3 * fy[:, 2:-1, :] / 2 - fy[:, 3:, :] / 2) * (v_f < 0))
    f_face_z = (theta * (fz[:, :, 1:-2] / 2 + fz[:, :, 2:-1] / 2) +
                (1 - theta) * (3 * fz[:, :, 1:-2] / 2 - fz[:, :, :-3] / 2) *
                (w_f >= 0) + (1 - theta) *
                (3 * fz[:, :, 2:-1] / 2 - fz[:, :, 3:] / 2) * (w_f < 0))
    flux = ((f_face_x[1:, :, :] * u_f[1:, :, :] -
             f_face_x[:-1, :, :] * u_f[:-1, :, :]) / dX[0] +
            (f_face_y[:, 1:, :] * v_f[:, 1:, :] -
             f_face_y[:, :-1, :] * v_f[:, :-1, :]) / dX[1] +
            (f_face_z[:, :, 1:] * w_f[:, :, 1:] -
             f_face_z[:, :, :-1] * w_f[:, :, :-1]) / dX[2])
    return flux

def gradient(f, dX, BCs=None):
    if BCs == None:
        BCs = [[1, 1, 1, 1, 1, 1], [0., 0., 0., 0., 0., 0.]]
    f_gc = get_GC_values(f, [[1, 1, 1, 1, 1, 1], [0., 0., 0., 0., 0., 0.]], dX)
    return np.concatenate(((np.diff(f, axis=0, append=f_gc[1]) +
                            np.diff(f, axis=0, prepend=f_gc[0])) / dX[0] / 2.,
                           (np.diff(f, axis=1, append=f_gc[3]) +
                            np.diff(f, axis=1, prepend=f_gc[2])) / dX[1] / 2.,
                           (np.diff(f, axis=2, append=f_gc[5]) +
                            np.diff(f, axis=2, prepend=f_gc[4])) / dX[2] / 2.),
                          axis=-1)

def get_face_vels(vel, dX, BCs=None):
    if BCs == None:
        BCs = [[1, 1, 1, 1, 1, 1], [0., 0., 0., 0., 0., 0.]]
    vel_gc = get_GC_values(vel, BCs, dX)

    u = np.concatenate(
        (vel_gc[0][:, :, :, [0]], vel[:, :, :, [0]], vel_gc[1][:, :, :, [0]]),
        axis=0)
    u_f = (u[1:, :, :] + u[:-1, :, :]) / 2

    v = np.concatenate(
        (vel_gc[2][:, :, :, [1]], vel[:, :, :, [1]], vel_gc[3][:, :, :, [1]]),
        axis=1)
    v_f = (v[:, 1:, :] + v[:, :-1, :]) / 2

    w = np.concatenate(
        (vel_gc[4][:, :, :, [2]], vel[:, :, :, [2]], vel_gc[5][:, :, :, [2]]),
        axis=2)
    w_f = (w[:, :, 1:] + w[:, :, :-1]) / 2
    return u_f, v_f, w_f

def get_face_vel_component(vel, dX, axis=0, BCs=None):
    if BCs == None:
        BCs = [[1, 1, 1, 1, 1, 1], [0., 0., 0., 0., 0., 0.]]
    vel_gc = get_GC_values(vel, BCs, dX)

    if axis == 0:
        u = np.concatenate(
            (vel_gc[0], vel, vel_gc[1]),
            axis=0)
        u_f = (u[1:, :, :] + u[:-1, :, :]) / 2
        return u_f

    if axis == 1:
        v = np.concatenate(
            (vel_gc[2], vel, vel_gc[3]),
            axis=1)
        v_f = (v[:, 1:, :] + v[:, :-1, :]) / 2
        return v_f

    if axis == 2:
        w = np.concatenate(
            (vel_gc[4], vel, vel_gc),
            axis=2)
        w_f = (w[:, :, 1:] + w[:, :, :-1]) / 2
        return v_f


class AM_3d():
    def __init__(self, args):
        self.args = args
        self.default_args()
        self.msh = args['mesh']
        self.msh_local = args['mesh_local']
        self.meshio_mesh = args['meshio_mesh']
        self.t = 0.
        self.eqn_T_init(args)
        self.eqn_V_init(args)
        self.clean_sols()
        
    def default_args(self):
        if 'h' not in self.args:
            self.args['h'] = 0.
        if 'stefan_boltzmann' not in self.args:
            self.args['stefan_boltzmann'] = 5.67e-8
        if 'emissivity' not in self.args:
            self.args['emissivity'] = 0.
         

    def time_integration(self):
        if self.args['heat_source'] == 1:
            Q = self.get_body_heat_source(self.t)
        else:
            Q = self.T*0.
        self.T, T_BCs, _ = self.eqn_T.update(self.T, self.conv_T, Q,
                                             bc_args = (self.t,self.T[:,:,-1,:]), cell_conn = self.cell_conn)
        fl0 = self.fluid_frac(self.T)

#         self.solidID += fl0
        self.solidID = np.maximum(self.solidID,fl0)

        x0, x1, y0, y1, z0, z1 = self.get_moving_box_boundary(self.t)

        vel, vel_BCs, grad_p0 = self.eqn_V.update(
            self.vel[x0:x1, y0:y1, z0:z1], self.conv[x0:x1, y0:y1, z0:z1],
            self.grad_p0[x0:x1, y0:y1, z0:z1], fl0[x0:x1, y0:y1, z0:z1],
            self.T[x0:x1, y0:y1, z0:z1], self.cell_conn_local)

        conv_T, conv = self.update_convective_terms(
            self.T[x0:x1, y0:y1, z0:z1], vel, vel_BCs)

        self.vel = self.vel.at[x0:x1, y0:y1, z0:z1].set(vel)
        self.grad_p0 = self.grad_p0.at[x0:x1, y0:y1, z0:z1].set(grad_p0)
        self.conv = self.conv.at[x0:x1, y0:y1, z0:z1].set(conv)
        self.conv_T = self.conv_T.at[x0:x1, y0:y1, z0:z1].set(conv_T)
        self.t += self.args['dt']
        
# #### iterative scheme as a comparsion with the Non-interative scheme (explicit convection)       
#     def time_integration_iter(self,it=10):
#         Q = self.get_body_heat_source(self.t)
        
#         for i in range(it):
#             T, T_BCs, _ = self.eqn_T.update(self.T, self.conv_T, Q,
#                                                  self.t, self.cell_conn)
#             fl0 = self.fluid_frac(T)

#             self.solidID += fl0

#             x0, x1, y0, y1, z0, z1 = self.get_moving_box_boundary(self.t)

#             vel, vel_BCs, grad_p0 = self.eqn_V.update(
#                 self.vel[x0:x1, y0:y1, z0:z1], self.conv[x0:x1, y0:y1, z0:z1],
#                 self.grad_p0[x0:x1, y0:y1, z0:z1], fl0[x0:x1, y0:y1, z0:z1],
#                 T[x0:x1, y0:y1, z0:z1], self.cell_conn_local)

#             conv_T, conv = self.update_convective_terms(T[x0:x1, y0:y1, z0:z1], vel, vel_BCs)
            
#             self.grad_p0 = self.grad_p0.at[x0:x1, y0:y1, z0:z1].set(grad_p0)
#             self.conv = self.conv.at[x0:x1, y0:y1, z0:z1].set(conv)
#             self.conv_T = self.conv_T.at[x0:x1, y0:y1, z0:z1].set(conv_T)

#         self.vel = self.vel.at[x0:x1, y0:y1, z0:z1].set(vel)
#         self.T = T
#         self.t += self.args['dt']

    def get_body_heat_source(self, t):
        def Gaussian_cylinder(xl, yl, zl, P, xc, yc, zc):
            eta = self.args['eta']
            r = self.args['rb']
            d = self.args['phi'] * P / self.args['speed'] / self.args['rb']**2
            Q_laser = 2 * P * eta / d / np.pi / r**2 * np.exp(-2 * (
                (xc - xl)**2 + (yc - yl)**2) / r**2)
            Q_laser = Q_laser * ((zl - zc) <= d)
            return Q_laser

        xl, yl, zl, P = self.toolpath(t)
        return Gaussian_cylinder(xl, yl, zl, P, self.msh.Xc[:, 0],
                              self.msh.Xc[:, 1], self.msh.Xc[:, 2])

    def get_moving_box_boundary(self, t):
        ### for moving box
        xl, yl, zl, _ = self.toolpath(t)
        x0 = round(xl / self.msh.dX[0]) - round(self.msh_local.shape[0] / 2)
        x0 = np.clip(x0,0,self.msh.shape[0]-self.msh_local.shape[0])
        x1 = x0 + self.msh_local.shape[0]

        y0 = round(yl / self.msh.dX[1]) - round(self.msh_local.shape[1] / 2)
        y0 = np.clip(y0,0,self.msh.shape[1]-self.msh_local.shape[1])
        y1 = y0 + self.msh_local.shape[1]

        z0 = round(zl / self.msh.dX[2]) - self.msh_local.shape[2]
        z0 = self.msh.shape[2] - self.msh_local.shape[2]
        z1 = self.msh.shape[2]

        return x0, x1, y0, y1, z0, z1

    def update_convective_terms(self, T, vel, vel_BCs):
        vel_f = get_face_vels(vel, self.msh.dX, vel_BCs)
        conv_T = div(T, vel_f, self.msh.dX)
#         conv = div(vel, vel_f, self.msh.dX, BCs=vel_BCs)
        conv = div(vel, vel_f, self.msh.dX)
        return conv_T, conv

    def eqn_T_init(self, args):
        self.eqn_T = energy_eqn(args)
        self.eqn_T.bc_fn = self.get_energy_bc_fn()
        self.cell_conn = np.copy(args['mesh'].cell_conn)

        self.T = np.zeros(args['mesh'].shape + [1]) + args['T_ref']
        self.solidID = np.zeros(args['mesh'].shape + [1])
        self.conv_T = self.T * 0.
        self.Q = self.T * 0.

    def eqn_V_init(self, args):
        self.eqn_V = velosity_eqn(args)
        self.eqn_V.bc_fn = self.get_vel_bc_fn()
        self.cell_conn_local = np.copy(args['mesh_local'].cell_conn)

        self.vel = np.zeros(args['mesh'].shape + [3])
        self.conv = self.vel * 0.
        self.grad_p0 = self.vel * 0.

    def toolpath(self, t):
        xl = self.args['X0'][0] + t * self.args['speed']
        yl = self.args['X0'][1]
        zl = self.args['X0'][2]
        P = self.args['P'] * (t < self.args['t_OFF'])
        return xl, yl, zl, P

    def get_vel_bc_fn(self):
        Dgamma_Dt = self.args['Marangoni']
        visco = self.args['visco']

        def marangoni_effect(T_top):
            fl_top = self.fluid_frac(T_top)
            T_top = np.pad(T_top, ((1, 1), (1, 1), (0, 0)), 'symmetric')
            return np.stack(((T_top[2:, 1:-1] - T_top[:-2, 1:-1]) / 2. /
                             self.msh_local.dX[0] * Dgamma_Dt / visco * fl_top,
                             (T_top[1:-1, 2:] - T_top[1:-1, :-2]) / 2. /
                             self.msh_local.dX[1] * Dgamma_Dt / visco * fl_top,
                             np.zeros_like(T_top[1:-1, 1:-1])),
                            axis=3)

        def vel_bc(T):
            marangoni = marangoni_effect(T[:, :, -1])
            vel_bc_type = [0, 0, 0, 0, 0, 2]
            v0 = np.array([0., 0., 0.])
            vel_bc_values = [v0, v0, v0, v0, v0, marangoni]
            vel_BCs = [vel_bc_type, vel_bc_values]

            ## scaler form, for solving the poisson eqn
            bc_type = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0]]
            bc_values = []
            for i in range(3):
                bc_values.append([
                    np.zeros(self.msh_local.surf_set_num[0]),
                    np.zeros(self.msh_local.surf_set_num[1]),
                    np.zeros(self.msh_local.surf_set_num[2]),
                    np.zeros(self.msh_local.surf_set_num[3]),
                    np.zeros(self.msh_local.surf_set_num[4]),
                    marangoni[:, :, :, i].flatten() * visco,
                ])
            return vel_BCs, bc_type, bc_values

        return vel_bc

    def get_energy_bc_fn(self):
        def Gaussian_flux(xl, yl, P, xc, yc):
            eta = self.args['eta']
            r = self.args['rb']
            q_laser = 2 * P * eta / np.pi / r**2 * np.exp(-2 * (
                (xc - xl)**2 + (yc - yl)**2) / r**2)

            return q_laser

        def energe_bc(args):
            t,T_top = args
            q_conv = self.args['h']*(T_top-self.args['T_ref'])
            q_rad = self.args['stefan_boltzmann']*self.args['emissivity']*(T_top**4-self.args['T_ref']**4)
            q = -q_conv - q_rad
            bc_type = [1, 1, 1, 1, 1, 1]
            bc_values = [
                np.zeros(self.msh.surf_set_num[0]),
                np.zeros(self.msh.surf_set_num[1]),
                np.zeros(self.msh.surf_set_num[2]),
                np.zeros(self.msh.surf_set_num[3]),
                np.zeros(self.msh.surf_set_num[4]),
                q.flatten()
            ]



            if self.args['heat_source'] == 0:
                xc = self.msh.surface[self.msh.surf_sets[-1]][:, 0]
                yc = self.msh.surface[self.msh.surf_sets[-1]][:, 1]
                xl, yl, zl, P = self.toolpath(t)
                q_laser = Gaussian_flux(xl, yl, P, xc, yc)
                bc_values[-1] = bc_values[-1] + q_laser

            return bc_type, bc_values

        return energe_bc

    def fluid_frac(self, T):
        Tl = self.args['Tl']
        Ts = self.args['Ts']
        return (np.clip(T, Ts, Tl) - Ts) / (Tl - Ts)


    def clean_sols(self):
        cfd_vtk_sols_folder = os.path.join(self.args['data_dir'], "vtk/cfd/sols")
        os.makedirs(cfd_vtk_sols_folder, exist_ok=True)
        files_vtk = glob.glob(cfd_vtk_sols_folder + f"/*")
        for f in files_vtk:
            os.remove(f)

    def inspect_sol(self, step, num_steps):
        print(f"\nstep {step} of {num_steps}, unix timestamp = {time.time()}")
        print(f"T_max:{self.T.max()}, vmax:{np.linalg.norm(self.vel,axis=3).max()}")
        if not np.all(np.isfinite(self.T)):          
            raise ValueError(f"Found np.inf or np.nan in T0 - stop the program")

    def write_sols(self, step):
        print(f"\nWrite CFD sols to file...")
        step = step // self.args['write_sol_interval']
        self.meshio_mesh.cell_data['T'] = [onp.array(self.T.reshape(-1, 1), dtype=onp.float32)]
        self.meshio_mesh.cell_data['vel'] = [onp.array(self.vel.reshape(-1, 3), dtype=onp.float32)]
        self.meshio_mesh.cell_data['solidID'] = [onp.array(self.solidID.reshape(-1, 1), dtype=onp.float32)]
        self.meshio_mesh.write(os.path.join(self.args['data_dir'], f"vtk/cfd/sols/u{step:03d}.vtu"))

        
class poisson():
    def __init__(self, mesh, nonlinear=False, mu=None, mu_fn=None,source_fn=None,bc_type=None, bc_value=None):
        self.msh = mesh
        self.ndof = self.msh.cell_num

        self.nonlinear = nonlinear
        if self.nonlinear == False:
            assert np.isscalar(
                mu), 'For linear problem, a constant mu value is required'
            self.mu = mu
            self.laplace_kernel = self.linear_laplace_kernel
        else:
            assert mu_fn != None, 'For nonlinear problem, please input mu_fn: mu = mu_fn(U)'
            self.mu_fn = mu_fn
            self.laplace_kernel = self.nonlinear_laplace_kernel

        if source_fn == None:
            self.source_fn = lambda *args : 0
        else:
            self.source_fn = source_fn

        self.set_bc(bc_type, bc_value)

    def linear_laplace_kernel(self, Up, Unb):
        return ((Unb - Up) / self.msh.D * self.mu * self.msh.dS).sum()

    def nonlinear_laplace_kernel(self, Up, Unb):
        # see Patankar, Suhas V. Numerical heat transfer and fluid flow. CRC press, 2018. page 44-47
        mu = 2*self.mu_fn(Up)*self.mu_fn(Unb)/(self.mu_fn(Unb)+self.mu_fn(Up))
        return ((Unb - Up) / self.msh.D * mu * self.msh.dS).sum()

    def laplace(self, U):
        U_cell_nb = U[self.msh.cell_conn]
        return jax.vmap(self.laplace_kernel)(U_cell_nb[:, 0], U_cell_nb[:, 1:])

    def source(self,U,*args):
        return jax.vmap(self.source_fn)(U,*args)

    def apply_BC_to_residual(self, U, residual):
        flux = np.array([])
        if self.bc_type != None:
            for i in range(len(self.bc_type)):
                if self.bc_type[i] == 0:
                    if self.nonlinear == True:
                        mu = self.mu_fn(self.bc_value[i])
                    else:
                        mu = self.mu
                    flux = np.concatenate(
                        (flux, mu * 2 *
                         (self.bc_value[i] -
                          U[self.msh.surf_cell[self.msh.surf_sets[i]]]) /
                         self.msh.D[i] * self.msh.dS[i]))
                if self.bc_type[i] == 1:
                    flux = np.concatenate(
                        (flux, self.bc_value[i] * np.abs(self.msh.dS[i])))
            residual = residual.at[self.msh.surf_cell].add(-flux)
        return residual

    def compute_residual_kernel(self, U_cell_nb, *args):
        Up = U_cell_nb[0]
        Unb = U_cell_nb[1:]
        return -self.laplace_kernel(Up, Unb) - self.source_fn(Up,*args)*self.msh.dV

    def compute_residual(self, U, *args):
        residual = -self.laplace(U) - self.source(U,*args)*self.msh.dV
        residual = self.apply_BC_to_residual(U, residual)
        return residual

    def apply_BC_to_matrix(self, values):
        value_Dirichlet = np.array([])
        loc_Dirichlet = np.array([], dtype=int)
        if self.bc_type != None:
            for i in range(len(self.bc_type)):
                if self.bc_type[i] == 0:
                    if self.nonlinear == True:
                        mu = self.mu_fn(self.bc_value[i])
                    else:
                        mu = self.mu + self.bc_value[i]*0.
                    value_Dirichlet = np.concatenate(
                        (value_Dirichlet,
                         mu * 2 / self.msh.D[i] * self.msh.dS[i]))
                    loc_Dirichlet = np.concatenate(
                        (loc_Dirichlet, self.msh.surf_cell[self.msh.surf_sets[i]]))
            values = values.at[loc_Dirichlet, 0].add(value_Dirichlet)
        return values

    def newton_update(self, U, *args):
        def D_fn(cell_nb_sol, *args):
            return jax.jacrev(self.compute_residual_kernel)(cell_nb_sol, *args)

        vmap_fn = jax.vmap(D_fn)
        values = vmap_fn(U[self.msh.cell_conn],*args)
        self.values = self.apply_BC_to_matrix(values)
        self.precond_values = ((self.msh.cell_conn == np.arange(self.msh.cell_num)[:,None])*self.values).sum(axis=1)

    def compute_linearized_residual(self, U):
        return (self.values * U[self.msh.cell_conn]).sum(axis=1)

    def jacobiPreconditioner(self,U):
        # return (1./self.values[:,0]*U)
        return (1./self.precond_values*U)

    def set_bc(self, bc_type, bc_value):
        self.bc_type = bc_type
        self.bc_value = bc_value


class energy_eqn():
    def __init__(self, args):
        cp_fn = lambda T: args['cp'](T) + args['latent_heat'] / (args[
            'Tl'] - args['Ts']) * (T > args['Ts']) * (T < args['Tl'])
        fn = lambda T, T0, conv, Q: -args['rho'] * (T - T0) * cp_fn(
            T0) / args['dt'] - args['rho'] * cp_fn(T0) * conv + Q
        mu_fn = lambda T: args['k'](T)
        self.step = poisson(mesh=args['mesh'],
                            nonlinear='True',
                            mu_fn=mu_fn,
                            source_fn=fn)
        self.bc_fn = None
    
    def update_BCs(self,*bc_args):
        if self.bc_fn != None:
            bc_types,bc_values = self.bc_fn(*bc_args)
            self.BCs = [bc_types,bc_values]
            self.step.set_bc(bc_types,bc_values)
        

    @partial(jax.jit, static_argnums=(0))
    def update(self, T0, conv_T0, Q, bc_args, cell_conn):
        self.step.msh.cell_conn = cell_conn
        self.update_BCs(bc_args)
        T, it = solver_nonlinear(self.step,
                                 T0.flatten(),
                                 conv_T0.flatten(),
                                 Q.flatten(),
                                 init=T0.flatten(),
                                 precond=True)
        return T.reshape(T0.shape), self.BCs, it


class velosity_eqn():
    def __init__(self, args):

        self.dt = args['dt']
        self.rho = args['rho']
        fn = lambda u, u0, conv, grad_p, fl: -self.rho * (
            u - u0) / self.dt - self.rho*conv - grad_p - 1e7 * self.rho * (
                1 - fl)**2 / (fl**3 + 1e-5) * u
        self.step = poisson(mesh=args['mesh_local'],
                            nonlinear= False,
                            mu=args['visco'],
                            source_fn=fn)

        source = lambda p, div_v: -div_v
        self.poisson_for_p = poisson(mesh=args['mesh_local'],
                                     mu=1.,
                                     source_fn=source)

        self.poisson_for_p.newton_update(np.zeros(self.poisson_for_p.ndof),
                                         np.zeros(self.poisson_for_p.ndof))

        self.bc_fn = None

    def update_BCs(self,*bc_args):
        if self.bc_fn != None:
            self.vel_BCs,bc_types,bc_values = self.bc_fn(*bc_args) #vel_BCs: vector form for calculate face velosities;
            return bc_types,bc_values

    @partial(jax.jit, static_argnums=(0))
    def update(self, vel0, conv0, grad_p0, fl, bc_args, cell_conn):
        self.step.msh.cell_conn = cell_conn
        # prediction step
        vel = []
        bc_types,bc_values = self.update_BCs(bc_args)
        for i in range(0, 3):
            self.step.set_bc(bc_types[i],bc_values[i])
            vel_i = solver_linear(self.step,
                        vel0[:, :, :, i].flatten(),
                        conv0[:, :, :, i].flatten(),
                        grad_p0[:, :, :, i].flatten(),
                        fl.flatten(),
                        tol=1e-10,
                        precond=True).reshape(self.step.msh.shape)
            vel.append(vel_i)
        vel = np.stack((vel[0], vel[1], vel[2]),axis=3)

        # correction step
        u_f, v_f, w_f = get_face_vels(vel, self.step.msh.dX, self.vel_BCs)
        div_vel = ((u_f[1:, :, :] - u_f[:-1, :, :]) / self.step.msh.dX[0] +
                   (v_f[:, 1:, :] - v_f[:, :-1, :]) / self.step.msh.dX[1] +
                   (w_f[:, :, 1:] - w_f[:, :, :-1]) /
                   self.step.msh.dX[2]) / self.dt * self.rho 

        p = solver_linear(self.poisson_for_p,
                          div_vel.flatten(),
                          tol=1e-10,
                          update=False,
                          precond=False)

        p = p.reshape(self.step.msh.shape + [1])
        vel = vel - self.dt * gradient(p, self.step.msh.dX) / self.rho
        # vel_f = get_face_vels(vel,self.step.msh.dX,self.vel_BCs)
        return vel, self.vel_BCs, grad_p0 + gradient(p, self.step.msh.dX)


class uniform_mesh():
    def __init__(self, domain, N):
        self.shape = N
        self.dim = len(N)
        self.generate_mesh(domain, N)
        self.get_neighbor(N)
        self.get_surface(N)


class mesh2d(uniform_mesh):
    def generate_mesh(self, domain, N):
        Nx, Ny = N
        domain_x, domain_y = domain
        self.cell_num = Nx * Ny

        x = onp.linspace(0., domain_x, Nx + 1)
        y = onp.linspace(0., domain_y, Ny + 1)

        xx, yy = onp.meshgrid(x, y, indexing='ij')

        xc = onp.array((xx[1:, 1:] + xx[:-1, 1:]) / 2)
        yc = onp.array((yy[1:, 1:] + yy[1:, :-1]) / 2)

        self.dX = np.array([domain_x / Nx, domain_y / Ny])
        self.Xc = np.stack((xc, yc), axis=2).reshape(-1, 2)

        self.dV = self.dX[0] * self.dX[1]
        self.D = np.array([-self.dX[0], self.dX[0], -self.dX[1], self.dX[1]])
        self.dS = np.array([
            -self.dX[1],
            self.dX[1],
            -self.dX[0],
            self.dX[0],
        ])

    def get_neighbor(self, N):
        Nx, Ny = N
        cell_idx = onp.arange(0, Nx * Ny).reshape((Nx, Ny))
        cell_index = onp.pad(cell_idx, 1, "symmetric")
        self.cell_conn = np.stack(
            (cell_index[1:-1, 1:-1], cell_index[:-2, 1:-1],
             cell_index[2:, 1:-1], cell_index[1:-1, :-2], cell_index[1:-1,
                                                                     2:]),
            axis=2).reshape(-1, 5)

    def get_surface(self, N):
        Nx, Ny = N
        cell_idx = np.arange(0, Nx * Ny).reshape((Nx, Ny))
        x_neg_idx = cell_idx[0, :].flatten()
        x_pos_idx = cell_idx[-1, :].flatten()
        y_neg_idx = cell_idx[:, 0].flatten()
        y_pos_idx = cell_idx[:, -1].flatten()

        self.surf_cell = np.concatenate(
            (x_neg_idx, x_pos_idx, y_neg_idx, y_pos_idx))

        self.surf_sets = [
            np.arange(0, Ny),
            np.arange(Ny, Ny * 2),
            np.arange(Ny * 2, Ny * 2 + Nx),
            np.arange(Ny * 2 + Nx, Ny * 2 + Nx * 2)
        ]

        self.surf_set_num = [Ny, Ny, Nx, Nx]
        self.surf_orient = np.concatenate((np.ones(Ny) * 0, np.ones(Ny) * 1,
                                           np.ones(Nx) * 2, np.ones(Nx) * 3))

        surf_x_neg = self.Xc[x_neg_idx] - onp.array([self.dX[0] / 2., 0.])
        surf_x_pos = self.Xc[x_pos_idx] + onp.array([self.dX[0] / 2., 0.])
        surf_y_neg = self.Xc[y_neg_idx] - onp.array([0., self.dX[1] / 2.])
        surf_y_pos = self.Xc[y_pos_idx] + onp.array([0., self.dX[1] / 2.])
        self.surface = np.concatenate(
            (surf_x_neg, surf_x_pos, surf_y_neg, surf_y_pos))


class mesh3d(uniform_mesh):
    def generate_mesh(self, domain, N):
        Nx, Ny, Nz = N
        domain_x, domain_y, domain_z = domain
        self.cell_num = Nx * Ny * Nz
        x = onp.linspace(0., domain_x, Nx + 1)
        y = onp.linspace(0., domain_y, Ny + 1)
        z = onp.linspace(0., domain_z, Nz + 1)

        xx, yy, zz = onp.meshgrid(x, y, z, indexing='ij')

        xc = onp.array((xx[1:, 1:, 1:] + xx[:-1, 1:, 1:]) / 2)
        yc = onp.array((yy[1:, 1:, 1:] + yy[1:, :-1, 1:]) / 2)
        zc = onp.array((zz[1:, 1:, 1:] + zz[1:, 1:, :-1]) / 2)

        self.dX = np.array([domain_x / Nx, domain_y / Ny, domain_z / Nz])
        self.Xc = np.stack((xc, yc, zc), axis=3).reshape(-1, 3)

        self.dV = self.dX[0] * self.dX[1] * self.dX[2]
        self.D = np.array([
            -self.dX[0], self.dX[0], -self.dX[1], self.dX[1], -self.dX[2],
            self.dX[2]
        ])
        self.dS = np.array([
            -self.dX[1] * self.dX[2], self.dX[1] * self.dX[2],
            -self.dX[2] * self.dX[0], self.dX[2] * self.dX[0],
            -self.dX[0] * self.dX[1], self.dX[0] * self.dX[1]
        ])

    def get_neighbor(self, N):
        Nx, Ny, Nz = N
        cell_idx = onp.arange(0, Nx * Ny * Nz).reshape((Nx, Ny, Nz))
        cell_index = onp.pad(cell_idx, 1, "symmetric")
        self.cell_conn = np.stack(
            (cell_index[1:-1, 1:-1, 1:-1], cell_index[:-2, 1:-1, 1:-1],
             cell_index[2:, 1:-1, 1:-1], cell_index[1:-1, :-2, 1:-1],
             cell_index[1:-1, 2:, 1:-1], cell_index[1:-1, 1:-1, :-2],
             cell_index[1:-1, 1:-1, 2:]),
            axis=3).reshape(-1, 7)

    def get_surface(self, N):
        Nx, Ny, Nz = N
        cell_idx = np.arange(0, Nx * Ny * Nz).reshape((Nx, Ny, Nz))
        x_neg_idx = cell_idx[0, :, :].flatten()
        x_pos_idx = cell_idx[-1, :, :].flatten()
        y_neg_idx = cell_idx[:, 0, :].flatten()
        y_pos_idx = cell_idx[:, -1, :].flatten()
        z_neg_idx = cell_idx[:, :, 0].flatten()
        z_pos_idx = cell_idx[:, :, -1].flatten()
        self.surf_cell = np.concatenate(
            (x_neg_idx, x_pos_idx, y_neg_idx, y_pos_idx, z_neg_idx, z_pos_idx))

        self.surf_sets = [
            np.arange(0, Ny * Nz),
            np.arange(Ny * Nz, Ny * Nz * 2),
            np.arange(Ny * Nz * 2, Ny * Nz * 2 + Nx * Nz),
            np.arange(Ny * Nz * 2 + Nx * Nz, Ny * Nz * 2 + Nx * Nz * 2),
            np.arange(Ny * Nz * 2 + Nx * Nz * 2,
                      Ny * Nz * 2 + Nx * Nz * 2 + Nx * Ny),
            np.arange(Ny * Nz * 2 + Nx * Nz * 2 + Nx * Ny,
                      Ny * Nz * 2 + Nx * Nz * 2 + Nx * Ny * 2)
        ]

        self.surf_set_num = [
            Ny * Nz, Ny * Nz, Nx * Nz, Nx * Nz, Ny * Nx, Ny * Nx
        ]
        self.surf_orient = np.concatenate(
            (np.ones(Ny * Nz) * 0, np.ones(Ny * Nz) * 1, np.ones(Nx * Nz) * 2,
             np.ones(Nx * Nz) * 3, np.ones(Ny * Nx) * 4, np.ones(Ny * Nx) * 5))

        surf_x_neg = self.Xc[x_neg_idx] - onp.array([self.dX[0] / 2., 0., 0.])
        surf_x_pos = self.Xc[x_pos_idx] + onp.array([self.dX[0] / 2., 0., 0.])
        surf_y_neg = self.Xc[y_neg_idx] - onp.array([0., self.dX[1] / 2., 0.])
        surf_y_pos = self.Xc[y_pos_idx] + onp.array([0., self.dX[1] / 2., 0.])
        surf_z_neg = self.Xc[z_neg_idx] - onp.array([0., 0., self.dX[2] / 2.])
        surf_z_pos = self.Xc[z_pos_idx] + onp.array([0., 0., self.dX[2] / 2.])
        self.surface = np.concatenate((surf_x_neg, surf_x_pos, surf_y_neg,
                                       surf_y_pos, surf_z_neg, surf_z_pos))


def solver_linear(eqn,*args,tol=1e-6,precond=False,update=True,relative=False):
# solve linear problems
    
    dofs = np.zeros(eqn.ndof)
    
    res = eqn.compute_residual(dofs,*args)
    if update:
        eqn.newton_update(dofs,*args)

    if relative:
        eqn.values = eqn.values.at[0].set(np.array([1., 0., 0., 0., 0., 0., 0.]))


    A_fn = eqn.compute_linearized_residual
    if precond:
        preconditoner = eqn.jacobiPreconditioner
        if relative:
            eqn.precond_values = eqn.precond_values.at[0].set(1.)
    else:
        preconditoner = None

    b = -res
    if relative:
        b = b.at[0].set(0.)

    # TODO(Tianju): Any way to detect if CG does not converge? The program can get stuck.
    inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, M=preconditoner, x0=None, tol=tol,maxiter=10000) # bicgstab
    dofs =  dofs + inc
    
    return dofs


def solver_nonlinear(eqn,*args,init=None,tol=1e-5,max_it=50,relaxation=1.,precond=False):
# solve nonlinear problems

    if precond:
        preconditoner = eqn.jacobiPreconditioner
    else:
        preconditoner = None

    if init == None:
        dofs = np.zeros(eqn.ndof)
    else:
        dofs = init

    b = -eqn.compute_residual(dofs,*args)
    res_init = np.linalg.norm(b)

    def cond_fun(carry):
        dofs,b,it = carry
        return (np.linalg.norm(b)/np.linalg.norm(res_init) > tol) & (it < max_it)

    def body_fun(carry):
        dofs,b,it = carry
        eqn.newton_update(dofs,*args)
        A_fn_linear = eqn.compute_linearized_residual
        inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b, M=preconditoner)
        dofs = dofs + inc*relaxation
        b = -eqn.compute_residual(dofs,*args)
        return (dofs,b,it+1)
    

    it = 0
    inc = np.ones_like(dofs)
    dofs,b,it = jax.lax.while_loop(cond_fun, body_fun, (dofs,b,it))

    
    return dofs,it