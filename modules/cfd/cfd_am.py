import jax
import jax.numpy as np
import numpy as onp
import scipy
from jax.experimental.sparse import BCOO
from functools import partial


class poisson():
    def __init__(self, mesh, nonlinear=False, miu=None, miu_fn=None,source_fn=None,bc_type=None, bc_value=None):
        self.msh = mesh
        self.ndof = self.msh.cell_num

        self.nonlinear = nonlinear
        if self.nonlinear == False:
            assert np.isscalar(
                miu), 'For linear problem, a constant miu value is required'
            self.miu = miu
            self.laplace_kernel = self.linear_laplace_kernal
        else:
            assert miu_fn != None, 'For nonlinear problem, please input miu_fn: miu = miu_fn(U)'
            self.miu_fn = miu_fn
            self.laplace_kernel = self.nonlinear_laplace_kernal

        if source_fn == None:
            self.source_fn = lambda *args : 0
        else:
            self.source_fn = source_fn

        self.set_bc(bc_type, bc_value)

    def linear_laplace_kernal(self, Up, Unb):
        return ((Unb - Up) / self.msh.D * self.miu * self.msh.dS).sum()

    def nonlinear_laplace_kernal(self, Up, Unb):
        miu = self.miu_fn(Up / 2. + Unb / 2.)
        return ((Unb - Up) / self.msh.D * miu * self.msh.dS).sum()

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
                        miu = self.miu_fn(self.bc_value[i])
                    else:
                        miu = self.miu
                    flux = np.concatenate(
                        (flux, miu * 2 *
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
                        miu = self.miu_fn(self.bc_value[i])
                    else:
                        miu = self.miu + self.bc_value[i]*0.
                    value_Dirichlet = np.concatenate(
                        (value_Dirichlet,
                         miu * 2 / self.msh.D[i] * self.msh.dS[i]))
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

    def compute_linearized_residual(self, U):
        return (self.values * U[self.msh.cell_conn]).sum(axis=1)

    def jacobiPreconditioner(self,U):
        return (self.values[:,0]*U)

    def set_bc(self, bc_type, bc_value):
        self.bc_type = bc_type
        self.bc_value = bc_value


class poisson_transient():
    def __init__(self,
                 mesh,
                 dt,
                 nonlinear=False,
                 miu=None,
                 rho=None,
                 miu_fn=None,
                 rho_fn=None,
                 source_fn=None):
        
        if source_fn == None:
            source_fn = lambda *args : 0
        if nonlinear: 
            assert rho_fn != None, 'For nonlinear problem, please input rho_fn: rho = rho_fn(U)'
            self.rho_fn = rho_fn
            fn = lambda U,U0,*args: -(rho_fn(U)/2.+rho_fn(U0)/2.)*(U-U0)/dt + source_fn(U,*args)
        else:
            assert np.isscalar(rho), 'For linear problem, a constant rho value is required'
            fn = lambda U,U0,*args: -rho*(U-U0)/dt + source_fn(U,*args)
            self.rho = rho
        self.step = poisson(mesh, nonlinear, miu, miu_fn, source_fn=fn)



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



class AM_3d():
    def __init__(self, params):
        self.params = params
        self.msh = params['mesh']
        self.msh_v = params['mesh_local']
        self.shape = params['mesh'].shape

        self.t = 0.
        self.dt = params['dt']
        self.T0 = np.zeros(self.shape) + params['T_ref']
        self.conv_T0 = np.zeros(self.shape)
        self.vel0 = np.zeros((self.shape[0],self.shape[1],self.shape[2],3))
        self.p0 = np.zeros((self.shape[0],self.shape[1],self.shape[2],1))

        rho_cp = lambda T: params['cp'](T) * params['rho']
        source = lambda T, conv: -conv
        self.eqn_T = poisson_transient(mesh=params['mesh'],
                                       dt=params['dt'],
                                       nonlinear=True,
                                       miu_fn=params['k'],
                                       rho_fn=rho_cp,
                                       source_fn=source)

        source = lambda u, conv, fl, grad_p: -1e5 * params['rho'] * (
            1 - fl)**2 / (fl**3 + 1e-3) * u - params['rho'] * conv - grad_p

        self.eqn_v = poisson_transient(mesh=params['mesh_local'],
                                       dt=params['dt'],
                                       nonlinear=False,
                                       miu=params['visco'],
                                       rho=params['rho'],
                                       source_fn=source)

        source = lambda p, div_v: -div_v
        self.poisson_for_p = poisson(mesh=params['mesh_local'],
                                     miu=1.,
                                     source_fn=source)
        self.poisson_for_p.newton_update(np.zeros(self.poisson_for_p.ndof),
                                         np.zeros(self.poisson_for_p.ndof))

        # for better JIT time, very ugly, to be modified
        self.cell_conn = np.copy(params['mesh'].cell_conn)
        self.cell_conn_local = np.copy(params['mesh_local'].cell_conn)

        ## vector form, for calculate ghost cell value
        self.vel_bc_type = [0, 0, 0, 0, 0, 0]
        v0 = np.array([0., 0., 0.])
        self.vel_bc_values = [v0, v0, v0, v0, v0, v0]

    def fluid_frac(self, T):
        Tl = self.params['Tl']
        Ts = self.params['Ts']
        return (np.clip(T, Ts, Tl) - Ts) / (Tl - Ts)
        
    def update_vel_BC(self, T_top):
        fl_top = self.fluid_frac(T_top)
        Dgamma_Dt = self.params['Marangoni']
        visco = self.params['visco']
        T_top = np.pad(T_top, ((1, 1), (1, 1), (0, 0)), 'symmetric')

        marangoni = np.stack(
            ((T_top[2:, 1:-1] - T_top[:-2, 1:-1]) / 2 / self.msh_v.dX[0] *
             Dgamma_Dt / visco * fl_top,
             (T_top[1:-1, 2:] - T_top[1:-1, :-2]) / 2 / self.msh_v.dX[1] *
             Dgamma_Dt / visco * fl_top, np.zeros_like(T_top[1:-1, 1:-1])),
            axis=3)

        ## vector form, for calculate ghost cell value
        self.vel_bc_type = [0, 0, 0, 0, 0, 2]
        v0 = np.array([0., 0., 0.])
        self.vel_bc_values = [v0, v0, v0, v0, v0, marangoni]

        ## scaler form, for solving the poisson eqn
        vel_bc_type = [[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0]]

        vel_bc_values = []
        for i in range(3):
            vel_bc_values.append([
                np.zeros(self.msh_v.surf_set_num[0]),
                np.zeros(self.msh_v.surf_set_num[1]),
                np.zeros(self.msh_v.surf_set_num[2]),
                np.zeros(self.msh_v.surf_set_num[3]),
                np.zeros(self.msh_v.surf_set_num[4]),
                marangoni[:, :, :, i].flatten() * visco,
            ])
        return vel_bc_type, vel_bc_values

    def update_engergy_BC(self, eqn, t):
        xl = self.params['X0'][0] + t * self.params['speed']
        yl = self.params['X0'][1]
        P = self.params['P']
        eta = self.params['eta']
        r = self.params['rb']

        xc = eqn.msh.surface[eqn.msh.surf_sets[-1]][:, 0]
        yc = eqn.msh.surface[eqn.msh.surf_sets[-1]][:, 1]
        q_laser = 2 * P * eta / np.pi / r**2 * np.exp(-2 *
                                                      ((xc - xl)**2 +
                                                       (yc - yl)**2) / r**2)

        q_laser = q_laser*(t<self.params['t_OFF']) ## laser off
        # top surf: laser + conv., other surfs: adiabatic
        bc_type = [1, 1, 1, 1, 1, 1]
        bc_values = [
            np.zeros(eqn.msh.surf_set_num[0]),
            np.zeros(eqn.msh.surf_set_num[1]),
            np.zeros(eqn.msh.surf_set_num[2]),
            np.zeros(eqn.msh.surf_set_num[3]),
            np.zeros(eqn.msh.surf_set_num[4]), q_laser
        ]
        eqn.set_bc(bc_type, bc_values)
        
    @partial(jax.jit, static_argnums=0)
    def update_tempearture(self, t, T0, conv_T0,cell_conn):
        # for better JIT time, very ugly, to be modified
        self.eqn_T.step.msh.cell_conn = cell_conn
        self.update_engergy_BC(self.eqn_T.step, t)

        T = solver_nonlinear(self.eqn_T.step, T0.flatten(), conv_T0.flatten(), init=T0.flatten()).reshape(self.shape)
        fl = self.fluid_frac(T)

        # for update vel BC
        T_top = T[:, :, -1] + self.msh.dX[2] / 2 * self.eqn_T.step.bc_value[
            -1].reshape(self.shape[0], self.shape[1]) / self.eqn_T.step.miu_fn(
                T[:, :, -1])
        
        return T, fl, T_top

    @partial(jax.jit, static_argnums=0)
    def update_velosity(self, vel0, p0, fl, T0_top, cell_conn):
        # for better JIT time, very ugly, to be modified
        self.eqn_v.step.msh.cell_conn = cell_conn

        vel0_gc = get_GC_values(vel0, [self.vel_bc_type, self.vel_bc_values],
                                self.msh_v.dX)
        vel0_f = get_face_vels(vel0, self.msh_v.dX,
                               [self.vel_bc_type, self.vel_bc_values])
        conv0 = div(vel0,
                    vel0_f,
                    self.msh_v.dX,
                    BCs=[self.vel_bc_type, self.vel_bc_values])
        grad_p0 = gradient(p0, self.msh_v.dX)

        # update vel BC
        vel_bc_type, vel_bc_values = self.update_vel_BC(T0_top)

        # prediction step
        vel = []
        for i in range(0, 3):
            self.eqn_v.step.set_bc(vel_bc_type[i], vel_bc_values[i])
            vel.append(
                solver_linear(self.eqn_v.step,
                              vel0[:, :, :, i].flatten(),
                              conv0[:, :, :, i].flatten(),
                              fl.flatten(),
                              grad_p0[:, :, :, i].flatten(),
                              tol=1e-6))

        vel = np.stack((vel[0].reshape(self.msh_v.shape), vel[1].reshape(
            self.msh_v.shape), vel[2].reshape(self.msh_v.shape)),
                       axis=3)

        # pressure eqn
        u_f, v_f, w_f = get_face_vels(vel, self.msh_v.dX,
                                      [self.vel_bc_type, self.vel_bc_values])
        div_vel = ((u_f[1:, :, :] - u_f[:-1, :, :]) / self.msh_v.dX[0] +
                   (v_f[:, 1:, :] - v_f[:, :-1, :]) / self.msh_v.dX[1] +
                   (w_f[:, :, 1:] - w_f[:, :, :-1]) /
                   self.msh_v.dX[2]) / self.dt * self.params['rho']
        p = solver_linear(self.poisson_for_p,
                          div_vel.flatten(),
                          tol=1e-6,
                          update=True)
        p = p.reshape(self.msh_v.shape[0], self.msh_v.shape[1],
                      self.msh_v.shape[2], 1)

        # correction step
        vel = vel - self.dt * gradient(p, self.msh.dX) / self.params['rho']
        p += p0
        return vel, p
    
    def time_integration(self):
        self.T0, fl, T0_top = self.update_tempearture(self.t, self.T0,self.conv_T0,
                                                       self.cell_conn)
        
        ### for moving box, modification needed
        x0 = int(self.msh.shape[0]/2) - int(self.msh_v.shape[0]/2)
        x1 = x0 + self.msh_v.shape[0]

        y0 = int(self.msh.shape[1]/2) - int(self.msh_v.shape[1]/2)
        y1 = y0 + self.msh_v.shape[1]

        z0 = self.msh.shape[2] - self.msh_v.shape[2]
        z1 = self.msh.shape[2]
        
        vel_local, p  = self.update_velosity(self.vel0[x0:x1,y0:y1,z0:z1], 
                                                self.p0[x0:x1,y0:y1,z0:z1],
                                                fl[x0:x1,y0:y1,z0:z1], 
                                                T0_top[x0:x1,y0:y1, None],
                                                self.cell_conn_local)
    
        self.vel0 = self.vel0.at[x0:x1,y0:y1,z0:z1].set(vel_local)
        self.p0 = self.p0.at[x0:x1,y0:y1,z0:z1].set(p)
        
        vel0_f = get_face_vels(self.vel0,self.msh.dX)
        self.conv_T0 = div(self.T0[:,:,:,None]*self.eqn_T.rho_fn(self.T0[:,:,:,None]),vel0_f,self.msh.dX)
    
        self.t += self.dt


# class AM_transient():
#     def __init__(self,
#                  mesh,
#                  dt,
#                  nonlinear=False,
#                  miu=None,
#                  rho=None,
#                  miu_fn=None,
#                  rho_fn=None,
#                  source_fn=None):
#         self.msh = mesh
#         self.poisson = poisson(mesh, nonlinear, miu, miu_fn, source_fn)
#         self.rho = rho
#         self.ndof = self.poisson.ndof
#         self.nonlinear = nonlinear
#         if self.nonlinear == False:
#             assert np.isscalar(rho), 'For linear problem, a constant rho value is required'
#             self.rho = rho
#         else:
#             assert rho_fn != None, 'For nonlinear problem, please input rho_fn: rho = rho_fn(U)'
#             self.rho_fn = rho_fn
#         self.dt = dt

#     def compute_residual(self, U, U0):
#         if self.nonlinear:
#             rho = self.rho_fn(U)
#         else:
#             rho = U*0. + self.rho
#         return self.poisson.compute_residual(
#             U, U0) + (U - U0) / self.dt * rho * self.msh.dV

#     def newton_update(self, U, U0):
#         self.poisson.newton_update(U, U0)
#         if self.nonlinear:
#             rho = self.rho_fn(U)
#         else:
#             rho = U*0. + self.rho
#         self.poisson.values = self.poisson.values.at[:, 0].add(rho / self.dt * self.msh.dV)

#     def compute_linearized_residual(self, U):
#         return self.poisson.compute_linearized_residual(U)

#     def jacobiPreconditioner(self,U):
#         return self.poisson.jacobiPreconditioner(U)

#     def update_BC(self, t):
#         speed = .8  # scan speed
#         x0, y0 = 5e-4, 5e-4
#         xl = x0 + t * speed
#         yl = y0
#         P = 195.  # power
#         eta = 0.43  # absorptivity
#         r = 5e-5  # 1/e^2 radius
#         xc = self.msh.surface[self.msh.surf_sets[-1]][:, 0]
#         yc = self.msh.surface[self.msh.surf_sets[-1]][:, 1]
#         q_laser = (xc - xl)
#         q_laser = 2 * P * eta / np.pi / r**2 * np.exp(-2 *
#                                                       ((xc - xl)**2 +
#                                                        (yc - yl)**2) / r**2)

#         # top surf: laser + conv., other surfs: adiabatic
#         bc_type = [1, 1, 1, 1, 1, 1]
#         bc_values = [
#             np.zeros(self.msh.surf_set_num[0]),
#             np.zeros(self.msh.surf_set_num[1]),
#             np.zeros(self.msh.surf_set_num[2]),
#             np.zeros(self.msh.surf_set_num[3]),
#             np.zeros(self.msh.surf_set_num[4]), q_laser
#         ]
#         self.poisson.set_bc(bc_type, bc_values)

#     # @partial(jax.jit, static_argnums=0)
#     def time_integration(self, U0, t, cell_conn):
#         # for better JIT time, very ugly, to be modified
#         self.poisson.msh.cell_conn = cell_conn
#         self.msh.cell_conn = cell_conn
        
#         self.update_BC(t)
#         if self.nonlinear == False:
#             dofs = solver_linear(self, U0, tol=1e-6)
#         else:
#             dofs = solver_nonlinear(self, U0)


#         T0_top_cell = dofs[self.msh.surf_cell[self.msh.surf_sets[-1]]]
#         T0_top = T0_top_cell + self.msh.dX[2]/2*self.poisson.bc_value[-1]/self.poisson.miu_fn(T0_top_cell)
#         T0_top = T0_top.reshape(self.msh.shape[0],self.msh.shape[1],1)
#         return dofs, T0_top



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


def solver_linear(eqn,*args,tol=1e-6,precond=False,update=True):
# solve linear problems
    
    dofs = np.zeros(eqn.ndof)
    
    res = eqn.compute_residual(dofs,*args)
    if update:
        eqn.newton_update(dofs,*args)

    A_fn = eqn.compute_linearized_residual
    if precond:
        preconditoner = eqn.jacobiPreconditioner
    else:
        preconditoner = None
    b = -res

    inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn, b, M=preconditoner, x0=None, tol=tol,maxiter=10000) # bicgstab
    dofs =  dofs + inc
    
    return dofs



def solver_nonlinear(eqn,*args,init=None,tol=1e-5,max_it=5000):
# solve nonlinear problems

    def cond_fun(carry):
        dofs,inc,it = carry
        return (np.linalg.norm(inc)/np.linalg.norm(dofs) > tol) & (it < max_it)

    def body_fun(carry):
        dofs,inc,it = carry
        b = -eqn.compute_residual(dofs,*args)
        eqn.newton_update(dofs,*args)
        A_fn_linear = eqn.compute_linearized_residual
        inc, info = jax.scipy.sparse.linalg.bicgstab(A_fn_linear, b)
        dofs = dofs + inc
        return (dofs,inc,it+1)
    
    if init == None:
        dofs = np.zeros(eqn.ndof)
    else:
        dofs = init
    it = 0
    inc = np.ones_like(dofs)
    dofs,inc,it = jax.lax.while_loop(cond_fun, body_fun, (dofs,inc,it))

    
    return dofs