import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

# jax.config.update("jax_enable_x64", True)


@jax.jit
def update_mat(T, gamma_args):
    # fraction of fluid: fl = 1(liquid), 0(solid), 0~1 (mushy zone)
    fl = (jnp.clip(T, gamma_args['Ts'], gamma_args['Tl']) - gamma_args['Ts'])/(gamma_args['Tl'] - gamma_args['Ts'])
    # update material properties
    k = 0.0163*jnp.clip(T,300,gamma_args['Tl']) + 4.5847
    cp = 0.2441*jnp.clip(T,300,gamma_args['Tl']) + 338.39 + (fl>0)*(fl<1)*gamma_args['L']
    return fl, k, cp


@jax.jit
def BC_thermal(T, k, xl, yl, x, y, P, dx, dy, dz, gamma_args):
    """Robin boundary condition
    """
    T_x_neg = (T[ 1,1:-1,1:-1] + T[ 0,1:-1,1:-1])/2.
    q_x_neg = (-gamma_args['h']*(T_x_neg-gamma_args['T0']) - gamma_args['SB']*gamma_args['eps']*(T_x_neg**4-gamma_args['T0']**4))
    T_x_pos = (T[-2,1:-1,1:-1] + T[-1,1:-1,1:-1])/2
    q_x_pos =  (gamma_args['h']*(T_x_pos-gamma_args['T0']) + gamma_args['SB']*gamma_args['eps']*(T_x_pos**4-gamma_args['T0']**4))
    T_y_neg = (T[1:-1, 1,1:-1] + T[1:-1, 0,1:-1])/2
    q_y_neg = (-gamma_args['h']*(T_y_neg-gamma_args['T0']) - gamma_args['SB']*gamma_args['eps']*(T_y_neg**4-gamma_args['T0']**4))
    T_y_pos = (T[1:-1,-2,1:-1] + T[1:-1,-1,1:-1])/2
    q_y_pos =  (gamma_args['h']*(T_y_pos-gamma_args['T0']) + gamma_args['SB']*gamma_args['eps']*(T_y_pos**4-gamma_args['T0']**4))
    T_z_neg = (T[1:-1,1:-1, 1] + T[1:-1,1:-1, 0])/2
    q_z_neg = (-gamma_args['h']*(T_z_neg-gamma_args['T0']) - gamma_args['SB']*gamma_args['eps']*(T_z_neg**4-gamma_args['T0']**4))
    T_z_pos = (T[1:-1,1:-1,-2] + T[1:-1,1:-1,-1])/2
    q_z_pos = ( gamma_args['h']*(T_z_pos-gamma_args['T0']) + gamma_args['SB']*gamma_args['eps']*(T_z_pos**4-gamma_args['T0']**4) \
               - 2*P*gamma_args['eta']/jnp.pi/gamma_args['r']**2*jnp.exp(-2*((x[:,:,-1]-xl)**2+(y[:,:,-1]-yl)**2)/gamma_args['r']**2))
    # -k*gradT = q 
    T = T.at[ 0,1:-1,1:-1].set(T[ 1,1:-1,1:-1]+q_x_neg/k[ 1,1:-1,1:-1]*dx)
    T = T.at[-1,1:-1,1:-1].set(T[-2,1:-1,1:-1]-q_x_pos/k[-2,1:-1,1:-1]*dx)
    T = T.at[1:-1, 0,1:-1].set(T[1:-1, 1,1:-1]+q_y_neg/k[1:-1, 1,1:-1]*dy)
    T = T.at[1:-1,-1,1:-1].set(T[1:-1,-2,1:-1]-q_y_pos/k[1:-1,-2,1:-1]*dy)
    T = T.at[1:-1,1:-1, 0].set(T[1:-1,1:-1, 1]+q_z_neg/k[1:-1,1:-1, 1]*dz)
    T = T.at[1:-1,1:-1,-1].set(T[1:-1,1:-1,-2]-q_z_pos/k[1:-1,1:-1,-2]*dz)
    return T


@jax.jit
def laplace(f, dx, dy, dz):
    # calculate f_xx, f_yy
    f_xx = (f[0:-2,:,:]-2*f[1:-1,:,:]+f[2:,:,:])/dx**2
    f_yy = (f[:,0:-2,:]-2*f[:,1:-1,:]+f[:,2:,:])/dy**2
    f_zz = (f[:,:,0:-2]-2*f[:,:,1:-1]+f[:,:,2:])/dz**2
    return f_xx[:,1:-1,1:-1],f_yy[1:-1,:,1:-1],f_zz[1:-1,1:-1,:]


@jax.jit
def update_T(T, xl, yl, x, y, P, dx, dy, dz, dt, gamma_args):
    fl,k,cp = update_mat(T, gamma_args)
    T = BC_thermal(T, k, xl, yl, x, y, P, dx, dy, dz, gamma_args)
    T_xx,T_yy,T_zz = laplace(T, dx, dy, dz)
    T = T.at[1:-1,1:-1,1:-1].add(dt*(k[1:-1,1:-1,1:-1]/gamma_args['rho']/cp[1:-1,1:-1,1:-1]*(T_xx+T_yy+T_zz)))
    return T
