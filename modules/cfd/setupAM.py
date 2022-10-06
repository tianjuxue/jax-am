import jax.numpy as np
import numpy as onp
from discretization import *

def generate_mesh(domain,N):
## 3D box mesh    
    domain_x,domain_y,domain_z = domain
    Nx,Ny,Nz = N
    
    cell_num = Nx*Ny*Nz
    num_nodes = (Nx + 1) * (Ny + 1) * (Nz + 1)
    x = onp.linspace(0., domain_x, Nx + 1)
    y = onp.linspace(0., domain_y, Ny + 1)
    z = onp.linspace(0., domain_z, Nz + 1)

    xx, yy, zz = onp.meshgrid(x, y, z, indexing='ij')
#     points = onp.vstack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T

    xc = np.array((xx[1:, 1:, 1:] + xx[:-1, 1:, 1:])/2)[:, :, :, None]
    yc = np.array((yy[1:, 1:, 1:] + yy[1:, :-1, 1:])/2)[:, :, :, None]
    zc = np.array((zz[1:, 1:, 1:] + zz[1:, 1:, :-1])/2)[:, :, :, None]

    dX = np.array([domain_x/Nx, domain_y/Ny, domain_z/Nz])
    
    return dX,xc,yc,zc


def update_cond(temp_dependent_cond,T,BCs,dX):
    T_bc = get_bound_values(T,BCs,dX)
    cond_surf_x = np.concatenate((temp_dependent_cond((T_bc[0]+T[[0],:,:])/2.),
                                  temp_dependent_cond((T[1:,:,:] + T[:-1,:,:])/2.),
                                  temp_dependent_cond((T_bc[1]+T[[-1],:,:])/2.)),axis=0)
    
    cond_surf_y = np.concatenate((temp_dependent_cond((T_bc[2]+T[:,[0],:])/2.),
                                  temp_dependent_cond((T[:,1:,:] + T[:,:-1,:])/2.),
                                  temp_dependent_cond((T_bc[3]+T[:,[-1],:])/2.)),axis=1)
        
    cond_surf_z = np.concatenate((temp_dependent_cond((T_bc[4]+T[:,:,[0]])/2.),
                                  temp_dependent_cond((T[:,:,1:] + T[:,:,:-1])/2.),
                                  temp_dependent_cond((T_bc[5]+T[:,:,[-1]])/2.)),axis=2)
    return [cond_surf_x,cond_surf_y,cond_surf_z]