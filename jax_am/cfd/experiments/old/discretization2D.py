import jax.numpy as np

### 2D fvm discretization
def BC_surf(BC, value, dx=None, neighbor=None):
    if BC == 0:
        return 2*value - neighbor
    if BC == 1:
        return neighbor + dx*value

    
def get_bound_values(f,BCs,dX):
    return [BC_surf(BCs[0][0],BCs[1][0],-dX[0],f[[0],:]),
            BC_surf(BCs[0][1],BCs[1][1],dX[0],f[[-1],:]),
            BC_surf(BCs[0][2],BCs[1][2],-dX[1],f[:,[0]]),
            BC_surf(BCs[0][3],BCs[1][3],dX[1],f[:,[-1]])]


# 2D diffusion: central difference
def diffusion(f,f_b,miu,dX):
    dif = ((np.diff(f,axis=0,append=f_b[1]) - np.diff(f,axis=0,prepend=f_b[0]))/dX[0]*dX[1] +
           (np.diff(f,axis=1,append=f_b[3]) - np.diff(f,axis=1,prepend=f_b[2]))/dX[1]*dX[0])*miu
    return dif


# 2D convection: QUICK: 3rd order upwind
def convection(f,f_b,vel_f,dX,theta = 0.25):
    u_f,v_f = vel_f
    fx = np.concatenate((f_b[0],
                    f_b[0],
                    f,
                    f_b[1],
                    f_b[1]),axis=0)
    fy = np.concatenate((f_b[2],
                        f_b[2],
                        f,
                        f_b[3],
                        f_b[3]),axis=1)
    
    f_face_x = (theta*(fx[1:-2,:]/2+fx[2:-1,:]/2) + (1-theta)*(3*fx[1:-2,:]/2-fx[:-3,:]/2)*(u_f>=0) 
                                                  + (1-theta)*(3*fx[2:-1,:]/2-fx[3:,:]/2)*(u_f<0)) 
    f_face_y = (theta*(fy[:,1:-2]/2+fy[:,2:-1]/2) + (1-theta)*(3*fy[:,1:-2]/2-fy[:,:-3]/2)*(v_f>=0) 
                                                  + (1-theta)*(3*fy[:,2:-1]/2-fy[:,3:]/2)*(v_f<0))
    flux = ((f_face_x[1:,:,:]*u_f[1:,:,:] - f_face_x[:-1,:,:]*u_f[:-1,:,:])*dX[1] + 
            (f_face_y[:,1:,:]*v_f[:,1:,:] - f_face_y[:,:-1,:]*v_f[:,:-1,:])*dX[0] )
    return flux


# compute face velocity from cell values
##### modificaiton requried: rhie chow interpolation
def get_face_values(vel,BC_info,dX):
    vel_bc = get_bound_values(vel,BC_info,dX)
    u = np.concatenate((vel_bc[0][:,:,[0]],
                    vel[:,:,[0]],
                    vel_bc[1][:,:,[0]]),axis=0)
    u_f = (u[1:,:]+u[:-1,:])/2
    v = np.concatenate((vel_bc[2][:,:,[1]],
                        vel[:,:,[1]],
                        vel_bc[3][:,:,[1]]),axis=1)
    v_f = (v[:,1:]+v[:,:-1])/2
    return u_f,v_f


# 2D gradient: central difference
def gradient(f,f_b,dX):
    return np.concatenate(((np.diff(f,axis=0,append=f_b[1]) + np.diff(f,axis=0,prepend=f_b[0]))/dX[0]/2.,
                (np.diff(f,axis=1,append=f_b[3]) + np.diff(f,axis=1,prepend=f_b[2]))/dX[1]/2.),axis=-1)


# # assuming zero gradient at the boundary
# def gradient(p):
#     p_b = np.pad(p,((1,1),(1,1),(0,0)),mode='symmetric')
#     p_b = p_b.at[0,0,0].set(0)
# #     p_b = np.pad(p,((1,1),(1,1),(0,0)),mode='constant',constant_values=0)
#     return np.concatenate(((p_b[2:,1:-1]-p_b[:-2,1:-1])/(dX[0]*2),
#                 (p_b[1:-1,2:]-p_b[1:-1,:-2])/(dX[1]*2)),axis=-1)