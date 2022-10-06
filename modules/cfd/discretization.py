import jax.numpy as np


def BC_surf(BC, value, dx=None, neighbor=None):
    if BC == 0:
        return 2*value - neighbor
    if BC == 1:
        return neighbor + dx*value
    if BC == 2:
        return np.concatenate( ((neighbor[:,:,:,[0]] + dx*value[:,:,:,[0]]),
                                (neighbor[:,:,:,[1]] + dx*value[:,:,:,[1]]),
                                (2*value[:,:,:,[2]] - neighbor[:,:,:,[2]])),axis=-1 )

    
def get_bound_values(f,dX,BCs,values):
    return [BC_surf(BCs[0],values[0],-dX[0],f[[1],:,:]),
            BC_surf(BCs[1],values[1],dX[0],f[[-2],:,:]),
            BC_surf(BCs[2],values[2],-dX[1],f[:,[1],:]),
            BC_surf(BCs[3],values[3],dX[1],f[:,[-2],:]),
            BC_surf(BCs[4],values[4],-dX[2],f[:,:,[1]]),
            BC_surf(BCs[5],values[5],dX[2],f[:,:,[-2]])]


# 3D diffusion
def diffusion(f,f_b,dX):
    dif = ((np.diff(f,axis=0,append=f_b[1]) - np.diff(f,axis=0,prepend=f_b[0]))/dX[0]*dX[1]*dX[2] +
           (np.diff(f,axis=1,append=f_b[3]) - np.diff(f,axis=1,prepend=f_b[2]))/dX[1]*dX[2]*dX[0] +
           (np.diff(f,axis=2,append=f_b[5]) - np.diff(f,axis=2,prepend=f_b[4]))/dX[2]*dX[0]*dX[1])
    return dif


# 3D convection
def convection(f,f_b,vel_f,dX,theta = 0.125):
    u_f,v_f,w_f = vel_f
    fx = np.concatenate((f_b[0],f_b[0],f,f_b[1],f_b[1]),axis=0)
    fy = np.concatenate((f_b[2],f_b[2],f,f_b[3],f_b[3]),axis=1)
    fz = np.concatenate((f_b[4],f_b[4],f,f_b[5],f_b[5]),axis=2)
    
    f_face_x = (theta*(fx[1:-2,:,:]/2+fx[2:-1,:,:]/2) + (1-theta)*(3*fx[1:-2,:,:]/2-fx[:-3,:,:]/2)*(u_f>=0) 
                                                      + (1-theta)*(3*fx[2:-1,:,:]/2-fx[3:,:,:]/2)*(u_f<0)) 
    f_face_y = (theta*(fy[:,1:-2,:]/2+fy[:,2:-1,:]/2) + (1-theta)*(3*fy[:,1:-2,:]/2-fy[:,:-3,:]/2)*(v_f>=0) 
                                                      + (1-theta)*(3*fy[:,2:-1,:]/2-fy[:,3:,:]/2)*(v_f<0))
    f_face_z = (theta*(fz[:,:,1:-2]/2+fz[:,:,2:-1]/2) + (1-theta)*(3*fz[:,:,1:-2]/2-fz[:,:,:-3]/2)*(w_f>=0) 
                                                      + (1-theta)*(3*fz[:,:,2:-1]/2-fz[:,:,3:]/2)*(w_f<0))
    flux = ((f_face_x[1:,:,:]*u_f[1:,:,:] - f_face_x[:-1,:,:]*u_f[:-1,:,:])*dX[1]*dX[2] + 
            (f_face_y[:,1:,:]*v_f[:,1:,:] - f_face_y[:,:-1,:]*v_f[:,:-1,:])*dX[2]*dX[0] +
            (f_face_z[:,:,1:]*w_f[:,:,1:] - f_face_z[:,:,:-1]*w_f[:,:,:-1])*dX[0]*dX[1])
    return flux


# compute face velocity from cell values
##### modificaiton requried: rhie chow interpolation
def get_face_values(vel,dX,BCs,values):
    vel_bc = get_bound_values(vel,dX,BCs,values)
                                                     
    u = np.concatenate((vel_bc[0][:,:,:,[0]],
                    vel[:,:,:,[0]],
                    vel_bc[1][:,:,:,[0]]),axis=0)
    u_f = (u[1:,:,:]+u[:-1,:,:])/2
    
    v = np.concatenate((vel_bc[2][:,:,:,[1]],
                        vel[:,:,:,[1]],
                        vel_bc[3][:,:,:,[1]]),axis=1)
    v_f = (v[:,1:,:]+v[:,:-1,:])/2
    
    w = np.concatenate((vel_bc[4][:,:,:,[2]],
                        vel[:,:,:,[2]],
                        vel_bc[5][:,:,:,[2]]),axis=2)
    w_f = (w[:,:,1:]+w[:,:,:-1])/2
    return u_f,v_f,w_f


## gradient
def gradient(p,dX,relative=False):
    p_b = np.pad(p,((1,1),(1,1),(1,1),(0,0)),mode='symmetric')
    if relative:
        p_b = p_b.at[0,0,0].set(0)
    return np.concatenate(((p_b[2:,1:-1,1:-1]-p_b[:-2,1:-1,1:-1])/(dX[0]*2),
                           (p_b[1:-1,2:,1:-1]-p_b[1:-1,:-2,1:-1])/(dX[1]*2),
                           (p_b[1:-1,1:-1,2:]-p_b[1:-1,1:-1,:-2])/(dX[2]*2)),axis=-1)