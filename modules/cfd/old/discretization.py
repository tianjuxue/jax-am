import jax.numpy as np


# calculate ghost_cell values
def ghost_cell(BC, value, dx=None, neighbor=None):
    if BC == 'Dirchlet' or BC == 0:
        return 2*value - neighbor
    if BC == 'Neumann' or BC == 1:
        return neighbor + dx*value
    if BC == 'Marangoni_Z' or BC == 2:
        return np.concatenate( ((neighbor[:,:,:,[0]] + dx*value[:,:,:,[0]]),
                                (neighbor[:,:,:,[1]] + dx*value[:,:,:,[1]]),
                                (2*value[:,:,:,[2]] - neighbor[:,:,:,[2]])),axis=-1)

    
def get_GC_values(f,BCs,dX):
    return [ghost_cell(BCs[0][0],BCs[1][0],-dX[0],f[[0],:,:]),
            ghost_cell(BCs[0][1],BCs[1][1],dX[0],f[[-1],:,:]),
            ghost_cell(BCs[0][2],BCs[1][2],-dX[1],f[:,[0],:]),
            ghost_cell(BCs[0][3],BCs[1][3],dX[1],f[:,[-1],:]),
            ghost_cell(BCs[0][4],BCs[1][4],-dX[2],f[:,:,[0]]),
            ghost_cell(BCs[0][5],BCs[1][5],dX[2],f[:,:,[-1]])]


# calculate surface values
def get_surf_values(f,BCs,dX,face):
    if face == 0 or face == '-x':
        return surf_value(BCs[0][0],BCs[1][0],-dX[0],f[[0],:,:])
    if face == 1 or face == '+x':
        return surf_value(BCs[0][1],BCs[1][1], dX[0],f[[1],:,:])
    if face == 2 or face == '-y':
        return surf_value(BCs[0][2],BCs[1][2],-dX[1],f[:,[0],:])
    if face == 3 or face == '+y':
        return surf_value(BCs[0][3],BCs[1][3],dX[1],f[:,[-1],:])
    if face == 4 or face == '-z':
        return surf_value(BCs[0][4],BCs[1][4],-dX[2],f[:,:,[0]])
    if face == 5 or face == '+z':
        return surf_value(BCs[0][5],BCs[1][5],dX[2],f[:,:,[-1]])
    
    
def surf_value(BC, value, dx=None, neighbor=None):
    if BC == 'Dirchlet' or BC == 0:
        return value * np.ones_like(neighbor)
    if BC == 'Neumann' or BC == 1:
        return neighbor + dx/2.*value
    if BC == 'Marangoni_Z' or BC == 2:
        return np.concatenate( ((neighbor[:,:,:,[0]] + dx/2.*value[:,:,:,[0]]),
                                (neighbor[:,:,:,[1]] + dx/2.*value[:,:,:,[1]]),
                                (value[:,:,:,[2]])),axis=-1 )


# get face values from cell values    
def cell_to_face(f,BCs,dX):
    f_gc = get_GC_values(f,BCs,dX)
    surf_x = np.concatenate(((f_gc[0]+f[[0],:,:])/2.,(f[1:,:,:]+f[:-1,:,:])/2.,(f_gc[1]+f[[-1],:,:])/2.),axis=0)
    surf_y = np.concatenate(((f_gc[2]+f[:,[0],:])/2.,(f[:,1:,:]+f[:,:-1,:])/2.,(f_gc[3]+f[:,[-1],:])/2.),axis=1)
    surf_z = np.concatenate(((f_gc[4]+f[:,:,[0]])/2.,(f[:,:,1:]+f[:,:,:-1])/2.,(f_gc[5]+f[:,:,[-1]])/2.),axis=2)
    return [surf_x,surf_y,surf_z]


# 3D diffusion: f - cell arrays, f_gc - ghostcell values 
def laplace(f,f_gc,miu,dX):
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


# 3D convection: vel_f - face velocities
def div(f,f_gc,vel_f,dX,theta = 0.125):
    u_f,v_f,w_f = vel_f
    fx = np.concatenate((f_gc[0],f_gc[0],f,f_gc[1],f_gc[1]),axis=0)
    fy = np.concatenate((f_gc[2],f_gc[2],f,f_gc[3],f_gc[3]),axis=1)
    fz = np.concatenate((f_gc[4],f_gc[4],f,f_gc[5],f_gc[5]),axis=2)
    
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


## gradient
def gradient(f,f_gc,dX):
    return dX[0]*dX[1]*dX[2]*np.concatenate(((np.diff(f,axis=0,append=f_gc[1]) + np.diff(f,axis=0,prepend=f_gc[0]))/dX[0]/2.,
                           (np.diff(f,axis=1,append=f_gc[3]) + np.diff(f,axis=1,prepend=f_gc[2]))/dX[1]/2.,
                           (np.diff(f,axis=2,append=f_gc[5]) + np.diff(f,axis=2,prepend=f_gc[4]))/dX[2]/2.),axis=-1)


# compute face velocity from cell values
##### modificaiton requried: rhie chow interpolation
def get_face_vels(vel,BCs,dX):
    vel_gc = get_GC_values(vel,BCs,dX)
                                                     
    u = np.concatenate((vel_gc[0][:,:,:,[0]],
                    vel[:,:,:,[0]],
                    vel_gc[1][:,:,:,[0]]),axis=0)
    u_f = (u[1:,:,:]+u[:-1,:,:])/2
    
    v = np.concatenate((vel_gc[2][:,:,:,[1]],
                        vel[:,:,:,[1]],
                        vel_gc[3][:,:,:,[1]]),axis=1)
    v_f = (v[:,1:,:]+v[:,:-1,:])/2
    
    w = np.concatenate((vel_gc[4][:,:,:,[2]],
                        vel[:,:,:,[2]],
                        vel_gc[5][:,:,:,[2]]),axis=2)
    w_f = (w[:,:,1:]+w[:,:,:-1])/2
    return u_f,v_f,w_f


