import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
import os
import glob
import matplotlib.pyplot as plt
from functools import partial

from jax_am.cfd.gamma import update_T
from jax_am.common import box_mesh


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

crt_file_path = os.path.dirname(__file__)
input_dir = os.path.join(crt_file_path, 'input')     
output_dir = os.path.join(crt_file_path, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
numpy_dir = os.path.join(output_dir, 'numpy')
os.makedirs(vtk_dir, exist_ok=True)
os.makedirs(numpy_dir, exist_ok=True)


def clean_files(format_dir):
    files = glob.glob(format_dir + f"/*")
    for f in files:
        os.remove(f)


@jax.jit
def interp_data(T, rot, xl, yl, dx, dy):
    surface_T = (T[1:-1,1:-1,-1]+T[1:-1,1:-1,-2])/2
    gridx_ = np.arange(-1e-3,0.28e-3,1e-5)
    gridy_ = np.arange(-0.32e-3,0.32e-3,1e-5)
    gridx,gridy = np.meshgrid(gridx_, gridy_,indexing='ij')
    X_rot = (gridx*rot[0]-gridy*rot[1]+xl-dx/2)/dx
    Y_rot = (gridx*rot[1]+gridy*rot[0]+yl-dx/2)/dy
    melt_2d = map_coordinates(surface_T,(X_rot,Y_rot),1)
    X_3d = jnp.repeat(X_rot[:,:,None],30,axis=2)
    Y_3d = jnp.repeat(Y_rot[:,:,None],30,axis=2)
    Z_3d = np.arange(0,30)[None,None,:].repeat(64,axis=1).repeat(128,axis=0)
    melt_3d = map_coordinates(T[1:-1,1:-1,-31:-1],(X_3d,Y_3d,Z_3d),1)
    return melt_2d, melt_3d


def simulation():
    # clean_files(vtk_dir)
    # clean_files(numpy_dir)

    nx = 500
    ny = 500
    nz = 100

    dx = 10e-6
    dy = 10e-6
    dz = 10e-6
    dt = 2e-6

    x_ = np.linspace(dx/2., dx/2.+dx*(nx-1), nx)
    y_ = np.linspace(dy/2., dy/2.+dy*(ny-1), ny)
    z_ = np.linspace(dz/2., dz/2.+dz*(nz-1), nz)

    x, y, z = jnp.meshgrid(x_, y_, z_, indexing='ij')

    gamma_args = {}

    # laser parameter
    gamma_args['eta'] = 0.43
    gamma_args['r'] = 5e-5
    gamma_args['rho'] = 8440
    gamma_args['h'] = 10
    gamma_args['eps'] = 0.4
    gamma_args['SB'] = 5.67e-8
    gamma_args['T0'] = 298.
    gamma_args['Ts'] = 1563.
    gamma_args['Tl'] = 1623.
    gamma_args['L'] = 290e3/(gamma_args['Tl']-gamma_args['Ts'])

    u = jnp.zeros((nx+1, ny+2, nz+2))
    v = jnp.zeros((nx+2, ny+1, nz+2))
    w = jnp.zeros((nx+2, ny+2, nz+1))
    p = jnp.zeros((nx+2, ny+2, nz+2))
    T = jnp.ones((nx+2, ny+2, nz+2), dtype=float)*gamma_args['T0']

    toolpath = np.loadtxt(os.path.join(input_dir, 'toolpath.txt'))
    t = np.arange(0, toolpath[-1, 0], dt) + dt/2
    direction = np.copy(toolpath[:,0:3])
    direction[0:,1:3] = np.array([0.,0.])
    direction[1:,1:3] = toolpath[1:,1:3] - toolpath[0:-1,1:3]
    direction[1:,1] = direction[1:,1]/np.sqrt(direction[1:,1]**2 + direction[1:,2]**2)
    direction[1:,2] = direction[1:,2]/np.sqrt(direction[1:,1]**2 + direction[1:,2]**2)
    pos_interp = interp1d(toolpath[:,0].T, toolpath[:,1:3].T, kind='linear')

    power = np.load(os.path.join(input_dir, f'power_0.npy'))
    
    print(f"power.shape = {power.shape}")
    print(f"t.shape = {t.shape}")
    print(f"direction.shape = {direction.shape}")
           
    P_interp = interp1d(power[:,0], power[:,1], kind='linear')
    dir_interp = interp1d(direction[:,0].T, direction[:,1:3].T, kind='next')
    pos = pos_interp(t)
    P = P_interp(t)
    D = dir_interp(t)

    meshio_mesh = box_mesh(nx, ny, nz, nx*dx, ny*dy, nz*dz)

    num = 0
    for i in range(0, t.shape[0]):
        print(f"Step {i} in {t.shape[0]}")
        T = update_T(T, pos[0, i], pos[1, i], x, y, P[i], dx, dy, dz, dt, gamma_args)
        if (i-24) % 50 == 0:
            print(f"t = {t[i]}, xl = {pos[0, i]}, yl = {pos[1, i]}, save melt_2d to local file.")
            melt_2d, melt_3d = interp_data(T-gamma_args['T0'], D[:,i], pos[0, i], pos[1, i], dx, dy)
            np.save(os.path.join(numpy_dir, f'melt_2d_step_{num:05d}.npy'), melt_2d)
            
            # Very expensive for writing the entire mesh to local
            meshio_mesh.cell_data['T'] = [np.array(T[1:-1, 1:-1, 1:-1], dtype=np.float32)]
            meshio_mesh.write(os.path.join(vtk_dir, f'u_{num:05d}.vtu'))
            
            num += 1


def visualization():
    melt_2d = np.load(os.path.join(numpy_dir, f'melt_2d_step_00001.npy'))
    fig = plt.figure()
    plt.imshow(melt_2d.T, vmax=1650, vmin=1000, cmap='hot', origin='lower')
    plt.colorbar()
    plt.show()


if __name__=="__main__":
    simulation()
    # visualization()
