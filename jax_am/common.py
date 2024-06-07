import jax
import numpy as onp
import os
import meshio
import json
import yaml
import pandas as pd 
import time
from functools import wraps
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from jax_am import logger


def json_parse(json_filepath):
    with open(json_filepath) as f:
        args = json.load(f)
    json_formatted_str = json.dumps(args, indent=4)
    print(json_formatted_str)
    return args


def yaml_parse(yaml_filepath):
    with open(yaml_filepath) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print("YAML parameters:")
        # TODO: These are just default parameters
        print(yaml.dump(args, default_flow_style=False))
        print("These are default parameters")
    return args


def box_mesh(Nx, Ny, Nz, domain_x, domain_y, domain_z):
    dim = 3
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    z = onp.linspace(0, domain_z, Nz + 1)
    xv, yv, zv = onp.meshgrid(x, y, z, indexing='ij')
    points_xyz = onp.stack((xv, yv, zv), axis=dim)
    points = points_xyz.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xyz = points_inds.reshape(Nx + 1, Ny + 1, Nz + 1)
    inds1 = points_inds_xyz[:-1, :-1, :-1]
    inds2 = points_inds_xyz[1:, :-1, :-1]
    inds3 = points_inds_xyz[1:, 1:, :-1]
    inds4 = points_inds_xyz[:-1, 1:, :-1]
    inds5 = points_inds_xyz[:-1, :-1, 1:]
    inds6 = points_inds_xyz[1:, :-1, 1:]
    inds7 = points_inds_xyz[1:, 1:, 1:]
    inds8 = points_inds_xyz[:-1, 1:, 1:]
    cells = onp.stack((inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8),
                      axis=dim).reshape(-1, 8)
    out_mesh = meshio.Mesh(points=points, cells={'hexahedron': cells})
    return out_mesh


def rectangle_mesh(Nx, Ny, domain_x, domain_y):
    dim = 2
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    xv, yv = onp.meshgrid(x, y, indexing='ij')
    points_xy = onp.stack((xv, yv), axis=dim)
    points = points_xy.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xy = points_inds.reshape(Nx + 1, Ny + 1)
    inds1 = points_inds_xy[:-1, :-1]
    inds2 = points_inds_xy[1:, :-1]
    inds3 = points_inds_xy[1:, 1:]
    inds4 = points_inds_xy[:-1, 1:]
    cells = onp.stack((inds1, inds2, inds3, inds4), axis=dim).reshape(-1, 4)
    out_mesh = meshio.Mesh(points=points, cells={'quad': cells})
    return out_mesh


def make_video(data_dir):
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is
    # enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following
    # "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite

    # TODO
    os.system(
        f'ffmpeg -y -framerate 10 -i {data_dir}/png/tmp/u.%04d.png -pix_fmt yuv420p -vf \
               "crop=trunc(iw/2)*2:trunc(ih/2)*2" {data_dir}/mp4/test.mp4') # noqa
    
    
def to_data_frame(data_dir, dt=2 * 1e-6):
    mesh_dir = os.path.join(data_dir, 'msh', 'box.msh')
    vtk_dir = os.path.join(data_dir, 'vtk')
    all_data = []
    mesh = meshio.read(mesh_dir)
    mesh_df = pd.DataFrame(mesh.points, columns=["x", "y", "z"])
    vtu_files = [f for f in os.listdir(vtk_dir) if f.endswith('.vtu')]
    
    # Assuming the first file is always available and correctly formatted
    stepsave = int(vtu_files[1][-9:-4])
    print(f"Step save: {stepsave}")
    
    i = 0
    
    for vtu_file in vtu_files:
        time_value = round(i * stepsave * dt, 8)
        time = pd.Series(onp.full(len(mesh_df), time_value))
        vtu_path = os.path.join(vtk_dir, vtu_file)
        vtu_data = meshio.read(vtu_path)
        data_dict = pd.concat([mesh_df, time.rename('time')], axis=1)
        
        if vtu_data.point_data:
            for name, data in vtu_data.point_data.items():
                # Handle the case where data has multiple components (e.g., a vector field)
                if data.ndim == 1:
                    # Data has a single component
                    data_dict[name] = data
                else:
                    # Flatten multi-dimensional data into separate columns
                    for idx in onp.ndindex(data.shape[1:]):
                        flattened_name = f"{name}_{'_'.join(map(str, idx))}"
                        data_dict[flattened_name] = data[(slice(None),) + idx]
                i += 1
                
            all_data.append(pd.DataFrame(data_dict))
    
    DF = pd.concat(all_data, ignore_index=True)
    DF.to_csv("data_frame.csv", index=False)
    print('save as data_frame.csv')
    return DF
    
def make_scatter(data_csv_dir,solname = 'sol_0', format = 'gif'):
    data_frame = pd.read_csv(data_csv_dir)
    X_unique = pd.unique(data_frame['x'])
    Y_unique = pd.unique(data_frame['y'])
    Z_unique = pd.unique(data_frame['z'])
    time_unique = pd.unique(data_frame['time'])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter([], [], [], c=[], cmap='jet', vmin=data_frame[solname].min(), vmax=data_frame[solname].max())
    cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.5)
    
    def init():
        ax.set_xlim(X_unique.min(), X_unique.max())
        ax.set_ylim(Y_unique.min(), Y_unique.max())
        ax.set_zlim(Z_unique.min(), Z_unique.max())
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return sc,

    def update(time_value):
        frame_data = data_frame[data_frame['time'] == time_value]
        ax.clear()
        ax.set_xlim(X_unique.min(), X_unique.max())
        ax.set_ylim(Y_unique.min(), Y_unique.max())
        ax.set_zlim(Z_unique.min(), Z_unique.max())
        sc = ax.scatter(frame_data['x'], frame_data['y'], frame_data['z'], c=frame_data[solname], cmap='jet', vmin=0, vmax=5000)
        ax.set_title(f"Temperature at t = {time_value:.5f} S")
        return sc,

    anim = FuncAnimation(fig, update, frames=time_unique, init_func=init, blit=False)

    anim.save(f'scatter_sol.{format}', writer='pillow', fps=10)

    plt.show()    

# A simpler decorator for printing the timing results of a function
def timeit(func):

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


# Wrapper for writing timing results to a file
def walltime(txt_dir=None, filename=None):

    def decorate(func):

        def wrapper(*list_args, **keyword_args):
            start_time = time.time()
            return_values = func(*list_args, **keyword_args)
            end_time = time.time()
            time_elapsed = end_time - start_time
            platform = jax.lib.xla_bridge.get_backend().platform
            logger.info(
                f"Time elapsed {time_elapsed} of function {func.__name__} "
                f"on platform {platform}"
            )
            if txt_dir is not None:
                os.makedirs(txt_dir, exist_ok=True)
                fname = 'walltime'
                if filename is not None:
                    fname = filename
                with open(os.path.join(txt_dir, f"{fname}_{platform}.txt"),
                          'w') as f:
                    f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
            return return_values

        return wrapper

    return decorate
