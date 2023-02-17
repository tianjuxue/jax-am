# Direct copy from https://github.com/Maarten-vd-Sande/lbm

import numpy as np
from numba import jit
from collections import namedtuple
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# the 9 directions of the D2Q9 grid
v = np.array([[ 0,  0],
              [ 0, -1],
              [ 0,  1],
              [-1,  0],
              [-1, -1],
              [-1,  1],
              [ 1,  0],
              [ 1, -1],
              [ 1,  1]])

# the 9 corresponding velocities of the D2Q9 grid
t = np.array([4 / 9,
              1 / 9,
              1 / 9,
              1 / 9,
              1 / 36,
              1 / 36,
              1 / 9,
              1 / 36,
              1 / 36])

# returns the index of the opposite direction of v
v_inv = [v.tolist().index((-v[i]).tolist()) for i in range(9)]


class CT_Enum:  # (enum.IntFlag)
    """
    Class that keeps track of our cell types by bitflags.
    Numba does not support IntFlag (yet) so temporary solution: convert to namedtuple
    """
    FLUID =          2 ** 0   # 1
    INTERFACE =      2 ** 1   # 2
    GAS =            2 ** 2   # 4
    OBSTACLE =       2 ** 3   # 8
    INLET =          2 ** 4   # 16
    OUTLET =         2 ** 5   # 32

    NO_FLUID_NEIGH = 2 ** 6   # 64
    NO_EMPTY_NEIGH = 2 ** 7   # 128
    NO_IFACE_NEIGH = 2 ** 8   # 256

    TO_FLUID =       2 ** 9   # 512
    TO_GAS =         2 ** 10  # 1024


def class_to_namedtuple(cls):
    """
    Hack to force 'enum.IntFlag'
    """
    newdict = dict((k, getattr(cls, k)) for k in dir(cls) if not k.startswith('_'))
    return namedtuple(cls.__name__, sorted(newdict, key=lambda k: newdict[k]))(**newdict)


CT = class_to_namedtuple(CT_Enum)



def init(nx, ny):
    """
    initialize all arrays (empty)
    """
    fin = np.zeros((9, nx, ny), dtype=np.float32)
    fout = np.zeros((9, nx, ny), dtype=np.float32)
    equi = np.zeros((9, nx, ny), dtype=np.float32)
    fdist = np.zeros((9, nx, ny), dtype=np.float32)
    inlet = np.zeros((2, nx, ny), dtype=np.float32)
    u = np.zeros((2, nx, ny), dtype=np.float32)
    rho = np.zeros((nx, ny), dtype=np.float32)
    mass = np.zeros((nx, ny), dtype=np.float32)
    cell_type = np.full((nx, ny), CT.GAS)

    return fin, fout, equi, fdist, inlet, u, rho, mass, cell_type


# @jit(nopython=True)
def lbm_collision(fin, fout, equi, inlet, u, rho, cell_type, omega, v, t, gravity):
    """
    Performs the collision step of the D2Q9 LBM, whilst keeping track of special cases such as inlets & outlets
    """
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):

            # skip empty cells
            if cell_type[ix, iy] & (CT.GAS | CT.OBSTACLE):
                continue

            # set outlet values
            elif cell_type[ix, iy] & CT.OUTLET:
                set_outlet(ix, iy, fin)
            # set inlet values
            elif cell_type[ix, iy] & CT.INLET:
                set_inlet(ix, iy, fin, rho, u, inlet)
            else:
                update_rho_u(ix, iy, fin, rho, u, v)

            # calculate the equilibrium state
            set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)

            # update the inlet speed
            if cell_type[ix, iy] & CT.INLET:
                for n in range(6, 9):
                    fin[n, ix, iy] = equi[n, ix, iy]

            # collision step
            collide(ix, iy, equi, fin, fout, omega, rho, v, t, gravity)


@jit(nopython=True)
def lbm_streaming(fin, fout, equi, u, rho, mass, cell_type, mass_prev, rho_prev, v, v_inv, t):
    """
    Performs the stream step of the D2Q9 LBM, whilst keeping track of special cases such as inlets & outlets
    """
    for ix in range(0, fin.shape[1]):
        for iy in range(0, fin.shape[2]):
            if cell_type[ix, iy] & (CT.OBSTACLE | CT.GAS):
                continue

            # fluid cell
            if cell_type[ix, iy] & (CT.FLUID | CT.INLET | CT.OUTLET):
                for n in range(9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    # skip out of bound for inlet and outlets
                    if 0 < ix_next >= fin.shape[1] or 0 < iy_next >= fin.shape[2]:
                        continue

                    if cell_type[ix_next, iy_next] & (CT.FLUID | CT.INTERFACE | CT.INLET | CT.OUTLET):
                        # streaming
                        fin[n, ix, iy] = fout[n, ix_next, iy_next]
                        # mass exchange
                        mass[ix, iy] += fout[n, ix_next, iy_next] - fout[v_inv[n], ix, iy]
                    else:  # obstacle
                        # stream back (no slip boundary)
                        fin[n, ix, iy] = fout[v_inv[n], ix, iy]

            # interface cell
            elif cell_type[ix, iy] & CT.INTERFACE:
                # get the fraction filled (epsilon)
                epsilon = get_epsilon(cell_type[ix, iy], rho_prev[ix, iy], mass_prev[ix, iy])
                set_equi(ix, iy, equi, 1.0, u[:, ix, iy], v, t)
                for n in range(1, 9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    if cell_type[ix_next, iy_next] & CT.FLUID:
                        # streaming
                        fin[n, ix, iy] = fout[n, ix_next, iy_next]
                        # mass exchange
                        mass[ix, iy] += fout[n, ix_next, iy_next] - fout[v_inv[n], ix, iy]
                    elif cell_type[ix_next, iy_next] & CT.INTERFACE:
                        # streaming
                        fin[n, ix, iy] = fout[n, ix_next, iy_next]
                        # mass exchange
                        epsilon_nei = get_epsilon(cell_type[ix_next, iy_next], rho_prev[ix_next, iy_next], mass_prev[ix_next, iy_next])
                        mass[ix, iy] += interface_mass_exchange(cell_type[ix, iy], cell_type[ix_next, iy_next], fout[v_inv[n], ix, iy], fout[n, ix_next, iy_next]) * \
                                        (epsilon + epsilon_nei) * 0.5
                    elif cell_type[ix_next, iy_next] & CT.GAS:
                        # streaming
                        fin[n, ix, iy] = equi[n, ix, iy] + equi[v_inv[n], ix, iy] - fout[v_inv[n], ix, iy]
                    else:  # obstacle
                        # streaming
                        fin[n, ix, iy] = fout[v_inv[n], ix, iy]

                # correct for surface normal
                normal = get_normal(ix, iy, cell_type, rho_prev, mass_prev)
                for n in range(1, 9):
                    if (normal[0] * v[v_inv[n], 0] + normal[1] * v[v_inv[n], 1]) > 0:
                        fin[n, ix, iy] = equi[n, ix, iy] + equi[v_inv[n], ix, iy] - fout[v_inv[n], ix, iy]

            # calculate density and velocity
            update_rho_u(ix, iy, fin, rho, u, v)
            if cell_type[ix, iy] & CT.FLUID:
                rho[ix, iy] = mass[ix, iy]


@jit()
def update_rho_u(ix, iy, fin, rho, u, v):
    """
    Set the density and velocity for the gridcell ix, iy
    """
    # set values to zeros
    rho[ix, iy] = 0
    u[0, ix, iy] = 0
    u[1, ix, iy] = 0

    for n in range(9):
        # calculate the density of this gridpoint
        rho[ix, iy] += fin[n, ix, iy]

        # calculate the velocity of this gridpoint
        u[0, ix, iy] += v[n, 0] * fin[n, ix, iy]
        u[1, ix, iy] += v[n, 1] * fin[n, ix, iy]

    # divide velocity by density
    u[:, ix, iy] /= rho[ix, iy]

    vel = (u[0, ix, iy] * u[0, ix, iy] + u[1, ix, iy] * u[1, ix, iy]) ** 0.5

    maxvel = (2/3) ** 0.5
    if vel > maxvel:
        u[:, ix, iy] *= maxvel / vel


@jit()
def set_equi(ix, iy, equi, rho, u, v, t):
    """
    Set the equilibrium value for the gridcell ix, iy
    """
    usqr = 3 / 2 * (u[0] ** 2 + u[1] ** 2)
    for n in range(9):
        vu = 3 * (v[n, 0] * u[0] + v[n, 1] * u[1])
        equi[n, ix, iy] = rho * t[n] * (1 + vu + 0.5 * vu * vu - usqr)

@jit()
def collide(ix, iy, equi, fin, fout, omega, rho, v, t, gravity):
    """
    Perform the collision step
    """
    for n in range(9):
        fout[n, ix, iy] = fin[n, ix, iy] + omega[0] * (equi[n, ix, iy] - fin[n, ix, iy])

        # add gravitational forces
        grav_temp = v[n, 0] * gravity[0] + \
                    v[n, 1] * gravity[1]
        fout[n, ix, iy] -= rho[ix, iy] * t[n] * grav_temp

@jit(nopython=True)
def get_epsilon(cell_type, rho, mass):
    """
    Calculate the fraction the cell is filled
    """
    if cell_type & CT.FLUID or cell_type & CT.OBSTACLE:
        return 1
    elif cell_type & CT.GAS:
        return 0
    else:
        if rho > 0:
            # clip
            epsilon = mass / rho

            if epsilon > 1:
                epsilon = 1
            elif epsilon < 0:
                epsilon = 0

            return epsilon
        return 0.5

@jit(nopython=True)
def get_normal(ix, iy, cell_type, rho, mass):
    x = 0.5 * (get_epsilon(cell_type[ix - 1, iy], rho[ix - 1, iy], mass[ix - 1, iy]) -
               get_epsilon(cell_type[ix + 1, iy], rho[ix + 1, iy], mass[ix + 1, iy]))
    y = 0.5 * (get_epsilon(cell_type[ix, iy - 1], rho[ix, iy - 1], mass[ix, iy - 1]) -
               get_epsilon(cell_type[ix, iy + 1], rho[ix, iy + 1], mass[ix, iy + 1]))
    return x, y


@jit()
def interface_mass_exchange(cell_type_self, cell_type_nei, fout_self, fout_nei):
    if cell_type_self & CT.NO_FLUID_NEIGH:
        if cell_type_nei & CT.NO_FLUID_NEIGH:
            return fout_nei - fout_self
        else:
            return -fout_self
    elif cell_type_self & CT.NO_EMPTY_NEIGH:
        if cell_type_nei & CT.NO_EMPTY_NEIGH:
            return fout_nei - fout_self
        else:
            return fout_nei
    else:
        if cell_type_nei & CT.NO_FLUID_NEIGH:
            return fout_nei
        elif cell_type_nei & CT.NO_EMPTY_NEIGH:
            return fout_self
        else:
            return fout_nei - fout_self




@jit(nopython=True)
def update_types(rho, cell_type, mass, v, fdist, u, equi, t, rho_prev, mass_prev, u_prev, cell_type_prev):
    """
    Update the cell flags, allowing for a free-surface
    """
    fill_offset = 0.003
    lonely_tresh = 0.1

    # set the to_fluid/gas flags
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.INTERFACE:

                if mass[ix, iy] > (1 + fill_offset) * rho[ix, iy] or \
                        (mass[ix, iy] >= (1 - lonely_tresh) * rho[ix, iy] and cell_type[ix, iy] & CT.NO_EMPTY_NEIGH):
                    cell_type[ix, iy] = CT.TO_FLUID

                elif mass[ix, iy] < -fill_offset * rho[ix, iy] or \
                        (mass[ix, iy] <= lonely_tresh * rho[ix, iy] and cell_type[ix, iy] & CT.NO_FLUID_NEIGH) or \
                        cell_type[ix, iy] & (CT.NO_IFACE_NEIGH | CT.NO_FLUID_NEIGH):
                    cell_type[ix, iy] = CT.TO_GAS

            # remove neighbourhood flags
            cell_type[ix, iy] &= ~(CT.NO_FLUID_NEIGH + CT.NO_EMPTY_NEIGH + CT.NO_IFACE_NEIGH)

    # interface -> fluid
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.TO_FLUID:
                for n in range(1, 9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    if cell_type[ix_next, iy_next] & CT.GAS:
                        cell_type[ix_next, iy_next] = CT.INTERFACE
                        average_surround(ix_next, iy_next, rho, rho_prev, mass, mass_prev, u, u_prev, v, cell_type_prev, equi, t)
                        cell_type[ix_next, iy_next] = CT.INTERFACE

    # interface-> gas
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.TO_GAS:
                for n in range(1, 9):
                    ix_next = ix - v[n, 0]
                    iy_next = iy - v[n, 1]

                    if cell_type[ix_next, iy_next] & CT.FLUID:
                        cell_type[ix_next, iy_next] = CT.INTERFACE

    # distribute excess mass
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.OBSTACLE:
                continue

            normal = get_normal(ix, iy, cell_type_prev, rho_prev, mass_prev)

            if cell_type[ix, iy] & CT.TO_FLUID:
                mex = mass[ix, iy] - rho[ix, iy]
                mass[ix, iy] = rho[ix, iy]

            elif cell_type[ix, iy] & CT.TO_GAS:
                mex = mass[ix, iy]
                normal = -normal[0], -normal[1]
                mass[ix, iy] = 0

            else:
                continue

            eta =  [0, 0, 0, 0, 0, 0, 0, 0, 0]
            isIF = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            eta_total = IF_total = 0

            for n in range(1, 9):
                ix_next = ix + v[n, 0]
                iy_next = iy + v[n, 1]

                if cell_type[ix_next, iy_next] & CT.INTERFACE:
                    eta[n] = v[n, 0] * normal[0] + v[n, 1] * normal[1]

                    if eta[n] < 0:
                        eta[n] = 0

                    eta_total += eta[n]
                    isIF[n] = 1
                    IF_total += 1

            if eta_total > 0:
                eta_frac = 1 / eta_total
                for n in range(1, 9):
                    fdist[n, ix, iy] = mex * eta[n] * eta_frac
            elif IF_total > 0:
                mex_rel = mex / IF_total
                for n in range(1, 9):
                    fdist[n, ix, iy] = mex_rel if isIF[n] else 0

    # collect distributed mass and finalize cell flags
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):

            if cell_type[ix, iy] & CT.INTERFACE:
                for n in range(1, 9):
                    mass[ix, iy] += fdist[n, ix + v[n, 0], iy + v[n, 1]]
            elif cell_type[ix, iy] & CT.TO_FLUID:
                cell_type[ix, iy] = CT.FLUID
            elif cell_type[ix, iy] & CT.TO_GAS:
                cell_type[ix, iy] = CT.GAS

    # set neighborhood flags
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            if cell_type[ix, iy] & CT.OBSTACLE:
                continue

            # print(bin(cell_type[ix, iy]))
            cell_type[ix, iy] |= (CT.NO_FLUID_NEIGH | CT.NO_EMPTY_NEIGH | CT.NO_IFACE_NEIGH)
            for n in range(1, 9):
                ix_next = ix - v[n, 0]
                iy_next = iy - v[n, 1]

                if cell_type[ix_next, iy_next] & CT.FLUID:
                    cell_type[ix, iy] &= ~CT.NO_FLUID_NEIGH
                elif cell_type[ix_next, iy_next] & CT.GAS:
                    cell_type[ix, iy] &= ~CT.NO_EMPTY_NEIGH
                elif cell_type[ix_next, iy_next] & CT.INTERFACE:
                    cell_type[ix, iy] &= ~CT.NO_IFACE_NEIGH

@jit()
def average_surround(ix, iy, rho, rho_prev, mass, mass_prev, u, u_prev, v, cell_type, equi, t):
    """
    Initialize cell ix, iy with the average of its surroundings
    """
    mass[ix, iy] = 0
    rho[ix, iy] = 0
    u[:, ix, iy] *= 0

    c = 0
    for n in range(1, 9):
        ix_next = ix + v[n, 0]
        iy_next = iy + v[n, 1]

        if cell_type[ix_next, iy_next] & (CT.FLUID | CT.INTERFACE):
            c += 1
            rho[ix, iy] += rho_prev[ix_next, iy_next]
            u[0, ix, iy] += u_prev[0, ix_next, iy_next]
            u[1, ix, iy] += u_prev[1, ix_next, iy_next]

    if c > 0:
        rho[ix, iy] /= c
        u[:, ix, iy] /= c
    set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)






def evolve(total_timesteps, fin, fout, equi, fdist, inlet, u, rho, mass, cell_type, omega, v, v_inv, t, gravity):
    velocities = np.zeros((total_timesteps, *u.shape))
    cell_types = np.zeros((total_timesteps, *cell_type.shape), dtype=type(cell_type))

    for time in range(total_timesteps):
        print(time)

        # collision step
        lbm_collision(fin, fout, equi, inlet, u, rho, cell_type, omega, v, t, gravity)

        # copy intermediate step
        mass_prev = np.copy(mass); rho_prev = np.copy(mass)

        # stream step
        lbm_streaming(fin, fout, equi, u, rho, mass, cell_type, mass_prev, rho_prev, v, v_inv, t)

        # copy intermediate step
        rho_prev = np.copy(rho); mass_prev = np.copy(mass); u_prev = np.copy(u); cell_type_prev = np.copy(cell_type)

        # update the types
        update_types(rho, cell_type, mass, v, fdist, u, equi, t, rho_prev, mass_prev, u_prev, cell_type_prev)

        velocities[time] = u
        cell_types[time] = cell_type

        print(f"max mass = {np.max(mass)}, max rho = {np.max(rho)}")

    return velocities, cell_types



# set the domain variables
nx = 170
ny = 170
omega = np.array([1.0])       # relaxation parameter
gravity = np.array([0, -0.1])
total_timesteps = 100

# get the variables
fin, fout, equi, fdist, inlet, u, rho, mass, cell_type = init(nx, ny)

# make a fluid square
minx, maxx = 0, 70
miny, maxy = 40, ny
xx, yy = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy))
cell_type[xx, yy] = CT.INTERFACE
cell_type[xx[1:-1, 1:-1], yy[1:-1, 1:-1]] = CT.FLUID

# all sides are no slip boundaries
cell_type[:, (0, -1)] = CT.OBSTACLE
cell_type[(0, -1), :] = CT.OBSTACLE

# set the mass and density for fluid cells
mass[cell_type == CT.FLUID] = 1
mass[cell_type == CT.INTERFACE] = 0.5
rho += mass

# initialize fin
for ix in range(rho.shape[0]):
    for iy in range(rho.shape[1]):
        set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)
fin = equi.copy()

# evolve the dam break
u, cell_types = evolve(total_timesteps, fin, fout, equi, fdist, inlet, u, rho, mass, cell_type, omega, v, v_inv, t, gravity)



# animate
def animate(i):
    tmp = np.where(cell_types[i] & CT.FLUID, 1., 0.)
    tmp = np.where(cell_types[i] & CT.INTERFACE, 0.5, tmp)
    im.set_array(tmp.T)
    return [im]

# # animate
# def animate(i):
#     im.set_array(rhos[i].T)
#     return [im]



# # animate
# def animate(i):
#     im.set_array((cell_types[i] & (CT.FLUID | CT.INTERFACE)).T != 0)
#     return [im]


fig = plt.figure()
im = plt.imshow((cell_types[0] & (CT.FLUID | CT.INTERFACE)).T != 0, cmap=plt.cm.Blues)
ani = animation.FuncAnimation(fig, animate, frames=range(0, total_timesteps, 2), interval=50, blit=True)


plt.show()
 
 