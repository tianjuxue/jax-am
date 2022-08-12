import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import time
from src.fem.AuTo.utilfuncs import Mesher, computeLocalElements, computeFilter
from src.fem.AuTo.mmaOptimize import optimize
import matplotlib.pyplot as plt

nelx, nely = 60, 30
elemSize = np.array([1., 1.])
mesh = {'nelx':nelx, 'nely':nely, 'elemSize':elemSize,\
        'ndof':2*(nelx+1)*(nely+1), 'numElems':nelx*nely}

material = {'Emax':1., 'Emin':1e-3, 'nu':0.3, 'penal':3.}

filterRadius = 1.5
H, Hs = computeFilter(mesh, filterRadius)
ft = {'type':1, 'H':H, 'Hs':Hs}


example = 1
if(example == 1):
    # tip cantilever
    force = np.zeros((mesh['ndof'],1))
    dofs=np.arange(mesh['ndof'])
    fixed = dofs[0:2*(nely+1):1]
    free = jnp.setdiff1d(np.arange(mesh['ndof']),fixed)
    force[2*(nelx+1)*(nely+1)-2*nely+1, 0 ] = -1
    symXAxis = False
    symYAxis = False
elif(example == 2):
    ndof = 2*(nelx+1)*(nely+1)
    force = np.zeros((mesh['ndof'],1))
    dofs=np.arange(mesh['ndof'])
    fixed = dofs[0:2*(nely+1):1]
    free = jnp.setdiff1d(np.arange(mesh['ndof']),fixed)
    force[2*(nelx+1)*(nely+1)- (nely+1), 0 ] = -1
    symXAxis = True
    symYAxis = False
bc = {'force':force, 'fixed':fixed,'free':free,\
          'symXAxis':symXAxis, 'symYAxis':symYAxis}


globalVolumeConstraint = {'isOn':True, 'vf':0.5}


optimizationParams = {'maxIters':200,'minIters':100,'relTol':0.05}
projection = {'isOn':False, 'beta':4, 'c0':0.5}


class ComplianceMinimizer:
    def __init__(self, mesh, bc, material, \
                 globalvolCons):
        self.mesh = mesh;
        self.material = material;
        self.bc = bc;
        M = Mesher();
        self.edofMat, self.idx = M.getMeshStructure(mesh);
        self.D0 = M.getD0(self.material);
        self.globalVolumeConstraint = globalvolCons;
        self.objectiveHandle = jit(value_and_grad(self.computeObjective))
        self.consHandle = self.computeConstraints;
        self.numConstraints = 1;
        self.projection = {'isOn':True, 'beta':4, 'c0':0.5}
    #-----------------------#
    def computeObjective(self, rho):
        
        #-----------------------#
        @jit
        def projectionFilter(rho):
            if(self.projection['isOn']):
                v1 = np.tanh(self.projection['c0']*self.projection['beta'])
                nm = v1 + jnp.tanh(self.projection['beta']*(rho-self.projection['c0']))
                dnm = v1 + jnp.tanh(self.projection['beta']*(1.-self.projection['c0']))
                return nm/dnm;
            else:
                return rho
        #-----------------------#
        @jit
        def SIMPMaterial(rho):
            Y = self.material['Emin'] + \
                (self.material['Emax']-self.material['Emin'])*\
                                (rho+0.01)**self.material['penal'];
            return Y;
        Y = SIMPMaterial(rho);
        #-----------------------#
        @jit
        def RAMPMaterial(rho): # test RAMP
            S = 8. # RAMP param
            Y = 0.001*self.material['Emax'] +\
                    self.material['Emax']*(rho/ (1.+S*(1.-rho)) )
            return Y;
        #Y = RAMPMaterial(rho);

        #-----------------------#
        @jit
        def assembleK(Y):
            K = jnp.zeros((self.mesh['ndof'], self.mesh['ndof']));
            kflat_t = (self.D0.flatten()[np.newaxis]).T 
            sK = (kflat_t*Y).T.flatten();
            K = K.at[self.idx[0], self.idx[1]].add(sK)
            return K;
        K = assembleK(Y)
        
        #-----------------------#
        @jit
        def solve(K): 
            u_free = jax.scipy.linalg.solve\
                    (K[self.bc['free'],:][:,self.bc['free']], \
                    self.bc['force'][self.bc['free']], \
                     sym_pos = True, check_finite=False);
            u = jnp.zeros((self.mesh['ndof']));
            u = u.at[self.bc['free']].set(u_free.reshape(-1)) #UPDATED


            return u;
        u = solve(K);
        #-----------------------#
        @jit
        def computeCompliance(K, u):
            J = jnp.dot(u.T, jnp.dot(K,u));
            return J;
        J = computeCompliance(K, u);
        return J;

    #-----------------------#
    def computeConstraints(self, rho, epoch): 
        @jit
        def computeGlobalVolumeConstraint(rho):
            vc = jnp.mean(rho)/self.globalVolumeConstraint['vf'] - 1.;
            return vc;

        c, gradc = value_and_grad(computeGlobalVolumeConstraint)\
                                    (rho);
        c, gradc = c.reshape((1,1)), gradc.reshape((1,-1));
        return c, gradc
    #-----------------------#
    def TO(self, optimizationParams, ft):
        optimize(self.mesh, optimizationParams, ft, \
             self.objectiveHandle, self.consHandle, self.numConstraints);
        

Opt = ComplianceMinimizer(mesh, bc, material, \
                globalVolumeConstraint);
Opt.TO(optimizationParams, ft)

