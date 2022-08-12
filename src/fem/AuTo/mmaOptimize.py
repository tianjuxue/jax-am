import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, grad, random, jacfwd, value_and_grad
from functools import partial
import time
import matplotlib.pyplot as plt
from src.fem.AuTo.utilfuncs import MMA,applySensitivityFilter


def optimize(mesh, optimizationParams, ft, \
             objectiveHandle, consHandle, numConstraints):
    rho = jnp.ones((mesh['nelx']*mesh['nely'])); 
    loop = 0; 
    change = 1.;
    m = numConstraints; # num constraints
    n = mesh['numElems'] ;
    mma = MMA();
    mma.setNumConstraints(numConstraints);
    mma.setNumDesignVariables(n);
    mma.setMinandMaxBoundsForDesignVariables\
        (np.zeros((n,1)),np.ones((n,1)));
    
    xval = rho[np.newaxis].T 
    xold1, xold2 = xval.copy(), xval.copy();
    mma.registerMMAIter(xval, xold1, xold2);
    mma.setLowerAndUpperAsymptotes(np.ones((n,1)), np.ones((n,1)));
    mma.setScalingParams(1.0, np.zeros((m,1)), \
                         10000*np.ones((m,1)), np.zeros((m,1)))
    mma.setMoveLimit(0.2);
    
    mmaTime = 0;
    
    t0 = time.perf_counter();
     
    while( (change > optimizationParams['relTol']) \
           and (loop < optimizationParams['maxIters'])\
           or (loop < optimizationParams['minIters'])):
        loop = loop + 1;
        
        J, dJ = objectiveHandle(rho); 

        vc, dvc = consHandle(rho, loop);

        dJ, dvc = applySensitivityFilter(ft, rho, dJ, dvc)
        J, dJ = J, dJ[np.newaxis].T

        tmr = time.perf_counter();
        mma.setObjectiveWithGradient(J, dJ);
        mma.setConstraintWithGradient(vc, dvc);

        xval = rho.copy()[np.newaxis].T;

        mma.mmasub(xval);

        xmma, _, _ = mma.getOptimalValues();
        xold2 = xold1.copy();
        xold1 = xval.copy();
        rho = xmma.copy().flatten()

        mma.registerMMAIter(rho, xval.copy(), xold1.copy())
        
        mmaTime += time.perf_counter() - tmr;
        
        status = 'Iter {:d}; J {:.2F}; vf {:.2F}'.\
                format(loop, J, jnp.mean(rho));
        print(status)
        if(loop%10 == 0):
            plt.imshow(-np.flipud(rho.reshape((mesh['nelx'], \
                     mesh['nely'])).T), cmap='gray');
            plt.title(status)
            plt.show()
    totTime = time.perf_counter() - t0;
    
    print('total time(s): ', totTime);  
    print('mma time(s): ', mmaTime);
    print('FE time(s): ', totTime - mmaTime);
    return rho;