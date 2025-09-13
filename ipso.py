import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
import math


def pso(func, lb, ub, seed=0, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, swarmsize=100, maxiter=100, minstep=1e-8, minfunc=1e-8):
   
    lb = np.array(lb)
    ub = np.array(ub)
   
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    
    # Check for constraint function(s) #########################################
    obj = lambda x: func(x, *args, **kwargs)
    if f_ieqcons is None:
        if not len(ieqcons):
            cons = lambda x: np.array([0])
        else:
            cons = lambda x: np.array([y(x, *args, **kwargs) for y in ieqcons])
    else:
        cons = lambda x: np.array(f_ieqcons(x, *args, **kwargs))
        
    def is_feasible(x):
        check = np.all(cons(x)>=0)
        return check
        
    # Initialize the particle swarm ############################################
    S = swarmsize
    D = len(lb)  # the number of dimensions each particle has
    np.random.seed(seed)
    random.seed(seed)
    x = np.random.rand(S, D)  # particle positions
    v = np.zeros_like(x)  # particle velocities
    p = np.zeros_like(x)  # best particle positions
    fp = np.zeros(S)  # best particle function values
    g = []  # best swarm position
    fg = 1e100  # artificial best swarm position starting value
    
    for i in range(S):
        # Initialize the particle's position
        x[i, :] = lb + x[i, :]*(ub - lb)
   
        # Initialize the particle's best known position
        p[i, :] = x[i, :]
       
        # Calculate the objective's value at the current particle's
        fp[i] = obj(p[i, :])
       
        # At the start, there may not be any feasible starting point, so just
        # give it a temporary "best" point since it's likely to change
        if i==0:
            g = p[0, :].copy()

        # If the current particle's position is better than the swarm's,
        # update the best swarm position
        if fp[i]<fg and is_feasible(p[i, :]):
            fg = fp[i]
            g = p[i, :].copy()
       
        # Initialize the particle's velocity
        np.random.seed(seed)
        random.seed(seed)
        r=np.random.rand(D)
        v[i, :] = vlow + r*(vhigh - vlow)
       
    # Iterate until termination criterion met ##################################
    it = 1
    w_max = 0.9
    w_min = 0.1
    phip_init = 2.5
    phip_end = 0.5
    phig_init = 0.5
    phig_end = 2.5
    while it<=maxiter:
        np.random.seed(seed)
        random.seed(seed)
        rp = np.random.uniform(size=(S, D))
        np.random.seed(seed)
        random.seed(seed)
        rg = np.random.uniform(size=(S, D))
        for i in range(S):
            
            omega = w_max - math.sin((math.pi)*(it/maxiter)**0.5)*(w_max-w_min)
            phip = phip_init + it * (phip_end - phip_init)/maxiter
            phig = phig_init + it * (phig_end - phig_init)/maxiter

            # Update the particle's velocity
            v[i, :] = omega*v[i, :] + phip*rp[i, :]*(p[i, :] - x[i, :]) + \
                      phig*rg[i, :]*(g - x[i, :])
                      
            # Update the particle's position, correcting lower and upper bound 
            # violations, then update the objective function value
            x[i, :] = x[i, :] + v[i, :]
            mark1 = x[i, :]<lb
            mark2 = x[i, :]>ub
            x[i, mark1] = lb[mark1]
            x[i, mark2] = ub[mark2]
            fx = obj(x[i, :])
            
            # Compare particle's best position (if constraints are satisfied)
            if fx<fp[i] and is_feasible(x[i, :]):
                p[i, :] = x[i, :].copy()
                fp[i] = fx

                # Compare swarm's best position to current particle's position
                # (Can only get here if constraints are satisfied)
                if fx<fg:
                    tmp = x[i, :].copy()
                    stepsize = np.sqrt(np.sum((g-tmp)**2))
                    if np.abs(fg - fx)<=minfunc:
                        print('Stopping search: Swarm best objective change less than {:}'.format(minfunc))
                        return tmp, fx
                    elif stepsize<=minstep:
                        print('Stopping search: Swarm best position change less than {:}'.format(minstep))
                        return tmp, fx
                    else:
                        g = tmp.copy()
                        fg = fx
        it += 1
    print('Stopping search: maximum iterations reached --> {:}'.format(maxiter))
    return g, fg


def get_optimasation_function(preds, labels, auc_type='roc'):
    labels = labels
    #--------------------------------------
    def aucroc_optimisation(weights):
        new_weights = weights
        pred_ensem = 0
        for i in range(len(preds)):
            pred_ensem += preds[i]*(new_weights[i]/sum(new_weights))
        
        #roc_auc = roc_auc_score(labels, pred_ensem)
        aupr = average_precision_score(labels, pred_ensem)
        objective = 1 - aupr
        return objective

    if auc_type == 'roc':
        return aucroc_optimisation