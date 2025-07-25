"""
Adapted from: https://github.com/ddocquier/Causal_Comp
Original author: David Docquier
MIT License | Date copied: June 2025
Modified by: Chivintar Amenty

This module computes the rate of information transfer (T) from variable xj to xi,
as well as the normalized transfer rate (tau) and Pearson correlation (R),
for multivariate time series.

Method based on:
  Liang (2021), "Normalized multivariate time series causality analysis and causal graph reconstruction",
  *Entropy*, doi: 10.3390/e23060679

Features:
- Estimates T, tau, and R using closed-form expressions
- Bootstraps to compute uncertainties (normal resampling of variables)

Inputs:
- x: numpy array of shape (nvar, N), where nvar is the number of variables and N the time series length
- dt: time step (e.g., 1 month)
- n_iter: number of bootstrap realizations for error estimation

Outputs:
- T: absolute rate of information transfer (nvar x nvar)
- tau: normalized (relative) information transfer rate (nvar x nvar)
- R: Pearson correlation coefficient (nvar x nvar)
- error_T, error_tau, error_R: corresponding bootstrap errors
"""

import numpy as np
#import numba # translates Python and numpy code into fast machine code
#from numba import jit

#@jit(nopython=True) # accelerator
def compute_sig(var,error,conf):
	'''
	Function to compute statistically significant values
	'''
	if (var-conf*error < 0. and var+conf*error < 0.) or (var-conf*error > 0. and var+conf*error > 0.):
		sig = 1
	else:
		sig = 0
	return sig


def compute_liang(x, dt, n_iter=500,method='bootstrap',return_boots=False,return_sig=False, alpha=0.05):
    alpha = alpha*100
    # helper: two-variable Liang index
    def liangk2(xi, xj, dt):
        data = np.vstack((xi, xj))
        N2 = xi.size
        dx = np.zeros_like(data)
        dx[0, :-1] = np.diff(xi) / dt
        dx[1, :-1] = np.diff(xj) / dt
        C2 = np.cov(data)
        dC2 = np.zeros_like(C2)
        for ii in range(2):
            for jj in range(2):
                dC2[jj, ii] = np.mean((data[jj]-data[jj].mean())*(dx[ii]-dx[ii].mean()))
        det2 = np.linalg.det(C2)
        Delta2 = np.linalg.inv(C2).T * det2
        # T2
        T2 = np.zeros((2,2))
        for ii in range(2):
            for jj in range(2):
                T2[jj,ii] = (1/det2)*np.sum(Delta2[jj,:]*dC2[:,ii])*(C2[ii,jj]/C2[ii,ii])
        # noise
        g2 = np.zeros(2)
        for ii in range(2):
            a1 = np.linalg.inv(C2) @ dC2[:,ii]
            f1 = dx[ii].mean() - (a1 * data.mean(axis=1)).sum()
            R1 = dx[ii] - f1 - a1 @ data
            g2[ii] = (R1**2).sum() * dt / N2
        # tau2
        tau2 = np.zeros((2,2))
        for ii in range(2):
            for jj in range(2):
                selfc = (1/det2) * np.sum(Delta2[ii,:] * dC2[:,ii])
                supp = np.abs(T2[:,ii]).sum() - abs(T2[ii,ii])
                noise = 0.5 * g2[ii] / C2[ii,ii]
                Z = abs(selfc) + supp + abs(noise)
                tau2[jj,ii] = 100.0 * T2[jj,ii] / Z
        return T2[1,0], tau2[1,0]

    nvar, N = x.shape
    # compute dx, C, dC, T, R, noise g, tau
    dx = np.zeros((nvar,N))
    for i in range(nvar): 
        dx[i,:-1] = np.diff(x[i]) / dt
    
    C = np.cov(x)
    dC = np.zeros_like(C)
    for i in range(nvar):
        for j in range(nvar):
            dC[j,i] = np.mean((x[j]-x[j].mean())*(dx[i]-dx[i].mean()))
    
    detC = np.linalg.det(C)
    Delta = np.linalg.inv(C).T * detC
    T = np.zeros((nvar,nvar)); R = np.zeros((nvar,nvar))
    
    for i in range(nvar):
        for j in range(nvar):
            T[j,i] = (1/detC)*np.sum(Delta[j,:]*dC[:,i])*(C[i,j]/C[i,i])
            R[j,i] = C[i,j] / np.sqrt(C[i,i]*C[j,j])
    g = np.zeros(nvar)
    
    for i in range(nvar):
        a1 = np.linalg.inv(C) @ dC[:,i]
        f1 = dx[i].mean() - (a1 * x.mean(axis=1)).sum()
        R1 = dx[i] - f1 - a1 @ x
        g[i] = (R1**2).sum() * dt / N
    tau = np.zeros((nvar,nvar))
    
    for i in range(nvar):
        for j in range(nvar):
            selfc = (1/detC)*np.sum(Delta[i,:]*dC[:,i])
            supp = np.abs(T[:,i]).sum() - abs(T[i,i])
            noise = 0.5 * g[i] / C[i,i]
            Z = abs(selfc) + supp + abs(noise)
            tau[j,i] = 100.0 * T[j,i] / Z
    
    # prepare defaults
    error_T = np.zeros_like(T)
    error_tau = np.zeros_like(tau) 
    error_R = np.zeros_like(R)
    sig_T = sig_tau = None

    if method == 'bootstrap':
        
        # bootstrap errors
        boot_T = np.zeros((n_iter,nvar,nvar))
        boot_tau = np.zeros((n_iter,nvar,nvar))
        boot_R = np.zeros((n_iter,nvar,nvar))
        
        for it in range(n_iter):
            idx = np.random.choice(N, N, replace=True)
            bx = x[:,idx]; bdx = dx[:,idx]
            bC = np.cov(bx)
            bdC = np.zeros_like(bC)
            
            for i in range(nvar):
                for j in range(nvar):
                    bdC[j,i] = np.mean((bx[j]-bx[j].mean())*(bdx[i]-bdx[i].mean()))
            bdet = np.linalg.det(bC)
            bDel = np.linalg.inv(bC).T * bdet
            
            # recompute T,R
            for i in range(nvar):
                for j in range(nvar):
                    boot_T[it,j,i] = (1/bdet)*np.sum(bDel[j,:]*bdC[:,i])*(bC[i,j]/bC[i,i])
                    boot_R[it,j,i] = bC[i,j] / np.sqrt(bC[i,i]*bC[j,j])
            
            # noise
            bg = np.zeros(nvar)
            for i in range(nvar):
                a1b = np.linalg.inv(bC) @ bdC[:,i]
                f1b = bdx[i].mean() - (a1b * bx.mean(axis=1)).sum()
                R1b = bdx[i] - f1b - a1b @ bx
                bg[i] = (R1b**2).sum() * dt / N
            
            # tau
            for i in range(nvar):
                for j in range(nvar):
                    selfc = (1/bdet)*np.sum(bDel[i,:]*bdC[:,i])
                    supp = np.abs(boot_T[it,:,i]).sum() - abs(boot_T[it,i,i])
                    noise = 0.5 * bg[i] / bC[i,i]
                    Z = abs(selfc) + supp + abs(noise)
                    boot_tau[it,j,i] = 100.0 * boot_T[it,j,i] / Z
       
        error_T = np.nanstd(boot_T, axis=0)
        error_tau = np.nanstd(boot_tau, axis=0)
        error_R = np.nanstd(boot_R, axis=0)
        
        if return_boots:
            sigs = (boot_T, boot_tau, boot_R)
    
    
    elif method == 'permutation':
        # permutation significance test replicating Hristopulos Matlab code 'permutation_test_lifr_v3'
        sig_T = np.zeros((nvar,nvar),int)
        sig_tau = np.zeros((nvar,nvar),int)
        
        for i in range(nvar):
            # observed cross-corr and delayed correlation
            rij_obs = np.corrcoef(x[i], x[i])[0,0]  # unused
            rijjdt = np.corrcoef(x[i], np.roll(x[i],1))[0,1]
            
            for j in range(nvar):
                if i == j: continue
                
                # observed values
                r12_obs = np.corrcoef(x[j], x[i])[0,1]
                xj_shift = np.roll(x[j],1)
                r12dt_obs = np.corrcoef(x[i], xj_shift)[0,1]
                rij_delay_obs = r12dt_obs - r12_obs * rijjdt
                Tij_obs = T[j,i]
                tau_obs = tau[j,i]
                
                # null distributions
                rij_per = np.zeros(n_iter)
                rij_delay_per = np.zeros(n_iter)
                T_per = np.zeros(n_iter)
                tau_per = np.zeros(n_iter)
                
                for k in range(n_iter):
                    xj_p = np.random.permutation(x[j])
                    r12p = np.corrcoef(xj_p, x[i])[0,1]
                    r12dt_p = np.corrcoef(np.roll(xj_p,1), x[i])[0,1]
                    rij_per[k] = r12p
                    rij_delay_per[k] = r12dt_p - r12p * rijjdt
                    T_per[k], tau_per[k] = liangk2(x[i], xj_p, dt)
                # bounds for zero-corr test
                low, high = np.percentile(rij_per, [alpha/2, 100-alpha/2])
                # mask fully randomized (within bounds)
                mask = (rij_per >= low) & (rij_per <= high)
                # H_r12: significant if r12_obs outside bounds
                H_r12 = int((r12_obs < low) or (r12_obs > high))
                # filter for delay test
                if mask.sum() > 0:
                    rd = rij_delay_per[mask]
                    lo2, hi2 = np.percentile(rd, [alpha/2, 100-alpha/2])
                    H_delay = int((rij_delay_obs < lo2) or (rij_delay_obs > hi2))
                else:
                    H_delay = 0
                # final IFR p-value on fully randomized set
                if mask.sum() > 0:
                    T_f = T_per[mask]
                    tau_f = tau_per[mask]
                    pT = np.mean(np.abs(T_f) >= abs(Tij_obs))
                    p_tau = np.mean(np.abs(tau_f) >= abs(tau_obs))
                else:
                    pT = p_tau = 1.0
                
                # IFR significance: require both H_r12 and H_delay and pT<p
                sig_T[j,i] = int(H_r12 and H_delay and (pT < alpha/100))
                sig_tau[j,i] = int(H_r12 and H_delay and (p_tau < alpha/100))
    
    # assemble outputs
    output = [T, tau, R, error_T, error_tau, error_R]
    if method=='bootstrap' and return_boots:
        output += list(sigs)
    if method=='permutation' and return_sig:
        output += [sig_T, sig_tau]
    return tuple(output)

