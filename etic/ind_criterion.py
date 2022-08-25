"""Functions for computing independence criterions.

Author: Lang Liu
Date: 08/23/2022
"""

import warnings

import numpy as np
import ot
import torch
from pykeops.numpy import LazyTensor
from ot.backend import get_backend


##########################################################################
# HSIC
##########################################################################

def hsic(xgram, ygram):
    """Compute the V-statistic version of HSIC.

    Parameters
    ----------
    xgram : array-like, shape (n, n)
        Gram matrix of :math:`\{X_i\}_{i=1}^n`.
    ygram : array-like, shape (n, n)
        Gram matrix of :math:`\{Y_i\}_{i=1}^n`.

    Returns
    -------
    stat : float
        The HSIC statistic.
    """

    term1 = np.mean(xgram * ygram)
    term2 = np.mean(xgram) * np.mean(ygram)
    term3 = np.mean(np.mean(xgram, axis=1) * np.mean(ygram, axis=1))
    return term1 + term2 - 2*term3


##########################################################################
# ETIC
##########################################################################

# TODO: add the discrete case
def etic(xmat, ymat, reg=1.0, max_iter=1000, low_rank=False):
    """Compute the ETIC statistic with additive cost.
    
    Parameters
    ----------
    xmat : array-like, shape (n, n)
        Cost matrix of :math:`\{X_i\}_{i=1}^n`.
        When `low_rank = True`, it is the low-rank approximation of the
        Gibbs kernel `exp(-xmat)`.
    ymat : array-like, shape (n, n)
        Cost matrix of :math:`\{Y_i\}_{i=1}^n`.
        When `low_rank = True`, it is the low-rank approximation of the
        Gibbs kernel `exp(-ymat)`.
    reg : float, optional
        Regularization parameter. Default is ``1.0``.
    max_iter : int, optional
        Maximum number of Sinkhorn iterations. Default is ``1000``.
    low_rank : bool, optional
        Use the low-rank approximation if ``True``. Default is ``False``.
    """
    
    if xmat.shape != ymat.shape:
        raise ValueError('xmat and ymat must have the same size.')
    nx = get_backend(xmat, ymat)

    cost = sinkhorn_grid(
        xmat, ymat, reg, a=nx.eye(len(xmat))/len(xmat), max_iter=max_iter, low_rank=low_rank)

    if low_rank:
        xgram = xmat @ xmat.T
        ygram = ymat @ ymat.T
        if nx.any(xgram < 0) or nx.any(ygram < 0):
            raise ValueError(
                'The elements of the approximated gram matrices should be positive.')
        cost1 = sinkhorn(-reg*nx.log(xgram*ygram), reg, max_iter=max_iter)
        cost2 = sinkhorn_grid_product(
            -reg*nx.log(xgram), -reg*nx.log(ygram), reg, max_iter=max_iter)
    else:
        cost1 = sinkhorn(xmat + ymat, reg, max_iter=max_iter)
        cost2 = sinkhorn_grid_product(xmat, ymat, reg, max_iter=max_iter)
    return cost - cost1/2 - cost2/2


def sinkhorn(cmat, reg, a=[], b=[], max_iter=1000, stop_thresh=1e-8,
             plan=False, cost_only=False, verbose=False, log=False):
    """Solve the EOT problem.
    
    Parameters
    ----------
    cmat : array-like, shape (n, n)
        Cost matrix.
    reg : float
        Regularization parameter.
    a, b : array-like, shape (n, m), optional
        Two marginal distributions. Default is ``[]``.
    max_iter : int, optional
        Maximum number of Sinkhorn iterations. Default is ``1000``.
    stop_thresh : float, optional
        Threshold for stopping. Default is ``1e-8``.
    plan : bool, optional
        Return the optimal transport plan if ``True``. Default is ``False``.
    cost_only : bool, optional
        Return the transport cost without the regularization if ``True``.
        Default is ``False``.
    verbose : bool, optional
        Print progress if ``True``. Default is ``False``.
    log : bool, optional
        Log progress if ``True``. Default is ``False``.
        
    Returns
    -------
    output : float
        Transport cost (with regularization if ``cost_only = False``) or
        transport plan if ``plan = True``.
    log : dict
        Logging information with keys ``'err'``, ``'u'``, and ``'v'``.
    """

    nx = get_backend(cmat)
    size = cmat.shape
    if len(a) == 0:
        a = nx.full((size[0],), 1.0/size[0], type_as=cmat)
    if len(b) == 0:
        b = nx.full((size[1],), 1.0/size[1], type_as=cmat)

    if log:
        sol, log = ot.sinkhorn(
            a, b, cmat, reg, numItermax=max_iter, stopThr=stop_thresh,
            verbose=verbose, log=log)
    else:
        sol = ot.sinkhorn(
            a, b, cmat, reg, numItermax=max_iter, stopThr=stop_thresh,
            verbose=verbose, log=log)
    if plan:
        output = sol
    else:
        output = nx.sum(sol * cmat)
        if not cost_only:
            output += reg * nx.sum(sol * nx.log(sol / nx.outer(a, b)))
    
    if log:
        return output, log
    return output


def sinkhorn_grid(M1, M2, reg, a=[], b=[], max_iter=1000, stop_thresh=1e-8,
                  low_rank=False, plan=False, cost_only=False,
                  verbose=False, log=False):
    """Solve the EOT problem on a grid with additive cost.
    
    Parameters
    ----------
    M1 : array-like, shape (n, n)
        Cost matrix of the first component. When ``low_rank = True``,
        it is the low-rank approximation of the Gibbs kernel `exp(-M1/reg)`.
    M2 : array-like, shape (m, m)
        Cost matrix of the second component. When ``low_rank = True``,
        it is the low-rank approximation of the Gibbs kernel `exp(-M2/reg)`.
    reg : float
        Regularization parameter.
    a, b : array-like, shape (n, m), optional
        Two marginal distributions. Default is ``[]``.
    max_iter : int, optional
        Maximum number of Sinkhorn iterations. Default is ``1000``.
    stop_thresh : float, optional
        Threshold for stopping. Default is ``1e-8``.
    low_rank : bool, optional
        Use the low-rank approximation if ``True``. Default is ``False``.
    plan : bool, optional
        Return the optimal transport plan if ``True``. Default is ``False``.
    cost_only : bool, optional
        Return the transport cost without the regularization if ``True``.
        Default is ``False``.
    verbose : bool, optional
        Print progress if ``True``. Default is ``False``.
    log : bool, optional
        Log progress if ``True``. Default is ``False``.
        
    Returns
    -------
    output : float
        Transport cost (with regularization if ``cost_only = False``) or
        transport plan if ``plan = True``.
    log : dict
        Logging information with keys ``'err'``, ``'u'``, and ``'v'``.
    """
    
    nx = get_backend(M1, M2)
    n, m = len(M1), len(M2)
    
    if log:
        log = {'err': []}
    if len(a) == 0:
        a = nx.full((n, m), 1.0/n/m, type_as=M1)
    if len(b) == 0:
        b = nx.full((n, m), 1.0/n/m, type_as=M2)

    # initialization
    u = nx.full((n, m), 1.0/n/m, type_as=M1)
    v = nx.full((n, m), 1.0/n/m, type_as=M2)
    
    # move to the same device
    if isinstance(M1, torch.Tensor) and M1.get_device() > -1:
        device = M1.get_device()
        a = a.to(device)
        b = b.to(device)
        u = u.to(device)
        v = v.to(device)

    if not low_rank:
        K1 = nx.exp(M1 / (-reg))
        K2 = nx.exp(M2 / (-reg))

    cpt, err = 0, 1
    while (err > stop_thresh and cpt < max_iter):
        uprev, vprev = u, v
        
        # update
        if low_rank:
            KtransposeU = M1 @ ((M1.T @ u) @ M2) @ M2.T
            v = b / KtransposeU
            u = a / (M1 @ ((M1.T @ v) @ M2) @ M2.T)
        else:
            KtransposeU = K1.T @ u @ K2
            v = b / KtransposeU
            u = a / (K1 @ v @ K2.T)

        if (nx.any(KtransposeU == 0) 
            or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
            or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration', cpt)
            u, v = uprev, vprev
            break
        
        # check right marginal violation
        if cpt % 10 == 0:
            # compute right marginal bhat = (diag(u)Kdiag(v))^T 1
            if low_rank:
                bhat = v * (M1 @ ((M1.T @ u) @ M2) @ M2.T)
            else:
                bhat = v * (K1.T @ u @ K2)
            # compute violation of the right marginal
            err = nx.sqrt(nx.sum((bhat - b)**2))

            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt += 1

    if err > stop_thresh:
        warnings.warn(
            'Sinkhorn did not converge. You may want to increase the number of iterations `max_iter` or the regularization parameter `reg`.')

    if low_rank:
        K1, K2 = M1 @ M1.T, M2 @ M2.T

    if plan:  # return EOT plan
        if isinstance(M1, np.ndarray):
            K = np.kron(K1, K2)
        if isinstance(M1, torch.Tensor):
            K = torch.kron(K1, K2)
        output = nx.reshape(u, (-1, 1)) * K * nx.reshape(v, (1, -1))
    else:  # return EOT cost
        if low_rank:
            tmp1 = (u @ M2) @ (M2.T @ v.T)
            tmp2 = (u.T @ M1) @ (M1.T @ v)
            M1 = -reg*nx.log(K1)
            M2 = -reg*nx.log(K2)
        else:
            tmp1 = u @ K2 @ v.T
            tmp2 = u.T @ K1 @ v

        if cost_only:
            output = nx.sum(K1 * M1 * tmp1) + nx.sum(K2 * M2 * tmp2)
        else:
            # sum_{i,j} \Pi_{ij} [\log{(u_i / a_i)} + log{(v_j / b_j)}
            aind, bind = a != 0, b != 0
            output = reg*nx.sum(a[aind]*nx.log(u[aind]/a[aind])) + \
                reg*nx.sum(b[bind]*nx.log(v[bind]/b[bind]))
        
    if log:
        log['u'], log['v'], log['niter'] = u, v, cpt
        return output, log
    return output


def sinkhorn_grid_product(
    M1, M2, reg, a=[], b=[], max_iter=1000, stop_thresh=1e-8,
    plan=False, cost_only=False, verbose=False, log=False):
    """Solve the EOT problem on a grid with additive cost and rank-1 marginals.
    
    Parameters
    ----------
    M1 : array-like, shape (n, n)
        Cost matrix of the first component.
    M2 : array-like, shape (m, m)
        Cost matrix of the second component.
    reg : float
        Regularization parameter.
    a, b : list of two arrays of shapes (n, 1) and (m, 1), optional
        Two marginal distributions. The first marginal is ``a[0] @ a[1].T``.
        Default is ``[]``.
    max_iter : int, optional
        Maximum number of Sinkhorn iterations. Default is ``1000``.
    stop_thresh : float, optional
        Threshold for stopping. Default is ``1e-8``.
    plan : bool, optional
        Return the optimal transport plan if ``True``. Default is ``False``.
    cost_only : bool, optional
        Return the transport cost without the regularization if ``True``.
        Default is ``False``.
    verbose : bool, optional
        Print progress if ``True``. Default is ``False``.
    log : bool, optional
        Log progress if ``True``. Default is ``False``.
        
    Returns
    -------
    output : float
        Transport cost (with regularization if ``cost_only = False``) or
        transport plan if ``plan = True``.
    log : dict
        Logging information with keys ``'err'``, ``'u'``, and ``'v'``.
    """
    
    nx = get_backend(M1, M2)
    n, m = M1.shape[0], M2.shape[0]

    if log:
        log = {'err': []}
    if len(a) == 0:
        a = (nx.full((n, 1), 1.0/n, type_as=M1), nx.full((m, 1), 1.0/m, type_as=M2))
    if len(b) == 0:
        b = (nx.full((n, 1), 1.0/n, type_as=M1), nx.full((m, 1), 1.0/m, type_as=M2))
        
    # initialization
    u = (nx.full((n, 1), 1.0/n, type_as=M1), nx.full((m, 1), 1.0/m, type_as=M2))
    v = (nx.full((n, 1), 1.0/n, type_as=M1), nx.full((m, 1), 1.0/m, type_as=M2))
    
    # move to the same device
    if isinstance(M1, torch.Tensor) and M1.get_device() > -1:
        device = M1.get_device()
        a = (a[0].to(device), a[1].to(device))
        b = (b[0].to(device), b[1].to(device))
        u = (u[0].to(device), u[1].to(device))
        v = (v[0].to(device), v[1].to(device))
    
    K1 = nx.exp(M1 / (-reg))
    K2 = nx.exp(M2 / (-reg))

    cpt, err = 0, 1
    while (err > stop_thresh and cpt < max_iter):
        uprev, vprev = u, v

        # update
        v = (b[0] / (K1.T @ u[0]), b[1] / (K2.T @ u[1]))
        u = (a[0] / (K1 @ v[0]), a[1] / (K2 @ v[1]))

        if (nx.any(nx.isnan(u[0])) or nx.any(nx.isnan(u[1]))
                or nx.any(nx.isnan(v[0])) or nx.any(nx.isnan(v[1]))
                or nx.any(nx.isinf(u[0])) or nx.any(nx.isinf(u[1]))
                or nx.any(nx.isinf(v[0])) or nx.any(nx.isinf(v[1]))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u, v = uprev, vprev
            break
        if cpt % 10 == 0:
            # compute violation of the right marginal
            bhat = (v[0]*(K1.T @ u[0]), v[1]*(K2.T @ u[1]))
            err = nx.sqrt(nx.sum((bhat[0] @ bhat[1].T - b[0] @ b[1].T)**2))

            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
        
    if err > stop_thresh:
        warnings.warn(
            'Sinkhorn did not converge. You may want to increase the number of iterations `max_iter` or the regularization parameter `reg`.')

    if plan:  # return EOT plan
        K1 = u[0] * K1 * nx.reshape(v[0], (1, -1))
        K2 = u[1] * K2 * nx.reshape(v[1], (1, -1))
        if isinstance(M1, np.ndarray):
            output = np.kron(K1, K2)
        if isinstance(M1, torch.Tensor):
            output = torch.kron(K1, K2)
    else:
        tmp1 = (u[1].T @ K2 @ v[1]) * u[0] @ v[0].T
        tmp2 = (u[0].T @ K1 @ v[0]) * u[1] @ v[1].T
        output = nx.sum(K1 * M1 * tmp1) + nx.sum(K2 * M2 * tmp2)
        if not cost_only:
            # sum_{i,j} \Pi_{ij} [\log{(u_i / a_i)} + log{(v_j / b_j)}
            output = reg*nx.sum(a[0]*nx.log(u[0]/a[0]))*nx.sum(a[1]) + \
                reg*nx.sum(a[1]*nx.log(u[1]/a[1]))*nx.sum(a[0]) + \
                reg*nx.sum(b[0]*nx.log(v[0]/b[0]))*nx.sum(b[1]) + \
                reg*nx.sum(b[1]*nx.log(v[1]/b[1]))*nx.sum(b[0])
    if log:
        log['u'], log['v'], log['niter'] = u, v, cpt
        return output, log
    return output


##########################################################################
# ETIC with large-scale Sinkhorn
##########################################################################

def large_scale_etic(xmat, ymat, reg=1.0, max_iter=1000, low_rank=False):
    """A large-scale implementation of the ETIC statistic.
    
    Parameters
    ----------
    xmat : pykeops.LazyTensor, shape (n, n)
        Cost matrix of :math:`\{X_i\}_{i=1}^n`.
        When `low_rank = True`, it is the low-rank approximation of the
        Gibbs kernel `exp(-xmat)`.
    ymat : pykeops.LazyTensor, shape (n, n)
        Cost matrix of :math:`\{Y_i\}_{i=1}^n`.
        When `low_rank = True`, it is the low-rank approximation of the
        Gibbs kernel `exp(-ymat)`.
    reg : float, optional
        Regularization parameter. Default is ``1.0``.
    max_iter : int, optional
        Maximum number of Sinkhorn iterations. Default is ``1000``.
    low_rank : bool, optional
        Use the low-rank approximation if ``True``. Default is ``False``.
    """
    
    if xmat.shape != ymat.shape:
        raise ValueError('xmat and ymat must have the same size.')

    cost = large_scale_sinkhorn_diag_prod(
        xmat, ymat, reg, max_iter=max_iter, low_rank=low_rank)

    if low_rank:
        xgram = _outer_prod(xmat, xmat)
        ygram = _outer_prod(ymat, ymat)
        cost1 = large_scale_sinkhorn(-(xgram*ygram).log()*reg, reg, max_iter=max_iter)
        cost2 = large_scale_sinkhorn_grid_product(
            -reg*(xgram.log()), -reg*(ygram.log()), reg, max_iter=max_iter)
    else:
        cost1 = large_scale_sinkhorn(xmat + ymat, reg, max_iter=max_iter)
        cost2 = large_scale_sinkhorn_grid_product(xmat, ymat, reg, max_iter=max_iter)
    return cost - cost1/2 - cost2/2


def _outer_prod(x, y):
    return (LazyTensor(x[:, None, :]) * LazyTensor(y[None, :, :])).sum(-1)


def _compute_error(a, b):
    out_a = _outer_prod(a[0], a[1])
    out_b = _outer_prod(b[0], b[1])
    return np.sqrt(np.sum(((out_a - out_b)**2).sum(1)))


def large_scale_sinkhorn(M, reg, a=[], b=[], max_iter=1000, stop_thresh=1e-8,
                         cost_only=False, verbose=False, log=False):
    """A large-scale implementation of the Sinkhorn algorithm.
    
    Parameters
    ----------
    M : pykeops.LazyTensor, shape (n, n)
        Cost matrix.
    reg : float
        Regularization parameter.
    a, b : array-like, shape (n, m), optional
        Two marginal distributions. Default is ``[]``.
    max_iter : int, optional
        Maximum number of Sinkhorn iterations. Default is ``1000``.
    stop_thresh : float, optional
        Threshold for stopping. Default is ``1e-8``.
    cost_only : bool, optional
        Return the transport cost without the regularization if ``True``.
        Default is ``False``.
    verbose : bool, optional
        Print progress if ``True``. Default is ``False``.
    log : bool, optional
        Log progress if ``True``. Default is ``False``.
        
    Returns
    -------
    cost : float
        Transport cost (with regularization if ``cost_only = False``)
    log : dict
        Logging information with keys ``'err'``, ``'u'``, and ``'v'``.
    """
    
    n = M.shape[0]
    if log:
        log = {'err': []}
    if len(a) == 0:
        a = np.full((n,), 1.0/n)
    if len(b) == 0:
        b = np.full((n,), 1.0/n)

    # initialization
    u = np.full((n,), 1.0/n)
    v = np.full((n,), 1.0/n)
    
    K = (M / (-reg)).exp()
    cpt, err = 0, 1
    while (err > stop_thresh and cpt < max_iter):
        uprev, vprev = u, v
        
        # update
        KtransposeU = K.T @ u
        v = b / KtransposeU
        u = a / (K @ v)

        if (np.any(KtransposeU == 0) 
            or np.any(np.isnan(u)) or np.any(np.isnan(v))
            or np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Warning: numerical errors at iteration', cpt)
            u, v = uprev, vprev
            break
        
        # check right marginal violation
        if cpt % 10 == 0:
            # compute right marginal bhat = (diag(u)Kdiag(v))^T 1
            bhat = v * (K.T @ u)
            err = np.sqrt(np.sum((bhat - b)**2))

            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt += 1

    if err > stop_thresh:
        warnings.warn(
            'Sinkhorn did not converge. You may want to increase the number of iterations `max_iter` or the regularization parameter `reg`.')

    if cost_only:
        cost = np.sum(((K * M) @ v) * u)
    else:
        # sum_{i,j} \Pi_{ij} [\log{(u_i / a_i)} + log{(v_j / b_j)}
        cost = reg*np.sum(a * np.log(u / a) + b * np.log(v / b))
        
    if log:
        log['u'], log['v'], log['niter'] = u, v, cpt
        return cost, log
    return cost


def large_scale_sinkhorn_grid_product(
    M1, M2, reg, a=[], b=[], max_iter=1000, stop_thresh=1e-8,
    cost_only=False, verbose=False, log=False):
    n, m = M1.shape[0], M2.shape[0]

    if log:
        log = {'err': []}
    if len(a) == 0:
        a = (np.full((n, 1), 1.0/n), np.full((m, 1), 1.0/m))
    if len(b) == 0:
        b = (np.full((n, 1), 1.0/n), np.full((m, 1), 1.0/m))

    # uniform initialization
    u = (np.full((n, 1), 1.0/n), np.full((m, 1), 1.0/m))
    v = (np.full((n, 1), 1.0/n), np.full((m, 1), 1.0/m))

    K1 = (M1 / (-reg)).exp()
    K2 = (M2 / (-reg)).exp()

    cpt, err = 0, 1
    while (err > stop_thresh and cpt < max_iter):
        uprev, vprev = u, v

        # update
        v = (b[0] / (K1.T @ u[0]), b[1] / (K2.T @ u[1]))
        u = (a[0] / (K1 @ v[0]), a[1] / (K2 @ v[1]))

        if (np.any(np.isnan(u[0])) or np.any(np.isnan(u[1]))
                or np.any(np.isnan(v[0])) or np.any(np.isnan(v[1]))
                or np.any(np.isinf(u[0])) or np.any(np.isinf(u[1]))
                or np.any(np.isinf(v[0])) or np.any(np.isinf(v[1]))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u, v = uprev, vprev
            break

        if cpt % 10 == 0:
            # compute right marginal tmp = (diag(u)Kdiag(v))^T1
            bhat = (v[0]*(K1.t() @ u[0]), v[1]*(K2.t() @ u[1]))
            err = _compute_error(bhat, b)

            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
        
    if err > stop_thresh:
        warnings.warn(
            'Sinkhorn did not converge. You may want to increase the number of iterations `max_iter` or the regularization parameter `reg`.')

    const = u[1].T @ (K2 @ v[1])
    tmp1 = _outer_prod(u[0]*const, v[0])
    const = u[0].T @ (K1 @ v[0])
    tmp2 = _outer_prod(u[1]*const, v[1])
    # this could be simplified if tmp1 and tmp2 are simply outer products
    cost = np.sum((K1 * M1 * tmp1).sum(1)) + np.sum((K2 * M2 * tmp2).sum(1))
    if not cost_only:
        cost = reg*np.sum(a[0]*np.log(u[0]/a[0]))*np.sum(a[1]) + \
            reg*np.sum(a[1]*np.log(u[1]/a[1]))*np.sum(a[0]) + \
            reg*np.sum(b[0]*np.log(v[0]/b[0]))*np.sum(b[1]) + \
            reg*np.sum(b[1]*np.log(v[1]/b[1]))*np.sum(b[0])

    if log:
        log['u'], log['v'], log['niter'] = u, v, cpt
        return cost, log
    else:
        return cost


def basis(n, ind):
    e = np.zeros(n)
    e[ind] = 1
    return e


def update_v_func(b, u, K1, K2):
    def v_func(ind, column=True):
        e = basis(len(u), ind)
        if column:
            denom = (K2 @ e) * u
            denom = K1.T @ denom
            return b[0] * b[1][ind] / denom
        else:
            denom = (K1 @ e) * u
            denom = K2.T @ denom
            return b[1] * b[0][ind] / denom
    return v_func


def large_scale_sinkhorn_diag_prod(
    M1, M2, reg, a=[], b=[], max_iter=1000, stop_thresh=1e-8,
    low_rank=False, cost_only=False, verbose=False, log=False):
    n = M1.shape[0]

    if log:
        log = {'err': []}
    if len(a) == 0:
        a = np.full(n, 1.0/n)
    if len(b) == 0:
        if low_rank:
            b = (np.full((n, 1), 1.0/n), np.full((n, 1), 1.0/n))
        else:
            b = (np.full(n, 1.0/n), np.full(n, 1.0/n))

    # uniform initialization
    if low_rank:
        u = (np.full((n, 1), 1.0/n), np.full((n, 1), 1.0/n))
        v = (np.full((n, 1), 1.0/n), np.full((n, 1), 1.0/n))
    else:
        u = (np.full(n, 1.0/n), np.full(n, 1.0/n))
        def v(ind, column=True):  # get a column of v
            return np.ones(n) / n**2

        K1 = (M1 / (-reg)).exp()
        K2 = (M2 / (-reg)).exp()

    cpt, err = 0, 1
    while (err > stop_thresh and cpt < max_iter):
        uprev, vprev_v = u, v
        if low_rank:
            if cpt == 0:
                v = (b[0] / np.dot(M1, np.dot(M1.T, u[0])),
                     b[1] / np.dot(M2, np.dot(M2.T, u[1])))
                u = a / (np.dot(M1, np.dot(M1.T, v[0])) * np.dot(
                    M2, np.dot(M2.T, v[1]))).reshape(-1)
                v = _outer_prod(v[0], v[1])  # LazyTensor
            else:
                v = _outer_prod(b[0], b[1]) / _outer_prod(
                    M1, np.dot(M2, np.dot(M2.T * u, M1)))
                tmp = np.dot(np.dot(M1.T, (v @ M2)), M2.T)
                u = a / np.einsum('ij,ji->i', M1, tmp)
        else:
            if cpt == 0:
                def v(ind, column=True):
                    K1U = K1.T @ u[0]
                    K2U = K2.T @ u[1]
                    if column:
                        return b[0] / K1U * b[1][ind] / K2U[ind]
                    else:
                        return b[1] / K2U * b[0][ind] / K1U[ind]
            else:
                v = update_v_func(b, u, K1, K2)

            udenom = np.zeros(n)
            for col in range(n):
                e = basis(n, col)
                Kv_col = K1 @ v(col)
                udenom += (K2 @ e) * Kv_col
            u = a / udenom

        if (np.any(np.isnan(u[0])) or np.any(np.isnan(u[1]))
                or np.any(np.isinf(u[0])) or np.any(np.isinf(u[1]))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u, v = uprev, vprev_v
            break
        if cpt % 10 == 0:
            # compute right marginal tmp= (diag(u)Kdiag(v))^T1
            if low_rank:
                margin = v * _outer_prod(
                    M1, np.dot(M2, np.dot(M2.T * u, M1)))
                out_b = _outer_prod(b[0], b[1])
                err = np.sqrt(np.sum(((margin - out_b)**2).sum(1)))
            else:
                err = 0.0  # violation of marginal
                for col in range(n):
                    e = basis(n, col)
                    tmp = (K2 @ e) * u
                    tmp = (K1.T @ tmp) * v(col)
                    err += np.sum((tmp - b[0]*b[1][col])**2)
                err = np.sqrt(err)

            if log:
                log['err'].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        cpt = cpt + 1
        
    if err > stop_thresh:
        warnings.warn(
            'Sinkhorn did not converge. You may want to increase the number of iterations `max_iter` or the regularization parameter `reg`.')

    if low_rank:
        tmp1 = _outer_prod(M2 * u.reshape(-1, 1), v @ M2)
        tmp2 = _outer_prod(M1 * u.reshape(-1, 1), v.T @ M1)
        K1 = _outer_prod(M1, M1)
        C1 = -reg * K1.log()
        K2 = _outer_prod(M2, M2)
        C2 = -reg * K2.log()
        cost = np.sum((K1 * C1 * tmp1).sum(1)) + np.sum((K2 * C2 * tmp2).sum(1))
    else:
        cost = 0.0
        for ind in range(n):
            e = basis(len(u), ind)
            tmp1 = v(ind, column=False)
            tmp1 = K2 @ tmp1 * u
            tmp1 *= (K1 * M1) @ e
            tmp2 = v(ind)
            tmp2 = K1 @ tmp2 * u
            tmp2 *= (K2 * M2) @ e
            cost += np.sum(tmp1) + np.sum(tmp2)
    
    if not cost_only:
        # sum_{i,j} \Pi_{ij} [\log{(u_i / a_i)} + log{(v_j / b_j)}
        cost = reg*np.sum(a*np.log(u/a))
        if low_rank:
            out_b = _outer_prod(b[0], b[1])
            cost += reg*np.sum((out_b * (v / out_b).log()).sum(1))
        else:
            for col in range(n):
                cost += reg*np.sum(b[0]*b[1][ind]*np.log(v(col)/b[0]/b[1][ind]))

    if log:
        log['u'], log['v'], log['niter'] = u, v, cpt
        return cost, log
    else:
        return cost
