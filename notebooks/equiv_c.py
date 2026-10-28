"""Equiv-c (EquivSINDy-c) from Yang et al., "Symmetry-Informed Governing Equation Discovery", arXiv:2405.16756,
following the sparse-regression path of github.com/Rose-STL-Lab/symmetry-ode-discovery (sindy.py: SINDyRegression
with L_list, solve_SINDy / solve_SINDy_one_step).

The paper derives the constraint from Lie-algebra generators (infinitesimal equivariance, M from J_Theta L z). A finite
group has no Lie algebra, so for the signed-permutation symmetries of dysts we use the group-element form of the same
constraint, f(g x) = g f(x)  =>  Theta(g x) = Theta(x) M_g and M_g W = W g^T, with M_g obtained by substitution (as in
symmetric_sindy_v1). Note: passing g as an "L" to the reference code would not work, because its get_M_list computes
M from J_Theta(z) L z, which equals Theta(L z) only for the linear terms.

What is reproduced from the reference implementation:
  - null-space parametrization vec(W) = Q beta, Q = right singular vectors of the stacked constraint matrix whose
    singular values are <= 5e-3 (absolute cutoff, as in get_Q)
  - constant terms are left unconstrained by default (constrain_constant=False -> allow_constant=True)
  - sequential thresholding on |W| of the raw (un-normalized) library: the mask accumulates, the regression is
    lstsq on Theta[:, mask] @ Q[mask] (masked entries are dropped from the fit, not constrained to zero), all-zero
    columns are removed, at most max_iter=5 rounds, stop when the mask stops changing
  - optional ridge w_sindy_reg (0 in the paper's EquivSINDy-c configs)
The reference code solves with torch / L-BFGS on GPU; here the same least-squares problems are solved with numpy.
"""
import numpy as np


class EquivC:
    def __init__(self, threshold=1e-2, constraint=None, max_iter=5, w_sindy_reg=0.0, constrain_constant=False,
                 sv_cutoff=5e-3):
        self.threshold = threshold
        self.constraint = constraint  # rows act on vec(W) in target order (column-major of the (h, n) W)
        self.max_iter = max_iter
        self.w_sindy_reg = w_sindy_reg
        self.constrain_constant = constrain_constant
        self.sv_cutoff = sv_cutoff

    def _null_basis(self, hn):
        if self.constraint is None or not len(self.constraint):
            return np.eye(hn)
        _, S, Vt = np.linalg.svd(self.constraint, full_matrices=True)
        S = np.concatenate([S, np.zeros(Vt.shape[0] - len(S))])
        return Vt[S <= self.sv_cutoff].T

    def fit(self, Phi, Y, const_index=0):
        Phi, Y = np.asarray(Phi, float), np.asarray(Y, float)
        m, h = Phi.shape
        n = Y.shape[1]
        Q = self._null_basis(h * n)
        if not self.constrain_constant and self.constraint is not None:
            E = np.zeros((h * n, n))
            E[const_index + h * np.arange(n), np.arange(n)] = 1.0  # one free constant per equation
            Q = np.hstack([Q, E])
        A1 = np.vstack([Phi, self.w_sindy_reg * np.eye(h)])
        B = np.vstack([Y, np.zeros((h, n))]).flatten(order="F")
        A = np.kron(np.eye(n), A1)  # block_diag over equations, acting on vec(W) in target order

        mask = np.ones(h * n, bool)
        W = np.zeros(h * n)
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            AQ = A[:, mask] @ Q[mask]
            eff = np.any(AQ != 0.0, axis=0)
            beta = np.zeros(Q.shape[1])
            beta[eff], *_ = np.linalg.lstsq(AQ[:, eff], B, rcond=None)
            W = (Q @ beta) * mask
            new_mask = mask & (np.abs(W) > self.threshold)
            self.n_iter_ += 1
            converged = np.array_equal(new_mask, mask)
            mask = new_mask
            if converged:
                break
        self.coef_ = (W * mask).reshape((h, n), order="F")  # forward() uses Xi * mask
        self.n_free_ = Q.shape[1]
        return self
