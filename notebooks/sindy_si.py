"""SINDy-SI: Sparse Identification of Nonlinear Dynamics with Side Information.

Machado & Jones, arXiv:2310.04227v1. Model f_hat(x) = W^T phi(x) with polynomial features phi (h of them).

Algorithm 3 of the paper:
  1. W_1 = argmin ||Y - Phi W||_F^2 + xi2 ||W||_1  s.t. f_hat in side-information sets  (problem SI)
  2. repeat: E = {(i, j) : |w_ij| < lam}; if E has no new entries stop, else re-solve (SI) with w_ij = 0 on E  (Eq. 13)
Columns of Phi are normalized to unit l2 norm so a single lam is meaningful (Sec. III); the threshold acts on the
normalized coefficients.

Side information supported (everything that is linear in W, or SOS-representable on a ball, as in the paper):
  - linear equalities C vec(W) = d, vec in "target" order (column-major of the (h, n) W), e.g. symmetry
    constraints (I (x) M_k - g_k (x) I) vec(W) = 0
  - known equilibria f_hat(x_eq) = 0
  - coordinate-wise monotonicity s * d f_hat_i / d x_j >= 0 on the ball ||x - c|| <= r, enforced via Putinar:
    s * d f_hat_i/d x_j = z^T Q z + sigma_1(x) (r^2 - ||x - c||^2), Q >= 0, sigma_1 SOS

The paper writes the residual as an LMI (Schur complement on tr[M]); here it is passed to cvxpy as a sum of squares
of the QR-reduced residual, which is the same convex program. When only equalities are present (and xi2 = 0) the
problem is an equality-constrained least squares, solved exactly in the null space of the constraints (no SDP).
"""
import itertools

import cvxpy as cp
import numpy as np
import scipy.linalg


# ---------------------------------------------------------------- polynomial helpers
def monomials_upto(n, deg):
    """Exponent tuples of all monomials in n variables of total degree <= deg (graded order)."""
    out = []
    for d in range(deg + 1):
        for c in itertools.combinations_with_replacement(range(n), d):
            e = [0] * n
            for k in c:
                e[k] += 1
            out.append(tuple(e))
    return out


def derivative_map(powers, j):
    """d/dx_j of each library monomial, as {exponent: [(feature index, coefficient), ...]}.

    d f_hat_i / d x_j = sum_k W[k, i] * d phi_k / d x_j, so the result is linear in column i of W."""
    terms = {}
    for k, p in enumerate(powers):
        if p[j] == 0:
            continue
        e = list(p)
        e[j] -= 1
        terms.setdefault(tuple(e), []).append((k, float(p[j])))
    return terms


def sos_poly(z_exps, name_shape=None):
    """SOS polynomial z^T Q z with Q PSD. Returns (Q variable, {exponent: affine expression})."""
    m = len(z_exps)
    Q = cp.Variable((m, m), PSD=True)
    coeffs = {}
    for a in range(m):
        for b in range(m):
            e = tuple(x + y for x, y in zip(z_exps[a], z_exps[b]))
            coeffs[e] = coeffs.get(e, 0) + Q[a, b]
    return Q, coeffs


def ball_poly(n, center, radius):
    """g(x) = r^2 - ||x - c||^2 as {exponent: float}."""
    g = {tuple([0] * n): radius ** 2 - float(np.sum(np.asarray(center) ** 2))}
    for i in range(n):
        e1 = [0] * n
        e1[i] = 1
        e2 = [0] * n
        e2[i] = 2
        g[tuple(e1)] = 2.0 * center[i]
        g[tuple(e2)] = -1.0
    return g


def poly_mul(p, q):
    """Product of a {exponent: expr} with a {exponent: float}."""
    out = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e = tuple(x + y for x, y in zip(e1, e2))
            out[e] = out.get(e, 0) + c1 * c2
    return out


def monotone_sos_constraints(W, powers, i, j, sign, center, radius):
    """cvxpy constraints certifying sign * d f_hat_i / d x_j >= 0 on the ball (Putinar, degree-matched)."""
    n = powers.shape[1]
    dmap = derivative_map(powers, j)
    deg = max((sum(e) for e in dmap), default=0)
    half = (deg + 1) // 2  # z has degree ceil(deg/2); multiplier has degree 2*half - 2
    Q0, s0 = sos_poly(monomials_upto(n, half))
    cons = [Q0 >> 0]
    rhs = dict(s0)
    if half >= 1:
        Q1, s1 = sos_poly(monomials_upto(n, half - 1))
        cons.append(Q1 >> 0)
        for e, c in poly_mul(s1, ball_poly(n, center, radius)).items():
            rhs[e] = rhs.get(e, 0) + c
    lhs = {e: sign * sum(c * W[k, i] for k, c in terms) for e, terms in dmap.items()}
    for e in set(lhs) | set(rhs):
        cons.append(lhs.get(e, 0) == rhs.get(e, 0))
    return cons


def certify_monotone(Xi, powers, i, j, sign, center, radius):
    """True iff the fixed coefficient matrix Xi admits the SOS certificate (used to decide which side info is true)."""
    W = cp.Parameter(Xi.shape, value=Xi)
    cons = monotone_sos_constraints(W, powers, i, j, sign, center, radius)
    prob = cp.Problem(cp.Minimize(0), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return False
    return prob.status == cp.OPTIMAL


# ---------------------------------------------------------------- estimator
class SINDySI:
    """Sequentially-thresholded, side-information-constrained polynomial regression.

    Parameters
    ----------
    threshold : float      lam, applied to |w_ij| of the column-normalized problem
    xi2 : float            l1 weight (0 in the paper's Table I)
    max_iter : int         N; the paper's bound is h*n
    normalize_columns : bool
    eq_lhs, eq_rhs :       linear equalities on vec(W) (target order)
    equilibria : list      points x_eq with f_hat(x_eq) = 0
    monotone : list        (i, j, sign) with sign * d f_hat_i/d x_j >= 0 on the ball
    ball : (center, radius)
    """

    def __init__(self, threshold=0.1, xi2=0.0, max_iter=None, normalize_columns=True, eq_lhs=None, eq_rhs=None,
                 equilibria=(), monotone=(), ball=None, solver="CLARABEL"):
        self.threshold = threshold
        self.xi2 = xi2
        self.max_iter = max_iter
        self.normalize_columns = normalize_columns
        self.eq_lhs, self.eq_rhs = eq_lhs, eq_rhs
        self.equilibria = list(equilibria)
        self.monotone = list(monotone)
        self.ball = ball
        self.solver = solver

    # Linear equalities on vec(W) (target order), collected from symmetry constraints and equilibria.
    def _linear_equalities(self, h, n, powers):
        A, b = [], []
        if self.eq_lhs is not None and len(self.eq_lhs):
            A.append(np.asarray(self.eq_lhs, float))
            b.append(np.zeros(A[-1].shape[0]) if self.eq_rhs is None else np.asarray(self.eq_rhs, float))
        for x_eq in self.equilibria:
            phi = np.prod(np.asarray(x_eq, float)[None, :] ** powers, axis=1)  # (h,)
            A.append(np.kron(np.eye(n), phi[None, :]))  # row i: phi^T W[:, i]
            b.append(np.zeros(n))
        if not A:
            return np.zeros((0, h * n)), np.zeros(0)
        return np.vstack(A), np.concatenate(b)

    def _solve_equality_ls(self, R, QtY, A, b, active):
        """min ||R W - QtY||_F^2 s.t. A vec(W) = b, W = 0 off `active`; null-space lstsq (never forms R^T R)."""
        h, n = active.shape
        idx = np.flatnonzero(active.flatten(order="F"))
        big = np.kron(np.eye(n), R)[:, idx]  # vec(R W) = (I (x) R) vec(W)
        rhs = QtY.flatten(order="F")
        A_act = A[:, idx]
        if A_act.shape[0]:
            w_p, *_ = np.linalg.lstsq(A_act, b, rcond=None)
            if np.linalg.norm(A_act @ w_p - b) > 1e-8 * max(1.0, np.linalg.norm(b)):
                raise ValueError("side-information equalities infeasible on the current support")
            N = scipy.linalg.null_space(A_act)
            z, *_ = np.linalg.lstsq(big @ N, rhs - big @ w_p, rcond=None)
            w = w_p + N @ z
        else:
            w, *_ = np.linalg.lstsq(big, rhs, rcond=None)
        vec = np.zeros(h * n)
        vec[idx] = w
        return vec.reshape((h, n), order="F")

    def _solve_sdp(self, R, QtY, A, b, active, D, powers):
        h, n = active.shape
        Wt = cp.Variable((h, n))  # normalized coefficients
        W = cp.multiply(1.0 / D[:, None], Wt)  # coefficients of the un-normalized library
        cons = [Wt[~active] == 0] if (~active).any() else []
        if A.shape[0]:
            cons.append(A @ cp.vec(W, order="F") == b)
        center, radius = self.ball if self.ball is not None else (np.zeros(n), 1.0)
        for i, j, s in self.monotone:
            cons += monotone_sos_constraints(W, powers, i, j, s, center, radius)
        obj = cp.sum_squares(R @ Wt - QtY)
        if self.xi2:
            obj = obj + self.xi2 * cp.sum(cp.abs(Wt))
        prob = cp.Problem(cp.Minimize(obj), cons)
        for solver, opts in [(self.solver, {}), ("SCS", dict(eps=1e-9, max_iters=200000))]:
            try:
                prob.solve(solver=solver, **opts)
            except cp.SolverError:
                continue
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
        if Wt.value is None or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise ValueError(f"SDP failed: {prob.status}")
        out = np.array(Wt.value)
        out[~active] = 0.0
        return out

    def fit(self, Phi, Y, powers):
        """Phi: (m, h) library evaluated on data; Y: (m, n) derivative targets; powers: (h, n) exponents of Phi."""
        Phi, Y = np.asarray(Phi, float), np.asarray(Y, float)
        h, n = Phi.shape[1], Y.shape[1]
        D = np.linalg.norm(Phi, axis=0) if self.normalize_columns else np.ones(h)
        D[D == 0] = 1.0
        Qm, R = np.linalg.qr(Phi / D)  # ||Y - Phi~ W~||^2 = ||R W~ - Q^T Y||^2 + const
        QtY = Qm.T @ Y
        A, b = self._linear_equalities(h, n, powers)
        # equalities act on W = W~ / D; rewrite them on W~
        A_t = A * np.tile(1.0 / D, n)[None, :] if A.shape[0] else A
        use_sdp = bool(self.monotone) or self.xi2 > 0

        def solve(active):
            if use_sdp:
                return self._solve_sdp(R, QtY, A, b, active, D, powers)
            return self._solve_equality_ls(R, QtY, A_t, b, active)

        active = np.ones((h, n), bool)
        Wt = solve(active)
        self.history_ = [int(active.sum())]
        for _ in range((self.max_iter or h * n) - 1):
            new_active = active & (np.abs(Wt) >= self.threshold)
            if np.array_equal(new_active, active):
                break
            active = new_active
            Wt = solve(active)
            self.history_.append(int(active.sum()))
        self.coef_normalized_ = Wt
        self.coef_ = Wt / D[:, None]  # (h, n): f_hat = coef_^T phi
        self.active_ = active
        return self
