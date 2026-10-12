# %% [markdown]
# # A first attempt at automating symmetric constraints for SINDy

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pysindy as ps
import cupy
import scipy.linalg
import sympy as sp

# %% [markdown]
# Load data:

# %%
# read data
data = np.load("/gscratch/stf/songk42/ehd/sindy_data_fine_v2_t850.npz")

for x in data:
    print(x, data[x].shape)

# %% [markdown]
# $G$ is the set of transformation matrices s.t. the system follows the same dynamics under the transformation. That is, $\frac{d}{dt}(X g_k^T) = \frac{dX}{dt} g_k^T$ for transformation $g_k \in G$ ($X$'s entries are given as rows)

# %%
G = np.array([
    [[1, 0, 0], [0, -1, 0], [0, 0, 1]],  # [a1, a2, a3] -> [a1, -a2, a3]
    [[1, 0, 0], [0, 1, 0], [0, 0, -1]],  # [a1, a2, a3] -> [a1, a2, -a3]
    [[1, 0, 0], [0, -1, 0], [0, 0, -1]],  # [a1, a2, a3] -> [a1, -a2, -a3]
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],  # [a1, a2, a3] -> [-a1, a3, a2]
    [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]  # [a1, a2, a3] -> [-a1, -a3, -a2]
])

# convert matrices to sympy substitution rules
a1, a2, a3 = sp.symbols('a1 a2 a3')
A_symb = [a1, a2, a3]
G_subs_rules = []
for g in G:
    subs_rule = {a: sum(g[i, j] * A_symb[j] for j in range(3)) for i, a in enumerate(A_symb)}
    G_subs_rules.append(subs_rule)

# %% [markdown]
# The system is $\frac{dX}{dt} = \Theta(X) \Xi$, where $X$, $\Theta(X)$, and $\Xi$ are defined as in Guan et al. 2021.

# %%
# find POD modes
charge = data["u"][..., 1]          # (n_x, n_y, n_t) charge-density field only
n_x, n_y, n_t = charge.shape
snapshots = charge.reshape(n_x * n_y, n_t)   # each column = one time snapshot

# POD via GPU SVD: snapshots = Phi @ diag(sigma) @ Vt
centered_snapshots = np.asarray(
    snapshots - snapshots.mean(axis=1, keepdims=True),
    dtype=np.float64,
    order="C",
)
snapshots_cu = cupy.asarray(snapshots)
Phi, sigma, Vt = cupy.linalg.svd(snapshots_cu - snapshots_cu.mean(axis=1, keepdims=True), full_matrices=False)
# save phi, sigma, Vt for later use
path = "/mmfs1/home/songk42/symmetry-ode-discovery/notebooks/pod_modes.npz"
np.savez(path, Phi=Phi, sigma=sigma, Vt=Vt)

pod_data = np.load(path)
Phi = pod_data["Phi"]
sigma = pod_data["sigma"]
Vt = pod_data["Vt"]

r = 3  # PC1-PC3, as in the paper (PC0 is the mean field, dropped before SVD)
modes = Phi[:, :r]                  # spatial POD modes, reshape to (n_x, n_y, r) to plot

# %%
X = (sigma[:r, None] * Vt[:r, :]).T  # shape (n_t, r) == paper's eq. (4.1) X matrix, columns a1,a2,a3

# rescale POD amplitudes to O(1): raw values span ~[-45, 80], so degree-3 library terms
# would span ~5 orders of magnitude, making the true coefficients tiny (~1e-2) relative to
# any reasonable sparsity threshold. Uniform scaling commutes with every g_k in G (they're all
# signed permutation matrices), so it doesn't affect the symmetry constraints built below.
X_scale = np.max(np.abs(X))
X = X / X_scale

# Augment with symmetry-transformed copies of the trajectory (paper eq. 4.8): each
# X @ g.T is itself a valid (a1, a2, a3) trajectory under the enforced symmetry, so
# augmenting means more rows of the same 3-column data, not more columns/state
# variables -- np.concatenate(..., axis=-1) would turn 3 states into 18 and break
# every shape downstream (that's what caused the ValueError). Fit each copy as its
# own trajectory (a list) rather than concatenating along time, so FiniteDifference
# doesn't compute a bogus derivative across the seam between copies.
X_traj = [X] + [X @ g.T for g in G]
t_traj = [data["t"] for _ in X_traj]

Theta = ps.PolynomialLibrary(degree=3, include_interaction=True, include_bias=True)
Theta.fit(X)

# %% [markdown]
# ## Enforcing symmetry

# %% [markdown]
# If $g_k \in G$ is linear, then we can write $\Theta(X g_k^T) = \Theta(X) M_k$ for some matrix $M_k$. $M_k$ is defined by mapping each term in the original library $\Theta(X)$ to the transformed library $\Theta(X g_k^T)$ manually.

# %%
def build_Mk(library, subs_rule, vars_, monomial_index):
    """Convention: Theta(g x) = Theta(x) @ M_k, i.e. theta_j(g x) = sum_i Theta_x[i] * M[i, j].
    So each coefficient found while processing column j (theta_j) goes into M[i, j].
    """
    n = len(library)
    M = sp.zeros(n, n)
    for j, theta_j in enumerate(library):
        transformed = sp.expand(theta_j.subs(subs_rule, simultaneous=True))
        poly = sp.Poly(transformed, *vars_)
        for exps, coeff in poly.terms():
            i = monomial_index[exps]
            M[i, j] += coeff

    return M

full_library = Theta.get_feature_names(["a1", "a2", "a3"])
full_library = [sp.sympify(" * ".join(f.split(" "))) for f in full_library]
def exponent_tuple(monom, vars_):
    poly = sp.Poly(monom, *vars_)
    if poly.is_zero:
        return tuple([0] * len(vars_))
    return poly.monoms()[0]

# get indices corresponding to each term in the library
monomial_index = {}
for idx, m in enumerate(full_library):
    monomial_index[exponent_tuple(m, A_symb)] = idx

# %%
M_list = []

for subs_rule in G_subs_rules:
    M_k = build_Mk(full_library, subs_rule, A_symb, monomial_index)
    M_list.append(M_k)

# %% [markdown]
# Substituting into the original system, we get $\Theta(X g_k^T) \Xi = \Theta(X) M_k \Xi = \Theta(X) \Xi g_k^T$. This implies that $M_k \Xi = \Xi g_k^T$, which is a linear constraint on $\Xi$.
# 
# More specifically, in the $C\xi = d$ format this can be written as $(I \otimes M_k - g_k \otimes I) \text{vec}(\Xi) = 0$, where $\text{vec}(\Xi)$ is the vectorization of $\Xi$ (stacking columns).

# %%
C_list = []
for i in range(len(G)):
    M_numeric = np.array(M_list[i], dtype=float)
    C_list.append(np.kron(np.eye(G[i].shape[0]), M_numeric) - np.kron(G[i], np.eye(M_numeric.shape[0])))

# %% [markdown]
# Add the energy-preserving quadratic constraint (Loiseau & Brunton / Schlegel-Noack): ties
# the pure-quadratic term in one equation to the corresponding quadratic coupling term in
# another ($a_2^2$ in $a_1'$ <-> $a_1 a_2$ in $a_2'$), so the quadratic nonlinearity can't pump
# unbounded energy into $a_1$. Without this the symmetry constraints alone don't stop the
# cubic terms from picking an unstable sign combination -- e.g. a dense (reg_weight_lam=0) fit
# on this data landed on $a_1' = -2.786 a_1 + 1.886 a_1^3 + ...$, whose *positive* cubic term
# blows up to infinity in finite time once $a_1$ crosses its unstable fixed point.

# %%
row = np.zeros(len(A_symb) * len(full_library))
row[0 * len(full_library) + monomial_index[exponent_tuple(a2**2, A_symb)]] = 1.0
row[1 * len(full_library) + monomial_index[exponent_tuple(a1 * a2, A_symb)]] = 1.0
energy_lhs = row[None, :]

# %% [markdown]
# ## Solving system with SINDy

# %% [markdown]
# I ended up removing dependent constraints explicitly, because it looks like when concatenating all the $C$ matrices from above some of the rows are linearly dependent.

# %%
constraints_lhs = np.vstack(C_list + [energy_lhs])  # each row of each C matrix from above is its own constraint - is this fine?
constraints_rhs = np.zeros(constraints_lhs.shape[0])

# Remove redundant constraints by selecting independent rows of the augmented system:
# concatenating all C_list rows leaves many linearly dependent (300 rows for a 60-dim
# vec(Xi) here), which makes ConstrainedSR3's C @ inv1 @ C.T singular. Use QR with
# column pivoting on A.T to find independent rows of A (constraints_lhs).
A = constraints_lhs
b = constraints_rhs
rank = np.linalg.matrix_rank(A)
if rank == 0:
    # no independent constraints
    constraints_lhs = np.empty((0, A.shape[1]))
    constraints_rhs = np.empty((0,))
else:
    # qr on A.T identifies independent rows of A via pivot indices
    Q, R, piv = scipy.linalg.qr(A.T, pivoting=True)
    independent_row_idx = piv[:rank]
    constraints_lhs = A[independent_row_idx, :]
    constraints_rhs = b[independent_row_idx]

# drop any all-zero rows (safety)
mask = ~np.isclose(constraints_lhs, 0).all(axis=1)
constraints_lhs = constraints_lhs[mask]
constraints_rhs = constraints_rhs[mask]

# %%
differentiation_method = ps.FiniteDifference(order=2)
# optimizer = ps.ConstrainedSTLSQ(
#     constraint_lhs=constraints_lhs,
#     constraint_rhs=constraints_rhs,
#     constraint_order="target",
# )
optimizer = ps.ConstrainedSR3(
    constraint_lhs=constraints_lhs,
    constraint_rhs=constraints_rhs,
    constraint_order="target",
    equality_constraints=True,
    reg_weight_lam=0,
)
model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=Theta,
    optimizer=optimizer,
)

# %%
model.fit(X_traj, t=t_traj, feature_names=[a.name for a in A_symb])
model.print()

# %% [markdown]
# ## Verifying the symmetry constraints
# 
# Directly check $M_k \Xi = \Xi g_k^T$ for each $g_k \in G$ using the fitted $\Xi$. This is independent of pysindy's internal `constraint_order` convention — it re-derives the physical statement from scratch, so it also catches ordering/wiring bugs in how the constraints were passed to the optimizer.

# %%
Xi = model.coefficients().T  # (n_library, n_targets); model.coefficients() returns Xi^T

scale = np.max(np.abs(Xi)) or 1.0
print(f"max|Xi| = {scale:.3e}\n")
print(f"{'g_k':>4} | {'max|M_k Xi - Xi g_k^T|':>24} | {'relative to max|Xi|':>20}")
for k, (Mk, gk) in enumerate(zip(M_list, G)):
    Mk_num = np.array(Mk, dtype=float)
    resid = Mk_num @ Xi - Xi @ gk.T
    max_resid = np.max(np.abs(resid))
    print(f"{k:>4} | {max_resid:24.3e} | {max_resid / scale:20.3e}")

# sanity check: confirm this matches the constraints as actually passed to ConstrainedSR3
# (catches a mismatched `constraint_order` even if the check above somehow passed)
vec_Xi = Xi.flatten(order="F")  # stacking columns, per the derivation above
lhs_resid = np.max(np.abs(constraints_lhs @ vec_Xi - constraints_rhs))
print(f"\nmax|constraints_lhs @ vec(Xi) - constraints_rhs| = {lhs_resid:.3e}")

# %% [markdown]
# ## Reproducing the paper's figures
# 
# I did get the symmetries working (at least for this particular set of symmetires), but this is why I am not convinced that I am using the data correctly...
# 
# Figure 2(c,d): spatial POD modes and their amplitude time series.

# %%
import matplotlib.pyplot as plt

Xp = X * X_scale  # undo the rescaling for physically-meaningful amplitude plots

fig, axes = plt.subplots(r, 2, figsize=(9, 2.5 * r))
for i in range(r):
    spatial = modes[:, i].reshape(n_x, n_y)
    im = axes[i, 0].imshow(spatial.T, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[i, 0].set_title(f"POD spatial mode PC{i + 1}")
    axes[i, 0].axis("off")

    axes[i, 1].plot(data["t"], Xp[:, i], lw=0.4)
    axes[i, 1].set_title(f"amplitude a{i + 1}(t)")
    axes[i, 1].set_xlabel("t")

fig.tight_layout()

# %% [markdown]
# Figure 3(a,b): 3D phase portrait of $(a_1, a_2, a_3)$ and its 2D projections

# %%
fig = plt.figure(figsize=(14, 3.2))

ax3d = fig.add_subplot(1, 4, 1, projection="3d")
ax3d.plot(Xp[:, 0], Xp[:, 1], Xp[:, 2], lw=0.3)
ax3d.set_xlabel("a1")
ax3d.set_ylabel("a2")
ax3d.set_zlabel("a3")
ax3d.set_title("3D phase portrait")

labels = ["a1", "a2", "a3"]
for i, (p, q) in enumerate([(0, 1), (0, 2), (1, 2)]):
    ax = fig.add_subplot(1, 4, i + 2)
    ax.plot(Xp[:, p], Xp[:, q], lw=0.3)
    ax.set_xlabel(labels[p])
    ax.set_ylabel(labels[q])
    ax.set_title(f"{labels[p]}-{labels[q]} projection")

fig.tight_layout()

# %% [markdown]
# Figure 4(a): simulate the fitted SINDy model from the data's initial condition and compare its trajectory to the actual data — the real test of whether the model reproduces the system's behavior, not just the symmetry constraints.

# %%
n_sim = 5000  # full 10000-step horizon also runs, but the model damps out well before then
t_sim = data["t"][:n_sim]
X_sim = model.simulate(X[0], t_sim) * X_scale

fig = plt.figure(figsize=(9, 4))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot(Xp[:n_sim, 0], Xp[:n_sim, 1], Xp[:n_sim, 2], lw=0.3)
ax1.set_title("data")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.plot(X_sim[:, 0], X_sim[:, 1], X_sim[:, 2], lw=0.3, color="C1")
ax2.set_title("SINDy model simulation")

fig.tight_layout()

# %%



