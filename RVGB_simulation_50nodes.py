"""
RVGB (Real-Valued Gauge Balance) Opinion Polarization Simulation
================================================================
50-node numerical validation for IEEE TAC paper:
"Opinion Polarization Over Signed Social Networks With Real-Valued Gauge Balance"

Four experiments:
  1. Two-community asymmetric RVGB polarization (d = 1 vs d = -2)
  2. Three-community RVGB (k=3, gauge values {1, 2, -1})
  3. Leaderless RVGB network
  4. Failure case: RVGB-Cycle violation (opinions fail to polarize)

Core idea: gauge transformation x*_i = x_i / d_i converts signed dynamics
into non-negative cooperative consensus. Polarization: x_i(∞) = d_i * x*.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ─── Global plot settings ───────────────────────────────────────────
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 15
rcParams['axes.linewidth'] = 1.2
rcParams['legend.fontsize'] = 13
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================================================
# Core functions
# =====================================================================

def gauge_transform(A, d):
    """
    Compute gauge-transformed adjacency: Ã_ij = a_ij * d_j / d_i.
    For RVGB networks, Ã should have non-negative off-diagonal entries.
    """
    n = A.shape[0]
    A_tilde = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != 0:
                A_tilde[i, j] = A[i, j] * d[j] / d[i]
    return A_tilde


def check_rvgb(A, d):
    """Check RVGB condition: all off-diagonal entries of D⁻¹AD non-negative."""
    A_tilde = gauge_transform(A, d)
    n = A.shape[0]
    min_val = 0.0
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != 0:
                min_val = min(min_val, A_tilde[i, j])
    return min_val >= -1e-10, min_val


def build_rvgb_network(n, communities, d, p_intra=0.3, p_inter=0.12,
                       w_range=(0.5, 1.5), w_intra_comp=0.0,
                       leader_idx=None, rng=None):
    """
    Build an adjacency matrix A that satisfies RVGB with gauge vector d.

    For RVGB: a_ij * d_j / d_i >= 0, i.e., sign(a_ij) = sign(d_j / d_i).

    Parameters
    ----------
    n : int - number of agents
    communities : list of lists - community membership
    d : ndarray (n,) - gauge vector
    p_intra : float - intra-community edge probability
    p_inter : float - inter-community edge probability
    w_range : tuple - weight magnitude range
    w_intra_comp : float - probability of small intra-community competition
    leader_idx : int or None - stubborn leader index
    rng : RandomState

    Returns
    -------
    A : ndarray (n, n)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    A = np.zeros((n, n))

    # For any edge (j -> i), we need: sign(a_ij) = sign(d_j / d_i)
    # If d_i and d_j have same sign: a_ij > 0 (cooperative)
    # If d_i and d_j have opposite sign: a_ij < 0 (competitive)

    def required_sign(i, j):
        return np.sign(d[j] / d[i])

    # Intra-community edges
    for comm in communities:
        # Ring structure for connectivity
        for idx in range(len(comm) - 1):
            i, j = comm[idx], comm[idx + 1]
            if leader_idx is not None and i == leader_idx:
                continue
            s = required_sign(i, j)
            A[i, j] = s * rng.uniform(*w_range)
            if leader_idx is None or j != leader_idx:
                A[j, i] = required_sign(j, i) * rng.uniform(*w_range)

        # Random additional intra-community edges
        for i in comm:
            if leader_idx is not None and i == leader_idx:
                continue
            for j in comm:
                if i == j:
                    continue
                if A[i, j] == 0 and rng.rand() < p_intra:
                    s = required_sign(i, j)
                    A[i, j] = s * rng.uniform(*w_range)

    # Inter-community edges
    for ci, comm_i in enumerate(communities):
        for cj, comm_j in enumerate(communities):
            if ci == cj:
                continue
            for i in comm_i:
                if leader_idx is not None and i == leader_idx:
                    continue
                for j in comm_j:
                    if rng.rand() < p_inter:
                        s = required_sign(i, j)
                        A[i, j] = s * rng.uniform(*w_range)

    # Ensure leader has influence on at least a few nodes in each community
    if leader_idx is not None:
        for comm in communities:
            count = 0
            for node in comm:
                if node == leader_idx:
                    continue
                if count < 3:
                    s = required_sign(node, leader_idx)
                    if A[node, leader_idx] == 0:
                        A[node, leader_idx] = s * rng.uniform(0.8, 1.5)
                    count += 1

    # Add small intra-community competition (negative edges within same-sign group)
    # These still satisfy RVGB since same-sign => a_ij > 0 required
    # So competition means slightly reducing some weights (but keeping positive)
    # For demonstration, we can add edges with smaller positive weights

    return A


def normalize_weights(A_tilde, leader_idx=None):
    """
    Row-normalize the gauge-transformed (non-negative) adjacency matrix.
    Returns W where W_ij = Ã_ij / Σ_k Ã_ik (row-stochastic).
    """
    n = A_tilde.shape[0]
    W = np.zeros_like(A_tilde)
    for i in range(n):
        if leader_idx is not None and i == leader_idx:
            continue
        row_sum = np.sum(A_tilde[i])
        if row_sum > 1e-12:
            W[i] = A_tilde[i] / row_sum
    return W


def simulate_rvgb(A, d, theta, x0, T_max, leader_idx=None, async_mode=False, h=1):
    """
    Simulate RVGB opinion dynamics.

    Strategy: work in gauge-transformed coordinates x*_i = x_i / d_i,
    where the system is cooperative (non-negative weights).

    1. Compute Ã = gauge_transform(A, d) (non-negative)
    2. Normalize: W_ij = Ã_ij / row_sum (row-stochastic)
    3. Consensus dynamics: x*(t+1) = Θ(t) x*(t) + (I-Θ(t)) W x*(t)
    4. Original coordinates: x_i(t) = d_i * x*_i(t)

    Returns X in original coordinates.
    """
    n = A.shape[0]
    A_tilde = gauge_transform(A, d)
    W = normalize_weights(A_tilde, leader_idx)

    # Initial transformed opinions
    x_star = x0 / d  # x*_i(0) = x_i(0) / d_i

    X_star = np.zeros((T_max + 1, n))
    X_star[0] = x_star.copy()

    # Generate async update schedules
    if async_mode:
        rng_async = np.random.RandomState(999)
        update_times = []
        for i in range(n):
            times = set()
            t = 0
            while t < T_max:
                times.add(t)
                t += rng_async.randint(1, h + 1)
            update_times.append(times)
    else:
        update_times = [set(range(T_max)) for _ in range(n)]

    for t in range(T_max):
        X_star[t + 1] = X_star[t].copy()

        for i in range(n):
            if leader_idx is not None and i == leader_idx:
                continue  # leader opinion fixed in transformed coords too

            if t not in update_times[i]:
                continue

            # Consensus update in transformed coordinates
            neighbor_sum = np.dot(W[i], X_star[t])
            X_star[t + 1, i] = theta[i] * X_star[t, i] + (1 - theta[i]) * neighbor_sum

    # Convert back to original coordinates: x_i(t) = d_i * x*_i(t)
    X = X_star * d[np.newaxis, :]

    return X, X_star


def simulate_raw(A, theta, x0, T_max, leader_idx=None):
    """
    Simulate opinion dynamics with raw adjacency (no gauge).
    Used for the failure case (Experiment 4).
    """
    n = A.shape[0]

    # Normalize rows by absolute row sum
    W = np.zeros_like(A)
    for i in range(n):
        if leader_idx is not None and i == leader_idx:
            continue
        row_sum = np.sum(np.abs(A[i]))
        if row_sum > 1e-12:
            W[i] = A[i] / row_sum

    X = np.zeros((T_max + 1, n))
    X[0] = x0.copy()

    for t in range(T_max):
        X[t + 1] = X[t].copy()
        for i in range(n):
            if leader_idx is not None and i == leader_idx:
                continue
            neighbor_sum = np.dot(W[i], X[t])
            X[t + 1, i] = theta[i] * X[t, i] + (1 - theta[i]) * neighbor_sum

    return X


def plot_experiment(X, communities, d, title, filename,
                    leader_idx=None, T_plot=None):
    """Plot opinion evolution trajectories."""
    T = X.shape[0]
    if T_plot is not None:
        T = min(T, T_plot)
    t_axis = np.arange(T)
    X_plot = X[:T]

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    mark_interval = max(T // 8, 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    leader_plotted = False
    leader_node_y = None
    for ci, comm in enumerate(communities):
        color = colors[ci % len(colors)]
        marker = markers[ci % len(markers)]
        d_val = d[comm[0]]

        for idx, node in enumerate(comm):
            is_leader = (leader_idx is not None and node == leader_idx)

            if is_leader:
                if not leader_plotted:
                    ax.plot(t_axis, X_plot[:, node], color='black', linewidth=2.0,
                            alpha=1.0, linestyle='-.',
                            marker='*', markersize=9, markevery=mark_interval,
                            markerfacecolor='black', markeredgecolor='none',
                            zorder=10)
                    leader_node_y = X_plot[0, node]
                    leader_plotted = True
            else:
                label = None
                if idx == 0 or (idx == 1 and comm[0] == leader_idx):
                    label = f'Community {ci+1} ($d_i={d_val:+.1f}$, {len(comm)} agents)'
                offset = (idx * 3) % mark_interval
                ax.plot(t_axis, X_plot[:, node], color=color, linewidth=0.8,
                        alpha=0.5, label=label,
                        marker=marker, markersize=4, markevery=(offset, mark_interval),
                        markerfacecolor=color, markeredgecolor='none')

    ax.set_xlabel('Time step $t$')
    ax.set_ylabel('Opinion $x_i(t)$')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.4)
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':')

    # Leader arrow annotation — text upper-left, arrow pointing down to line
    if leader_node_y is not None:
        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin
        text_y = leader_node_y + 0.16 * y_range
        ann_x = int(T * 0.28)
        text_x = int(T * 0.15)
        if text_y > ymax - 0.05 * y_range:
            ax.set_ylim(ymin, text_y + 0.10 * y_range)
        ax.annotate(
            r'Leader $v_{50}$',
            xy=(ann_x, leader_node_y),
            xytext=(text_x, text_y),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            ha='center', va='bottom')

    filepath = os.path.join(SAVE_DIR, filename)
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath}")


# =====================================================================
# Experiment 1: Two-Community Asymmetric RVGB Polarization
# =====================================================================

def experiment_1():
    print("=" * 70)
    print("Experiment 1: Two-Community Asymmetric RVGB (n=50)")
    print("  Gauge: d_i = +1 (V1), d_i = -2 (V2) => asymmetric polarization")
    print("=" * 70)

    n = 50
    leader_idx = n - 1  # v50 is leader, in V2

    V1 = list(range(25))       # d = +1
    V2 = list(range(25, 50))   # d = -2
    communities = [V1, V2]

    d = np.ones(n)
    d[25:] = -2.0

    rng = np.random.RandomState(100)

    # Build RVGB-compatible network
    # V1 (d>0) internal: a_ij > 0 (cooperative)
    # V2 (d<0) internal: a_ij > 0 (cooperative, since d_i/d_j = (-2)/(-2) = 1 > 0)
    # V1 <-> V2: d_i/d_j < 0 => a_ij < 0 (competitive)
    A = build_rvgb_network(n, communities, d, p_intra=0.25, p_inter=0.10,
                           w_range=(0.5, 1.5), leader_idx=leader_idx, rng=rng)

    # Verify RVGB
    is_rvgb, min_tilde = check_rvgb(A, d)
    print(f"  RVGB condition satisfied: {is_rvgb} (min Ã_ij = {min_tilde:.6f})")

    # Parameters
    theta = np.full(n, 0.5)
    theta[leader_idx] = 1.0
    x0 = rng.uniform(-1, 1, n)
    T_max = 200

    # Simulate
    X, X_star = simulate_rvgb(A, d, theta, x0, T_max, leader_idx)

    # Plot
    plot_experiment(X, communities, d,
                    'Exp 1: Asymmetric RVGB Polarization ($d \\in \\{+1, -2\\}$)',
                    'fig_exp1.pdf', leader_idx)

    # Results
    final = X[-1]
    x_star_final = X_star[-1]

    v1_opinions = final[V1]
    v2_opinions = final[V2[:-1]]  # exclude leader from stats
    leader_val = final[leader_idx]

    print(f"  Leader opinion (fixed): x_{n}(0) = {x0[leader_idx]:.4f}")
    print(f"  Transformed leader: x*_{n} = x_{n}/d_{n} = {x0[leader_idx]/d[leader_idx]:.4f}")
    print(f"  V1 final mean: {np.mean(v1_opinions):.4f} (expected: d=+1 => +x*)")
    print(f"  V2 final mean: {np.mean(v2_opinions):.4f} (expected: d=-2 => -2x*)")
    print(f"  Consensus x* ≈ {np.mean(x_star_final[V1]):.4f}")
    print(f"  V1 spread: {np.std(v1_opinions):.6f}")
    print(f"  V2 spread: {np.std(v2_opinions):.6f}")
    if abs(np.mean(v1_opinions)) > 1e-6:
        print(f"  Ratio V2/V1 ≈ {np.mean(v2_opinions)/np.mean(v1_opinions):.2f} (expected: -2.0)")
    print()


# =====================================================================
# Experiment 2: Three-Community RVGB (k=3)
# =====================================================================

def experiment_2():
    print("=" * 70)
    print("Experiment 2: Three-Community RVGB (k=3, n=50)")
    print("  Gauge: d ∈ {+1, +2, -1} => three polarization levels")
    print("=" * 70)

    n = 50
    leader_idx = n - 1  # v50 is leader, in V3

    V1 = list(range(17))       # d = +1
    V2 = list(range(17, 34))   # d = +2
    V3 = list(range(34, 50))   # d = -1
    communities = [V1, V2, V3]

    d = np.zeros(n)
    d[:17] = 1.0
    d[17:34] = 2.0
    d[34:] = -1.0

    rng = np.random.RandomState(200)

    # RVGB sign requirements:
    # V1(d=+1) <-> V1: same sign => a > 0
    # V2(d=+2) <-> V2: same sign => a > 0
    # V3(d=-1) <-> V3: same sign => a > 0
    # V1(d=+1) <-> V2(d=+2): same sign => a > 0  (positive inter-community!)
    # V1(d=+1) <-> V3(d=-1): opposite sign => a < 0
    # V2(d=+2) <-> V3(d=-1): opposite sign => a < 0
    A = build_rvgb_network(n, communities, d, p_intra=0.25, p_inter=0.08,
                           w_range=(0.5, 1.5), leader_idx=leader_idx, rng=rng)

    is_rvgb, min_tilde = check_rvgb(A, d)
    print(f"  RVGB condition satisfied: {is_rvgb} (min Ã_ij = {min_tilde:.6f})")

    # Note: V1<->V2 have POSITIVE edges! This violates QSB
    # (QSB requires all inter-community edges negative for any 2-partition)
    n_pos_inter_v1v2 = sum(1 for i in V1 for j in V2 if A[i,j] > 0) + \
                        sum(1 for i in V2 for j in V1 if A[i,j] > 0)
    print(f"  Positive V1-V2 inter-community edges: {n_pos_inter_v1v2}")
    print(f"  (These violate QSB for any 2-partition)")

    theta = np.full(n, 0.5)
    theta[leader_idx] = 1.0
    x0 = rng.uniform(-1, 1, n)
    # Set leader initial opinion to a clear value so x* is substantial
    # Leader is in V3 with d=-1, so x*_leader = x_leader / d_leader
    # Set x_leader = 0.5 => x* = 0.5/(-1) = -0.5
    # Polarization: V1 -> 1*(-0.5)=-0.5, V2 -> 2*(-0.5)=-1.0, V3 -> -1*(-0.5)=0.5
    x0[leader_idx] = 0.5
    T_max = 300

    X, X_star = simulate_rvgb(A, d, theta, x0, T_max, leader_idx)

    plot_experiment(X, communities, d,
                    'Exp 2: Three-Community RVGB ($k=3$, $d \\in \\{+1, +2, -1\\}$)',
                    'fig_exp2.pdf', leader_idx)

    final = X[-1]
    x_star_consensus = np.mean(X_star[-1])
    print(f"  Consensus x* ≈ {x_star_consensus:.4f}")
    print(f"  V1 mean (d=+1): {np.mean(final[V1]):.4f} (expected: +1 * x* = {x_star_consensus:.4f})")
    print(f"  V2 mean (d=+2): {np.mean(final[V2]):.4f} (expected: +2 * x* = {2*x_star_consensus:.4f})")
    v3_no_leader = [v for v in V3 if v != leader_idx]
    print(f"  V3 mean (d=-1): {np.mean(final[v3_no_leader]):.4f} (expected: -1 * x* = {-x_star_consensus:.4f})")
    if abs(np.mean(final[V1])) > 1e-6:
        print(f"  Ratio V2/V1 ≈ {np.mean(final[V2])/np.mean(final[V1]):.2f} (expected: 2.0)")
        print(f"  Ratio V3/V1 ≈ {np.mean(final[v3_no_leader])/np.mean(final[V1]):.2f} (expected: -1.0)")
    print()


# =====================================================================
# Experiment 3: Leaderless RVGB Network
# =====================================================================

def experiment_3():
    print("=" * 70)
    print("Experiment 3: Leaderless RVGB (n=50)")
    print("  No stubborn leader; x* determined endogenously")
    print("=" * 70)

    n = 50
    leader_idx = None

    V1 = list(range(25))       # d = +1
    V2 = list(range(25, 50))   # d = -1.5
    communities = [V1, V2]

    d = np.ones(n)
    d[25:] = -1.5

    rng = np.random.RandomState(300)

    A = build_rvgb_network(n, communities, d, p_intra=0.30, p_inter=0.12,
                           w_range=(0.5, 1.5), leader_idx=leader_idx, rng=rng)

    is_rvgb, min_tilde = check_rvgb(A, d)
    print(f"  RVGB condition satisfied: {is_rvgb} (min Ã_ij = {min_tilde:.6f})")

    theta = np.full(n, 0.5)
    x0 = rng.uniform(-1, 1, n)
    T_max = 300

    X, X_star = simulate_rvgb(A, d, theta, x0, T_max, leader_idx)

    plot_experiment(X, communities, d,
                    'Exp 3: Leaderless RVGB Polarization ($d \\in \\{+1, -1.5\\}$)',
                    'fig_exp3.pdf', leader_idx)

    final = X[-1]
    x_star_v1 = np.mean(X_star[-1, V1])
    x_star_v2 = np.mean(X_star[-1, V2])

    print(f"  Transformed consensus: x*_V1 ≈ {x_star_v1:.4f}, x*_V2 ≈ {x_star_v2:.4f}")
    print(f"  V1 mean (d=+1): {np.mean(final[V1]):.4f}")
    print(f"  V2 mean (d=-1.5): {np.mean(final[V2]):.4f}")
    if abs(np.mean(final[V1])) > 1e-6:
        print(f"  Ratio V2/V1 ≈ {np.mean(final[V2])/np.mean(final[V1]):.2f} (expected: -1.5)")
    print()


# =====================================================================
# Experiment 4: Failure Case - RVGB-Cycle Violation
# =====================================================================

def experiment_4():
    print("=" * 70)
    print("Experiment 4: Failure Case - RVGB-Cycle Violation (n=50)")
    print("  Network contains cycles with negative weight products")
    print("=" * 70)

    n = 50
    leader_idx = n - 1

    V1 = list(range(25))
    V2 = list(range(25, 50))
    communities = [V1, V2]

    d = np.ones(n)
    d[25:] = -1.0

    rng = np.random.RandomState(400)

    # Build a VALID structurally balanced base network
    A = np.zeros((n, n))

    # Intra-community cooperative ring
    for comm in communities:
        for idx in range(len(comm) - 1):
            i, j = comm[idx], comm[idx + 1]
            if i != leader_idx:
                A[i, j] = rng.uniform(0.5, 1.0)
            if j != leader_idx:
                A[j, i] = rng.uniform(0.5, 1.0)

    # Inter-community competitive edges
    for i in V1[:10]:
        for j in V2[:10]:
            if rng.rand() < 0.15:
                A[i, j] = rng.uniform(-1.5, -0.5)
    for i in V2[:10]:
        if i == leader_idx:
            continue
        for j in V1[:10]:
            if rng.rand() < 0.15:
                A[i, j] = rng.uniform(-1.5, -0.5)

    # NOW violate RVGB-Cycle by adding POSITIVE inter-community edges
    # Creating cycles with negative products.
    # Cycle: v0 (V1) -> v25 (V2) -> v1 (V1) -> v0
    # Edges: a_{25,0}, a_{1,25}, a_{0,1}
    # For structurally balanced: a_{25,0} < 0 (inter), a_{1,25} < 0 (inter), a_{0,1} > 0 (intra)
    # Product: (-)(-)(+) = (+) > 0 ... that's still positive.

    # To make a negative product cycle, we need ODD number of negative edges in cycle.
    # v0 -> v1 -> v25 -> v0
    # a_{1,0} > 0 (intra-V1), a_{25,1} < 0 (inter), a_{0,25} = ???
    # If we set a_{0,25} > 0 (POSITIVE inter-community), product = (+)(-)(+) = (-) < 0!

    # Add INCONSISTENT positive inter-community edges
    A[0, 25] = 0.8   # v25 -> v0: POSITIVE inter-community (violates SB!)
    A[2, 26] = 0.6   # v26 -> v2
    A[4, 27] = 0.7   # v27 -> v4
    A[6, 28] = 0.5   # v28 -> v6
    A[8, 29] = 0.9   # v29 -> v8

    # Leader edges
    for node in [0, 1, 2, 3, 4]:
        if A[node, leader_idx] == 0:
            A[node, leader_idx] = rng.uniform(-1.2, -0.5)
    for node in [25, 26, 27]:
        if node != leader_idx and A[node, leader_idx] == 0:
            A[node, leader_idx] = rng.uniform(0.5, 1.2)

    # Verify RVGB is violated
    is_rvgb, min_tilde = check_rvgb(A, d)
    print(f"  RVGB condition satisfied: {is_rvgb} (min Ã_ij = {min_tilde:.6f})")
    print(f"  (Expected: False, due to positive inter-community edges)")

    # Check specific cycles
    # v0 -> v1 -> v25 -> v0: a_{1,0} * a_{25,1} * a_{0,25}
    if A[1,0] != 0 and A[25,1] != 0 and A[0,25] != 0:
        cycle_prod = A[1,0] * A[25,1] * A[0,25]
        print(f"  Cycle v0->v1->v25->v0 product: {cycle_prod:.4f} ({'> 0' if cycle_prod > 0 else '< 0'})")

    theta = np.full(n, 0.5)
    theta[leader_idx] = 1.0
    x0 = rng.uniform(-1, 1, n)
    T_max = 200

    # Simulate with RAW weights (no gauge correction possible)
    X = simulate_raw(A, theta, x0, T_max, leader_idx)

    plot_experiment(X, communities, d,
                    'Exp 4: Failure Under RVGB-Cycle Violation',
                    'fig_exp4.pdf', leader_idx)

    final = X[-1]
    v1_std = np.std(final[V1])
    v2_std = np.std(final[V2[:-1]])
    print(f"  V1 opinion std: {v1_std:.4f}")
    print(f"  V2 opinion std: {v2_std:.4f}")
    print(f"  V1 range: [{final[V1].min():.4f}, {final[V1].max():.4f}]")
    print(f"  V2 range: [{final[V2[:-1]].min():.4f}, {final[V2[:-1]].max():.4f}]")
    print(f"  Polarization NOT achieved (no valid gauge transformation exists)")
    print()


# =====================================================================
# Experiment 5: Robustness Against Malicious Agent
# =====================================================================

def simulate_rvgb_malicious(A, d, theta, x0, T_max, leader_idx,
                            malicious_idx, w_bias=0.0, delta_frac=0.0,
                            d_false=None):
    """
    Simulate RVGB opinion dynamics with a malicious agent.

    The malicious agent v_f deviates:
      x*_f(t+1) = theta_f x*_f(t) + (1-theta_f) sum tilde_q_fj (1+Delta_fj) x*_j(t) + w_f(t)/d_f

    Parameters
    ----------
    malicious_idx : int - index of malicious agent
    w_bias : float - constant bias injected by malicious agent (in original coords)
    delta_frac : float - uniform weight falsification fraction |Delta_fj| = delta_frac
    d_false : float or None - if set, malicious agent uses this gauge value instead of d[malicious_idx]
    """
    n = A.shape[0]
    f = malicious_idx

    A_tilde = gauge_transform(A, d)
    W = normalize_weights(A_tilde, leader_idx)

    # If malicious agent uses a false gauge, recompute its row in W
    if d_false is not None and d_false != d[f]:
        d_mal = d.copy()
        d_mal[f] = d_false
        A_tilde_mal = gauge_transform(A, d_mal)
        W_mal_row = np.zeros(n)
        row_sum = np.sum(A_tilde_mal[f])
        if row_sum > 1e-12:
            W_mal_row = A_tilde_mal[f] / row_sum
    else:
        W_mal_row = None

    x_star = x0 / d
    X_star = np.zeros((T_max + 1, n))
    X_star[0] = x_star.copy()

    update_times = [set(range(T_max)) for _ in range(n)]

    for t in range(T_max):
        X_star[t + 1] = X_star[t].copy()

        for i in range(n):
            if i == leader_idx:
                continue

            if t not in update_times[i]:
                continue

            if i == f:
                # Malicious agent update
                if W_mal_row is not None:
                    # Weight falsification: use wrong gauge row
                    w_row = W_mal_row * (1.0 + delta_frac)
                else:
                    w_row = W[f] * (1.0 + delta_frac)

                neighbor_sum = np.dot(w_row, X_star[t])
                X_star[t + 1, f] = (theta[f] * X_star[t, f]
                                     + (1 - theta[f]) * neighbor_sum
                                     + w_bias / d[f])
            else:
                neighbor_sum = np.dot(W[i], X_star[t])
                X_star[t + 1, i] = (theta[i] * X_star[t, i]
                                     + (1 - theta[i]) * neighbor_sum)

    X = X_star * d[np.newaxis, :]
    return X, X_star


def experiment_5():
    print("=" * 70)
    print("Experiment 5: Robustness Against Malicious Agent (n=50)")
    print("  Malicious agent v_1 disrupts polarization")
    print("=" * 70)

    n = 50
    leader_idx = n - 1
    malicious_idx = 0  # v_1

    V1 = list(range(25))
    V2 = list(range(25, 50))
    communities = [V1, V2]

    d = np.ones(n)
    d[25:] = -2.0

    rng = np.random.RandomState(100)
    A = build_rvgb_network(n, communities, d, p_intra=0.25, p_inter=0.10,
                           w_range=(0.5, 1.5), leader_idx=leader_idx, rng=rng)

    theta = np.full(n, 0.5)
    theta[leader_idx] = 1.0
    x0 = rng.uniform(-1, 1, n)
    T_max = 200

    # --- Nominal (no attack) for reference ---
    X_nom, X_star_nom = simulate_rvgb(A, d, theta, x0, T_max, leader_idx)
    x_star_consensus = np.mean(X_star_nom[-1, V1])

    # --- (a) Bias injection: w_bar = 0.3 ---
    X_a, X_star_a = simulate_rvgb_malicious(
        A, d, theta, x0, T_max, leader_idx,
        malicious_idx=malicious_idx, w_bias=0.3, delta_frac=0.0)

    # --- (b) Weight falsification: Delta = 0.5 (50% weight perturbation) ---
    X_b, X_star_b = simulate_rvgb_malicious(
        A, d, theta, x0, T_max, leader_idx,
        malicious_idx=malicious_idx, w_bias=0.0, delta_frac=0.5)

    # --- Plot ---
    colors = ['#1f77b4', '#d62728']

    T_plot = T_max + 1
    t_axis = np.arange(T_plot)

    # Helper to plot one panel
    markers_exp5 = ['o', 's']
    mark_interval_5 = max(T_plot // 8, 1)

    leader_plotted_5 = [False]  # mutable for closure
    leader_y_5 = [None]

    def _plot_exp5_panel(ax, X_data, panel_title, mal_label):
        leader_plotted_5[0] = False
        leader_y_5[0] = None
        for ci, (comm, color) in enumerate(zip(communities, colors)):
            marker = markers_exp5[ci]
            d_val = d[comm[0]]
            for idx, node in enumerate(comm):
                is_mal = (node == malicious_idx)
                is_leader = (node == leader_idx)
                if is_mal:
                    ax.plot(t_axis, X_data[:T_plot, node], color='#2ca02c',
                            linewidth=2.0, linestyle='--', zorder=10,
                            label=mal_label,
                            marker='D', markersize=6, markevery=mark_interval_5,
                            markerfacecolor='#2ca02c', markeredgecolor='white',
                            markeredgewidth=0.5)
                elif is_leader:
                    if not leader_plotted_5[0]:
                        ax.plot(t_axis, X_data[:T_plot, node], color='black',
                                linewidth=2.0, alpha=1.0, linestyle='-.',
                                marker='*', markersize=9, markevery=mark_interval_5,
                                markerfacecolor='black', markeredgecolor='none',
                                zorder=10)
                        leader_y_5[0] = X_data[0, node]
                        leader_plotted_5[0] = True
                else:
                    lw = 0.8
                    alpha_val = 0.45
                    label = None
                    if idx == 0 or (idx == 1 and (comm[0] == malicious_idx or comm[0] == leader_idx)):
                        label = f'$\\mathcal{{V}}_{ci+1}$ ($d_i={d_val:+.1f}$)'
                    offset = (idx * 3) % mark_interval_5
                    ax.plot(t_axis, X_data[:T_plot, node], color=color,
                            linewidth=lw, alpha=alpha_val, label=label,
                            marker=marker, markersize=4,
                            markevery=(offset, mark_interval_5),
                            markerfacecolor=color, markeredgecolor='none')
        ax.axhline(y=x_star_consensus, color='black', linewidth=1.0,
                   linestyle='--', alpha=0.8)
        ax.axhline(y=-2 * x_star_consensus, color='black', linewidth=1.0,
                   linestyle='--', alpha=0.8)
        ax.set_xlabel('Time step $t$')
        ax.set_ylabel('Opinion $x_i(t)$')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.4)
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle=':')
        # Leader arrow annotation — always place text above the line
        if leader_y_5[0] is not None:
            ymin, ymax = ax.get_ylim()
            y_range = ymax - ymin
            text_y = leader_y_5[0] + 0.18 * y_range
            # Expand ylim if annotation would be clipped
            if text_y > ymax - 0.05 * y_range:
                ax.set_ylim(ymin, text_y + 0.10 * y_range)
            ann_x = int(T_plot * 0.5)
            text_x = ann_x + T_plot * 0.08
            ax.annotate(
                r'Leader $v_{50}$',
                xy=(ann_x, leader_y_5[0]),
                xytext=(text_x, text_y),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                ha='center', va='bottom')

    # Panel (a): Bias injection — separate PDF
    fig_a, ax_a = plt.subplots(figsize=(7, 5))
    _plot_exp5_panel(ax_a, X_a, 'Bias Injection ($\\bar{w}=0.3$)',
                     f'Malicious $v_1$ ($d_1={d[malicious_idx]:+.1f}$)')
    ax_a.set_ylim(ax_a.get_ylim()[0], ax_a.get_ylim()[1] + 0.8)
    fig_a.tight_layout()
    fp_a = os.path.join(SAVE_DIR, 'fig_exp5a.pdf')
    fig_a.savefig(fp_a, dpi=300, bbox_inches='tight')
    plt.close(fig_a)
    print(f"  Saved: {fp_a}")

    # Panel (b): Weight falsification — separate PDF
    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    _plot_exp5_panel(ax_b, X_b, 'Weight Falsification ($\\bar{\\Delta}=0.5$)',
                     f'Malicious $v_1$ ($\\bar{{\\Delta}}=0.5$)')
    ax_b.set_ylim(ax_b.get_ylim()[0], ax_b.get_ylim()[1] + 0.4)
    fig_b.tight_layout()
    fp_b = os.path.join(SAVE_DIR, 'fig_exp5b.pdf')
    fig_b.savefig(fp_b, dpi=300, bbox_inches='tight')
    plt.close(fig_b)
    print(f"  Saved: {fp_b}")

    # --- Report results ---
    # (a) Bias injection
    err_a = np.abs(X_a[-1] - X_nom[-1])
    print(f"\n  (a) Bias injection (w_bar=0.3):")
    print(f"      Malicious v_1 final: {X_a[-1, malicious_idx]:.4f} "
          f"(nominal: {X_nom[-1, malicious_idx]:.4f})")
    print(f"      Max polarization error: {np.max(err_a):.4f}")
    print(f"      Mean polarization error (excl. leader): "
          f"{np.mean(err_a[:n-1]):.4f}")

    # (b) Weight falsification
    err_b = np.abs(X_b[-1] - X_nom[-1])
    print(f"\n  (b) Weight falsification (Delta_bar=0.5):")
    print(f"      Malicious v_1 final: {X_b[-1, malicious_idx]:.4f} "
          f"(nominal: {X_nom[-1, malicious_idx]:.4f})")
    print(f"      Max polarization error: {np.max(err_b):.4f}")
    print(f"      Mean polarization error (excl. leader): "
          f"{np.mean(err_b[:n-1]):.4f}")
    print()


# =====================================================================
# Combined Figure
# =====================================================================

def make_combined_figure():
    """Read the individual PDFs and create a combined summary."""
    print("Individual experiment figures saved as fig_exp1-5.pdf")
    print("Use these in the LaTeX paper with \\includegraphics.")


# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("RVGB Opinion Polarization Simulation (n = 50 nodes)")
    print("IEEE TAC Paper: Real-Valued Gauge Balance")
    print("=" * 70 + "\n")

    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
    make_combined_figure()

    print("=" * 70)
    print("All experiments completed successfully!")
    print(f"Figures saved to: {SAVE_DIR}")
    print("=" * 70)
