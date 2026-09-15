"""
RVGB Real-World Dataset Validation: Slashdot Zoo Signed Network
================================================================
Downloads the Slashdot081106 signed network from Stanford SNAP,
extracts a dense subgraph, detects communities, verifies RVGB
conditions, and runs opinion polarization dynamics.

Dataset: Slashdot Zoo friend/foe network (77,350 nodes, 516,575 edges)
Source:  https://snap.stanford.edu/data/soc-sign-Slashdot081106.html
Ref:    Leskovec, Huttenlocher, Kleinberg. WWW 2010.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import urllib.request
import ssl
import gzip
import shutil
from collections import defaultdict, deque

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
DATA_URL = "https://snap.stanford.edu/data/soc-sign-Slashdot081106.txt.gz"
DATA_FILE = os.path.join(SAVE_DIR, "soc-sign-Slashdot081106.txt.gz")
DATA_TXT = os.path.join(SAVE_DIR, "soc-sign-Slashdot081106.txt")


# =====================================================================
# Step 1: Download and load dataset
# =====================================================================

def download_dataset():
    """Download Slashdot dataset from SNAP if not already cached."""
    if os.path.exists(DATA_TXT):
        print(f"Dataset already exists: {DATA_TXT}")
        return
    if not os.path.exists(DATA_FILE):
        print(f"Downloading {DATA_URL} ...")
        # Bypass SSL verification for SNAP download
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx))
        with opener.open(DATA_URL) as response:
            with open(DATA_FILE, 'wb') as out_f:
                shutil.copyfileobj(response, out_f)
        print(f"Downloaded: {DATA_FILE}")
    # Decompress
    print("Decompressing...")
    with gzip.open(DATA_FILE, 'rb') as f_in:
        with open(DATA_TXT, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Decompressed: {DATA_TXT}")


def load_edge_list():
    """
    Load signed edge list.
    Returns: list of (src, dst, sign) tuples where sign in {+1, -1}.
    """
    edges = []
    with open(DATA_TXT, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                src, dst, sign = int(parts[0]), int(parts[1]), int(parts[2])
                edges.append((src, dst, sign))
    print(f"Loaded {len(edges)} edges")

    # Count positive/negative
    n_pos = sum(1 for _, _, s in edges if s > 0)
    n_neg = sum(1 for _, _, s in edges if s < 0)
    print(f"  Positive edges: {n_pos} ({100*n_pos/len(edges):.1f}%)")
    print(f"  Negative edges: {n_neg} ({100*n_neg/len(edges):.1f}%)")
    return edges


# =====================================================================
# Step 2: Extract dense subgraph
# =====================================================================

def build_adjacency_lists(edges):
    """Build adjacency lists from edge list."""
    out_neighbors = defaultdict(dict)  # out_neighbors[i][j] = sign
    in_neighbors = defaultdict(dict)
    nodes = set()
    for src, dst, sign in edges:
        out_neighbors[src][dst] = sign
        in_neighbors[dst][src] = sign
        nodes.add(src)
        nodes.add(dst)
    return out_neighbors, in_neighbors, nodes


def extract_dense_subgraph(edges, target_size=150, seed_strategy='high_neg_degree'):
    """
    Extract a dense strongly-connected subgraph of target_size nodes.
    Strategy: BFS from node with most negative edges (adversarial hub),
    preferring neighbors with high negative-edge density.
    This yields subgraphs more amenable to RVGB analysis.
    """
    out_neighbors, in_neighbors, all_nodes = build_adjacency_lists(edges)

    # Compute degrees
    degree = defaultdict(int)
    neg_degree = defaultdict(int)  # count of negative edges
    for src, dst, sign in edges:
        degree[src] += 1
        degree[dst] += 1
        if sign < 0:
            neg_degree[src] += 1
            neg_degree[dst] += 1

    # Score: prefer nodes with high negative-edge fraction and reasonable degree
    def node_score(v):
        d = degree[v]
        if d < 10:
            return 0
        return neg_degree[v] / d * np.sqrt(d)

    # Start BFS from node with highest adversarial score
    seed = max(all_nodes, key=node_score)
    print(f"Seed node: {seed} (degree={degree[seed]}, neg_frac={neg_degree[seed]/max(degree[seed],1):.2f})")

    # BFS preferring nodes with high negative-edge fraction
    visited = set()
    queue = deque([seed])
    visited.add(seed)
    collect_size = target_size * 3

    while queue and len(visited) < collect_size:
        node = queue.popleft()
        neighbors = list(out_neighbors[node].keys()) + list(in_neighbors[node].keys())
        # Sort by adversarial score (high neg fraction first)
        neighbors = sorted(set(neighbors), key=lambda x: -node_score(x))
        for nb in neighbors:
            if nb not in visited and len(visited) < collect_size:
                visited.add(nb)
                queue.append(nb)

    subgraph_nodes = visited
    print(f"BFS subgraph: {len(subgraph_nodes)} nodes")

    # Build sub-edges
    sub_edges = []
    for src, dst, sign in edges:
        if src in subgraph_nodes and dst in subgraph_nodes:
            sub_edges.append((src, dst, sign))

    # Find largest strongly connected component via Tarjan/Kosaraju
    scc = largest_scc(subgraph_nodes, sub_edges)
    print(f"Largest SCC: {len(scc)} nodes")

    # If SCC is too large, trim to target_size (keep highest-degree nodes)
    if len(scc) > target_size:
        scc_degrees = {n: degree[n] for n in scc}
        top_nodes = sorted(scc_degrees, key=scc_degrees.get, reverse=True)[:target_size]
        scc = set(top_nodes)
        # Re-check SCC within trimmed set
        trimmed_edges = [(s, d, sg) for s, d, sg in sub_edges if s in scc and d in scc]
        scc = largest_scc(scc, trimmed_edges)
        print(f"Trimmed SCC: {len(scc)} nodes")

    # Re-index nodes to 0..n-1
    scc_list = sorted(scc)
    node_map = {old: new for new, old in enumerate(scc_list)}
    n = len(scc_list)

    A = np.zeros((n, n))
    for src, dst, sign in edges:
        if src in node_map and dst in node_map:
            i, j = node_map[src], node_map[dst]
            if i != j:
                A[i, j] = sign  # +1 or -1

    print(f"Subgraph adjacency: {n} nodes, {int(np.count_nonzero(A))} directed edges")
    return A, n, scc_list, node_map


def largest_scc(nodes, edges):
    """Find largest strongly connected component using Kosaraju's algorithm."""
    adj = defaultdict(list)
    adj_rev = defaultdict(list)
    for src, dst, _ in edges:
        if src in nodes and dst in nodes:
            adj[src].append(dst)
            adj_rev[dst].append(src)

    # Pass 1: DFS on original graph, record finish order
    visited = set()
    finish_order = []

    def dfs1(v):
        stack = [(v, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                finish_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for nb in adj[node]:
                if nb not in visited:
                    stack.append((nb, False))

    for v in nodes:
        if v not in visited:
            dfs1(v)

    # Pass 2: DFS on reversed graph in reverse finish order
    visited2 = set()
    components = []

    def dfs2(v):
        component = []
        stack = [v]
        while stack:
            node = stack.pop()
            if node in visited2:
                continue
            visited2.add(node)
            component.append(node)
            for nb in adj_rev[node]:
                if nb not in visited2:
                    stack.append(nb)
        return component

    for v in reversed(finish_order):
        if v not in visited2:
            comp = dfs2(v)
            components.append(comp)

    # Return largest
    largest = max(components, key=len)
    return set(largest)


# =====================================================================
# Step 3: Community detection (spectral bipartitioning)
# =====================================================================

def sign_consistent_partition(A):
    """
    Partition nodes into two communities by sign-consistent BFS.
    Assigns communities so that positive edges connect same-community nodes
    and negative edges connect cross-community nodes — directly maximizing
    RVGB compatibility.

    Algorithm: BFS from highest-degree node; for each neighbor, assign to
    same community if edge is positive, opposite if negative.
    Conflicts are resolved by majority vote of already-assigned neighbors.
    """
    n = A.shape[0]
    A_sym = (A + A.T) / 2  # symmetrize for BFS

    # Start from highest-degree node
    degrees = np.sum(np.abs(A_sym), axis=1)
    seed = np.argmax(degrees)

    label = np.full(n, -1, dtype=int)  # -1 = unassigned
    label[seed] = 0

    queue = deque([seed])
    visited = {seed}

    while queue:
        node = queue.popleft()
        # Process all neighbors
        for j in range(n):
            if A_sym[node, j] == 0 or node == j:
                continue
            if j in visited:
                continue
            visited.add(j)
            # Assign based on edge sign
            if A_sym[node, j] > 0:
                label[j] = label[node]  # same community
            else:
                label[j] = 1 - label[node]  # opposite community
            queue.append(j)

    # Handle any unvisited nodes (assign by majority of signed neighbors)
    for i in range(n):
        if label[i] == -1:
            votes = [0, 0]
            for j in range(n):
                if A_sym[i, j] != 0 and label[j] >= 0:
                    if A_sym[i, j] > 0:
                        votes[label[j]] += abs(A_sym[i, j])
                    else:
                        votes[1 - label[j]] += abs(A_sym[i, j])
            label[i] = 0 if votes[0] >= votes[1] else 1

    V1 = [i for i in range(n) if label[i] == 0]
    V2 = [i for i in range(n) if label[i] == 1]

    # If extremely unbalanced, refine with spectral method
    min_size = max(n // 5, 5)
    if len(V1) < min_size or len(V2) < min_size:
        print(f"  Sign-BFS unbalanced ({len(V1)} vs {len(V2)}), using spectral refinement")
        return spectral_bipartition_signed(A)

    print(f"Community detection (sign-BFS): |V1|={len(V1)}, |V2|={len(V2)}")

    # Report partition quality
    intra_pos = sum(1 for i in V1 for j in V1 if i != j and A[i,j] > 0) + \
                sum(1 for i in V2 for j in V2 if i != j and A[i,j] > 0)
    inter_neg = sum(1 for i in V1 for j in V2 if A[i,j] < 0) + \
                sum(1 for i in V2 for j in V1 if A[i,j] < 0)
    total = int(np.count_nonzero(A))
    print(f"  Intra-community positive: {intra_pos}, Inter-community negative: {inter_neg}")
    print(f"  RVGB-compatible: {intra_pos + inter_neg}/{total} = "
          f"{100*(intra_pos + inter_neg)/total:.1f}%")

    return [V1, V2]


def spectral_bipartition_signed(A):
    """Fallback: signed Laplacian spectral bipartitioning."""
    n = A.shape[0]
    A_s = (A + A.T) / 2
    D_plus = np.diag(np.sum(np.abs(A_s), axis=1))
    L_signed = D_plus - A_s
    eigvals, eigvecs = np.linalg.eigh(L_signed)
    fiedler = eigvecs[:, 0]
    V1 = [i for i in range(n) if fiedler[i] >= 0]
    V2 = [i for i in range(n) if fiedler[i] < 0]
    if len(V1) == 0 or len(V2) == 0:
        mid = n // 2
        V1 = list(range(mid))
        V2 = list(range(mid, n))
    print(f"Community detection (spectral): |V1|={len(V1)}, |V2|={len(V2)}")
    return [V1, V2]


# =====================================================================
# Step 4: RVGB verification and gauge assignment
# =====================================================================

def assign_gauge_and_verify(A, communities, d_values=(1.0, -2.0)):
    """
    Assign gauge vector d based on communities, then verify RVGB condition.
    Returns: d, rvgb_satisfied_ratio, A_rvgb (with violating edges removed).
    """
    n = A.shape[0]
    d = np.zeros(n)
    for ci, comm in enumerate(communities):
        for node in comm:
            d[node] = d_values[ci]

    # Check RVGB condition: sgn(a_ij) == sgn(d_j / d_i)
    total_edges = 0
    satisfied = 0
    violated = 0
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != 0:
                total_edges += 1
                required_sign = np.sign(d[j] / d[i])
                actual_sign = np.sign(A[i, j])
                if actual_sign == required_sign:
                    satisfied += 1
                else:
                    violated += 1

    ratio = satisfied / total_edges if total_edges > 0 else 0
    print(f"RVGB verification: {satisfied}/{total_edges} edges satisfied "
          f"({100*ratio:.1f}%), {violated} violated")

    # Create RVGB-consistent adjacency by removing violating edges
    A_rvgb = A.copy()
    removed = 0
    for i in range(n):
        for j in range(n):
            if i != j and A_rvgb[i, j] != 0:
                required_sign = np.sign(d[j] / d[i])
                actual_sign = np.sign(A_rvgb[i, j])
                if actual_sign != required_sign:
                    A_rvgb[i, j] = 0
                    removed += 1

    print(f"Removed {removed} violating edges -> {int(np.count_nonzero(A_rvgb))} remaining")
    return d, ratio, A_rvgb


# =====================================================================
# Step 5: RVGB simulation (reuse core logic)
# =====================================================================

def gauge_transform(A, d):
    """Compute gauge-transformed adjacency: Ã_ij = a_ij * d_j / d_i."""
    n = A.shape[0]
    A_tilde = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != 0:
                A_tilde[i, j] = A[i, j] * d[j] / d[i]
    return A_tilde


def normalize_weights(A_tilde, leader_idx=None):
    """Row-normalize gauge-transformed adjacency to row-stochastic W."""
    n = A_tilde.shape[0]
    W = np.zeros_like(A_tilde)
    for i in range(n):
        if leader_idx is not None and i == leader_idx:
            continue
        row_sum = np.sum(A_tilde[i])
        if row_sum > 1e-12:
            W[i] = A_tilde[i] / row_sum
    return W


def check_rvgb(A, d):
    """Check RVGB condition: all off-diagonal entries of D⁻¹AD non-negative."""
    A_tilde = gauge_transform(A, d)
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != 0:
                if A_tilde[i, j] < -1e-10:
                    return False
    return True


def simulate_rvgb(A, d, theta, x0, T_max, leader_idx=None):
    """Simulate RVGB opinion dynamics in gauge-transformed coordinates."""
    n = A.shape[0]
    A_tilde = gauge_transform(A, d)
    W = normalize_weights(A_tilde, leader_idx)

    x_star = x0 / d
    X_star = np.zeros((T_max + 1, n))
    X_star[0] = x_star.copy()

    for t in range(T_max):
        X_star[t + 1] = X_star[t].copy()
        for i in range(n):
            if leader_idx is not None and i == leader_idx:
                continue
            neighbor_sum = np.dot(W[i], X_star[t])
            X_star[t + 1, i] = theta[i] * X_star[t, i] + (1 - theta[i]) * neighbor_sum

    X = X_star * d[np.newaxis, :]
    return X, X_star


# =====================================================================
# Step 6: Visualization
# =====================================================================

def plot_slashdot_experiment(A_orig, A_rvgb, X, communities, d,
                             n, rvgb_ratio, leader_idx=None,
                             filename='fig_exp6.pdf'):
    """Generate 3 separate subfigure PDFs for the Slashdot experiment."""
    colors_comm = ['#1f77b4', '#d62728']
    markers_comm = ['o', 's']
    T = X.shape[0]
    t_axis = np.arange(T)
    T_plot = min(T, 300)

    # ── Compute layout (shared by panel a) ──
    A_sym = (np.abs(A_rvgb) + np.abs(A_rvgb.T)) / 2
    D_diag = A_sym.sum(axis=1)
    D_diag[D_diag == 0] = 1
    L_norm = np.eye(n) - np.diag(1.0 / np.sqrt(D_diag)) @ A_sym @ np.diag(1.0 / np.sqrt(D_diag))
    eigvals, eigvecs = np.linalg.eigh(L_norm)
    pos_x = eigvecs[:, 1]
    pos_y = eigvecs[:, 2]

    # ── Panel (a): Subgraph topology ──
    fig_a, ax = plt.subplots(figsize=(8, 5))

    edge_list_pos = []
    edge_list_neg = []
    for i in range(n):
        for j in range(n):
            if i != j and A_rvgb[i, j] != 0:
                if A_rvgb[i, j] > 0:
                    edge_list_pos.append((i, j))
                else:
                    edge_list_neg.append((i, j))

    rng_vis = np.random.RandomState(42)
    max_display = 300
    if len(edge_list_pos) > max_display:
        idx = rng_vis.choice(len(edge_list_pos), max_display, replace=False)
        edge_list_pos = [edge_list_pos[k] for k in idx]
    if len(edge_list_neg) > max_display:
        idx = rng_vis.choice(len(edge_list_neg), max_display, replace=False)
        edge_list_neg = [edge_list_neg[k] for k in idx]

    for i, j in edge_list_pos:
        ax.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]],
                'k-', alpha=0.06, lw=0.3)
    for i, j in edge_list_neg:
        ax.plot([pos_x[i], pos_x[j]], [pos_y[i], pos_y[j]],
                'r-', alpha=0.10, lw=0.4)

    for ci, comm in enumerate(communities):
        cx = [pos_x[k] for k in comm]
        cy = [pos_y[k] for k in comm]
        ax.scatter(cx, cy, c=colors_comm[ci], s=20, zorder=5,
                   edgecolors='white', linewidths=0.3,
                   label=f'$\\mathcal{{V}}_{ci+1}$ ($d_i={d[comm[0]]:+.1f}$, {len(comm)} nodes)')

    ax.legend(loc='upper right', framealpha=0.9)
    ax.axis('off')

    n_pos = sum(1 for i in range(n) for j in range(n) if i != j and A_rvgb[i, j] > 0)
    n_neg = sum(1 for i in range(n) for j in range(n) if i != j and A_rvgb[i, j] < 0)
    ax.text(0.02, 0.02, f'{n} nodes, {n_pos}+ / {n_neg}$-$ edges\n'
            f'RVGB compat.: {100*rvgb_ratio:.1f}%',
            transform=ax.transAxes, fontsize=7, va='bottom', color='gray')

    fig_a.tight_layout()
    fp_a = os.path.join(SAVE_DIR, 'fig_exp6a.pdf')
    fig_a.savefig(fp_a, dpi=300, bbox_inches='tight')
    plt.close(fig_a)
    print(f"  Saved: {fp_a}")

    # ── Panel (b): Opinion evolution ──
    fig_b, ax = plt.subplots(figsize=(8, 5))
    mark_int_b = max(T_plot // 8, 1)
    leader_node_y = None
    for ci, comm in enumerate(communities):
        color = colors_comm[ci]
        marker = markers_comm[ci]
        d_val = d[comm[0]]
        for idx, k in enumerate(comm):
            is_leader = (leader_idx is not None and k == leader_idx)
            if is_leader:
                ax.plot(t_axis[:T_plot], X[:T_plot, k], color='black',
                        linewidth=2.0, alpha=1.0, linestyle='-.',
                        marker='*', markersize=9, markevery=mark_int_b,
                        markerfacecolor='black', markeredgecolor='none',
                        zorder=10)
                leader_node_y = X[0, k]
            else:
                label = None
                if idx == 0 or (idx == 1 and leader_idx is not None and comm[0] == leader_idx):
                    label = (f'$\\mathcal{{V}}_{ci+1}$ '
                             f'($d_i={d_val:+.1f}$, {len(comm)} agents)')
                offset = (idx * 3) % mark_int_b
                ax.plot(t_axis[:T_plot], X[:T_plot, k],
                        color=color, alpha=0.5, lw=0.8, label=label,
                        marker=marker, markersize=4,
                        markevery=(offset, mark_int_b),
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
        ann_x = int(T_plot * 0.28)
        text_x = int(T_plot * 0.15)
        if text_y > ymax - 0.05 * y_range:
            ax.set_ylim(ymin, text_y + 0.10 * y_range)
        ax.annotate(
            r'Leader',
            xy=(ann_x, leader_node_y),
            xytext=(text_x, text_y),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            ha='center', va='bottom')

    fig_b.tight_layout()
    fp_b = os.path.join(SAVE_DIR, 'fig_exp6b.pdf')
    fig_b.savefig(fp_b, dpi=300, bbox_inches='tight')
    plt.close(fig_b)
    print(f"  Saved: {fp_b}")

    # ── Panel (c): Polarization ratio convergence ──
    fig_c, ax = plt.subplots(figsize=(8, 5))
    V1, V2 = communities[0], communities[1]
    mean_V1 = np.array([np.mean(X[t, V1]) for t in range(T_plot)])
    mean_V2 = np.array([np.mean(X[t, V2]) for t in range(T_plot)])

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.abs(mean_V2) / np.maximum(np.abs(mean_V1), 1e-10)
    ratio = np.clip(ratio, 0, 5)

    mark_int_c = max(T_plot // 10, 1)
    ax.plot(t_axis[:T_plot], ratio[:T_plot], color='#1f77b4', lw=2.0,
            label='$|\\bar{x}_{\\mathcal{V}_2}| / |\\bar{x}_{\\mathcal{V}_1}|$',
            marker='o', markersize=6, markevery=mark_int_c,
            markerfacecolor='#1f77b4', markeredgecolor='white', markeredgewidth=0.5)
    expected_ratio = abs(d[V2[0]] / d[V1[0]])
    ax.axhline(y=expected_ratio, color='red', linestyle='--', lw=1.5,
               label=f'Expected $|d_2/d_1|={expected_ratio:.1f}$')

    ax.set_xlabel('Time step $t$')
    ax.set_ylabel('Polarization ratio')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.4)
    ax.set_ylim([0, expected_ratio + 1.5])

    fig_c.tight_layout()
    fp_c = os.path.join(SAVE_DIR, 'fig_exp6c.pdf')
    fig_c.savefig(fp_c, dpi=300, bbox_inches='tight')
    plt.close(fig_c)
    print(f"  Saved: {fp_c}")


# =====================================================================
# Main experiment
# =====================================================================

def experiment_slashdot():
    """
    Experiment 6: Real-world validation on Slashdot signed network.
    """
    print("=" * 60)
    print("Experiment 6: Slashdot Zoo Signed Network Validation")
    print("=" * 60)

    # Step 1: Download & load
    download_dataset()
    edges = load_edge_list()

    # Step 2: Extract dense subgraph (~150 nodes)
    A_orig, n, original_ids, node_map = extract_dense_subgraph(edges, target_size=150)

    # Step 3: Community detection (sign-consistent BFS)
    communities = sign_consistent_partition(A_orig)

    # Step 4: Assign gauge and verify RVGB
    d, rvgb_ratio, A_rvgb = assign_gauge_and_verify(A_orig, communities,
                                                      d_values=(1.0, -2.0))

    # Verify gauge transform is valid
    is_rvgb = check_rvgb(A_rvgb, d)
    print(f"RVGB check after edge removal: {is_rvgb}")

    if not is_rvgb:
        print("WARNING: RVGB condition still not satisfied after edge removal!")
        return

    # Step 5: Simulate
    # Choose a leader (highest in-degree node in V2)
    V2 = communities[1]
    in_deg = np.sum(np.abs(A_rvgb[:, V2]), axis=0)
    leader_local = V2[np.argmax(in_deg)]
    print(f"Leader node: {leader_local} (original ID: {original_ids[leader_local]})")

    # Initial opinions: random in [-1, 1], scaled by gauge
    rng = np.random.RandomState(42)
    x0 = rng.uniform(-1, 1, n)
    x0 *= d  # scale by gauge so transformed opinions are in [-1,1]
    # Set leader opinion explicitly for visible polarization
    x0[leader_local] = d[leader_local] * 0.5  # x*_leader = 0.5

    # Self-confidence parameters
    theta = np.full(n, 0.3)
    theta[leader_local] = 1.0  # leader is stubborn

    T_max = 300
    print(f"\nSimulating {n} agents for {T_max} steps...")
    X, X_star = simulate_rvgb(A_rvgb, d, theta, x0, T_max, leader_idx=leader_local)

    # Report final polarization
    V1, V2 = communities[0], communities[1]
    mean_V1_final = np.mean(X[-1, V1])
    mean_V2_final = np.mean(X[-1, V2])
    print(f"\nFinal mean opinions:")
    print(f"  V1 (d=+1): mean = {mean_V1_final:.4f}")
    print(f"  V2 (d=-2): mean = {mean_V2_final:.4f}")
    if abs(mean_V1_final) > 1e-6:
        print(f"  Ratio |V2/V1| = {abs(mean_V2_final/mean_V1_final):.4f} "
              f"(expected ≈ 2.0)")
    print(f"  V1 std = {np.std(X[-1, V1]):.6f}")
    print(f"  V2 std = {np.std(X[-1, V2]):.6f}")

    # Step 6: Plot
    plot_slashdot_experiment(A_orig, A_rvgb, X, communities, d,
                             n, rvgb_ratio, leader_idx=leader_local,
                             filename='fig_exp6.pdf')


if __name__ == '__main__':
    experiment_slashdot()
