"""
Generate communication topology figure for RVGB simulation.
- Black arrows: cooperative edges (a_ij > 0)
- Red arrows: competitive edges (a_ij < 0)
- Nodes colored by community
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Reuse network construction from simulation ──

def build_rvgb_network(n, communities, d, p_intra=0.3, p_inter=0.12,
                       w_range=(0.5, 1.5), leader_idx=None, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    A = np.zeros((n, n))

    def required_sign(i, j):
        return np.sign(d[j] / d[i])

    for comm in communities:
        for idx in range(len(comm) - 1):
            i, j = comm[idx], comm[idx + 1]
            if leader_idx is not None and i == leader_idx:
                continue
            s = required_sign(i, j)
            A[i, j] = s * rng.uniform(*w_range)
            if leader_idx is None or j != leader_idx:
                A[j, i] = required_sign(j, i) * rng.uniform(*w_range)
        for i in comm:
            if leader_idx is not None and i == leader_idx:
                continue
            for j in comm:
                if i == j:
                    continue
                if A[i, j] == 0 and rng.rand() < p_intra:
                    s = required_sign(i, j)
                    A[i, j] = s * rng.uniform(*w_range)

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
    return A


def compute_layout(n, communities, leader_idx=None):
    """
    Compute node positions: two arcs (semicircles) for two communities,
    leader node at center-right.
    """
    positions = {}

    n_comm = len(communities)

    if n_comm == 2:
        # Two-community layout: two semicircles facing each other
        V1, V2 = communities[0], communities[1]

        # V1: upper semicircle
        n1 = len(V1)
        for idx, node in enumerate(V1):
            angle = np.pi * 0.15 + np.pi * 0.7 * idx / max(n1 - 1, 1)
            r = 3.8
            positions[node] = (r * np.cos(angle), r * np.sin(angle))

        # V2: lower semicircle
        n2 = len(V2)
        for idx, node in enumerate(V2):
            if leader_idx is not None and node == leader_idx:
                continue
            angle = -np.pi * 0.15 - np.pi * 0.7 * idx / max(n2 - 2, 1)
            r = 3.8
            positions[node] = (r * np.cos(angle), r * np.sin(angle))

        # Leader at a prominent position
        if leader_idx is not None:
            positions[leader_idx] = (5.0, 0.0)

    elif n_comm == 3:
        # Three-community layout: three arcs at 120 degrees
        angles_center = [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3]
        for ci, comm in enumerate(communities):
            nc = len(comm)
            center_angle = angles_center[ci]
            arc_span = 0.8  # radians
            for idx, node in enumerate(comm):
                if leader_idx is not None and node == leader_idx:
                    continue
                offset = -arc_span / 2 + arc_span * idx / max(nc - 2, 1)
                angle = center_angle + offset
                r = 3.5 + 0.3 * np.sin(idx * 0.5)
                positions[node] = (r * np.cos(angle), r * np.sin(angle))

        if leader_idx is not None:
            positions[leader_idx] = (0.0, 0.0)

    return positions


def draw_topology(A, communities, d, positions, leader_idx, title, filename,
                  comm_labels=None, show_all_edges=True, max_edges_display=None):
    """
    Draw directed communication topology.
    - Black arrows: cooperative (a_ij > 0)
    - Red arrows: competitive (a_ij < 0)
    """
    n = A.shape[0]
    comm_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

    fig, ax = plt.subplots(figsize=(10, 8))

    # Node-to-community mapping
    node_comm = {}
    for ci, comm in enumerate(communities):
        for node in comm:
            node_comm[node] = ci

    # ── Draw edges ──
    edges_coop = []
    edges_comp = []
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] != 0:
                # Edge from j -> i means j influences i, i.e., a_ij != 0
                # Arrow direction: j --> i
                if A[i, j] > 0:
                    edges_coop.append((j, i, A[i, j]))
                else:
                    edges_comp.append((j, i, A[i, j]))

    # If too many edges, subsample for clarity
    rng_draw = np.random.RandomState(42)
    if max_edges_display is not None:
        if len(edges_coop) > max_edges_display:
            idx_sel = rng_draw.choice(len(edges_coop), max_edges_display, replace=False)
            edges_coop = [edges_coop[k] for k in idx_sel]
        if len(edges_comp) > max_edges_display:
            idx_sel = rng_draw.choice(len(edges_comp), max_edges_display, replace=False)
            edges_comp = [edges_comp[k] for k in idx_sel]

    # Draw cooperative edges (black)
    for (src, dst, w) in edges_coop:
        if src not in positions or dst not in positions:
            continue
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        # Shorten arrow to not overlap with node
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.01:
            continue
        shrink = 0.25 / dist
        xs, ys = x0 + dx * shrink, y0 + dy * shrink
        xe, ye = x1 - dx * shrink, y1 - dy * shrink
        ax.annotate('', xy=(xe, ye), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle='->', color='black',
                                    lw=0.4, alpha=0.25,
                                    connectionstyle='arc3,rad=0.1'))

    # Draw competitive edges (red)
    for (src, dst, w) in edges_comp:
        if src not in positions or dst not in positions:
            continue
        x0, y0 = positions[src]
        x1, y1 = positions[dst]
        dx, dy = x1 - x0, y1 - y0
        dist = np.sqrt(dx**2 + dy**2)
        if dist < 0.01:
            continue
        shrink = 0.25 / dist
        xs, ys = x0 + dx * shrink, y0 + dy * shrink
        xe, ye = x1 - dx * shrink, y1 - dy * shrink
        ax.annotate('', xy=(xe, ye), xytext=(xs, ys),
                    arrowprops=dict(arrowstyle='->', color='red',
                                    lw=0.5, alpha=0.35,
                                    connectionstyle='arc3,rad=0.1'))

    # ── Draw nodes ──
    for ci, comm in enumerate(communities):
        color = comm_colors[ci % len(comm_colors)]
        for node in comm:
            if node not in positions:
                continue
            x, y = positions[node]
            is_leader = (leader_idx is not None and node == leader_idx)
            if is_leader:
                # Leader: larger, star-shaped marker
                ax.plot(x, y, marker='*', markersize=18, color='gold',
                        markeredgecolor='black', markeredgewidth=1.2, zorder=10)
                ax.annotate(f'$v_{{50}}$', (x, y), textcoords='offset points',
                            xytext=(10, 8), fontsize=10, fontweight='bold', zorder=11)
            else:
                ax.plot(x, y, 'o', markersize=6.5, color=color,
                        markeredgecolor='white', markeredgewidth=0.3, zorder=5)

    # ── Community boundary ellipses ──
    from matplotlib.patches import Ellipse
    if len(communities) == 2:
        # V1 ellipse (upper)
        ell1 = Ellipse((0, 2.5), 9.0, 4.5, angle=0,
                        facecolor=comm_colors[0], alpha=0.06, edgecolor=comm_colors[0],
                        linestyle='--', linewidth=1.5)
        ax.add_patch(ell1)
        # V2 ellipse (lower)
        ell2 = Ellipse((0, -2.5), 9.0, 4.5, angle=0,
                        facecolor=comm_colors[1], alpha=0.06, edgecolor=comm_colors[1],
                        linestyle='--', linewidth=1.5)
        ax.add_patch(ell2)

    # ── Legend ──
    legend_elements = []
    if comm_labels is None:
        comm_labels = [f'$\\mathcal{{V}}_{ci+1}$' for ci in range(len(communities))]
    for ci in range(len(communities)):
        d_val = d[communities[ci][0]]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=comm_colors[ci], markersize=9,
                       label=f'{comm_labels[ci]} ($d_i={d_val:+.1f}$, {len(communities[ci])} nodes)')
        )
    if leader_idx is not None:
        legend_elements.append(
            plt.Line2D([0], [0], marker='*', color='w',
                       markerfacecolor='gold', markeredgecolor='black',
                       markersize=14, label=f'Leader $v_{{50}}$')
        )

    # Arrow legend
    legend_elements.append(
        plt.Line2D([0], [0], color='black', linewidth=1.2,
                   label='Cooperative ($a_{ij}>0$)')
    )
    legend_elements.append(
        plt.Line2D([0], [0], color='red', linewidth=1.2,
                   label='Competitive ($a_{ij}<0$)')
    )

    ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
              framealpha=0.9, borderpad=0.8)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_aspect('equal')
    ax.set_xlim(-6.0, 6.5)
    ax.set_ylim(-5.5, 5.5)
    ax.axis('off')

    # Edge count annotation
    n_coop_total = sum(1 for i in range(n) for j in range(n) if i != j and A[i,j] > 0)
    n_comp_total = sum(1 for i in range(n) for j in range(n) if i != j and A[i,j] < 0)
    ax.text(0.98, 0.02,
            f'Edges: {n_coop_total} cooperative + {n_comp_total} competitive = {n_coop_total+n_comp_total} total',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            color='gray')

    filepath = os.path.join(SAVE_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


# =====================================================================
# Generate topology for Experiment 1
# =====================================================================

def topology_exp1():
    """Experiment 1: Two-community asymmetric RVGB (d = +1 vs -2)."""
    n = 50
    leader_idx = n - 1

    V1 = list(range(25))
    V2 = list(range(25, 50))
    communities = [V1, V2]

    d = np.ones(n)
    d[25:] = -2.0

    rng = np.random.RandomState(100)
    A = build_rvgb_network(n, communities, d, p_intra=0.25, p_inter=0.10,
                           w_range=(0.5, 1.5), leader_idx=leader_idx, rng=rng)

    positions = compute_layout(n, communities, leader_idx)
    draw_topology(A, communities, d, positions, leader_idx,
                  'Communication Topology: Experiment 1 ($n=50$, $d \\in \\{+1, -2\\}$)',
                  'fig_topology_exp1.pdf',
                  comm_labels=['$\\mathcal{V}_1$', '$\\mathcal{V}_2$'])


# =====================================================================
# Generate topology for Experiment 2
# =====================================================================

def topology_exp2():
    """Experiment 2: Three-community RVGB (d in {+1, +2, -1})."""
    n = 50
    leader_idx = n - 1

    V1 = list(range(17))
    V2 = list(range(17, 34))
    V3 = list(range(34, 50))
    communities = [V1, V2, V3]

    d = np.zeros(n)
    d[:17] = 1.0
    d[17:34] = 2.0
    d[34:] = -1.0

    rng = np.random.RandomState(200)
    A = build_rvgb_network(n, communities, d, p_intra=0.25, p_inter=0.08,
                           w_range=(0.5, 1.5), leader_idx=leader_idx, rng=rng)

    # Three-community circular layout
    positions = {}
    # V1: top-left arc
    for idx, node in enumerate(V1):
        angle = np.pi * 0.55 + np.pi * 0.55 * idx / max(len(V1) - 1, 1)
        r = 3.8
        positions[node] = (r * np.cos(angle), r * np.sin(angle))

    # V2: top-right arc
    for idx, node in enumerate(V2):
        angle = np.pi * 0.0 + np.pi * 0.45 * idx / max(len(V2) - 1, 1)
        r = 3.8
        positions[node] = (r * np.cos(angle), r * np.sin(angle))

    # V3: bottom arc
    v3_no_leader = [v for v in V3 if v != leader_idx]
    for idx, node in enumerate(v3_no_leader):
        angle = -np.pi * 0.15 - np.pi * 0.7 * idx / max(len(v3_no_leader) - 1, 1)
        r = 3.8
        positions[node] = (r * np.cos(angle), r * np.sin(angle))

    positions[leader_idx] = (0.0, -0.3)

    draw_topology(A, communities, d, positions, leader_idx,
                  'Communication Topology: Experiment 2 ($n=50$, $k=3$)',
                  'fig_topology_exp2.pdf',
                  comm_labels=['$\\mathcal{V}_1$', '$\\mathcal{V}_2$', '$\\mathcal{V}_3$'])


if __name__ == '__main__':
    print("Generating communication topology figures...")
    topology_exp1()
    topology_exp2()
    print("Done!")
