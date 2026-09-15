"""
Generate three illustrative topology diagrams for:
  (a) Structural Balance (SB)
  (b) Quasi-Structural Balance (QSB) but not SB
  (c) Real-Valued Gauge Balance (RVGB) but not QSB
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse
import os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_edge(ax, pos_src, pos_dst, sign, curve=0.2, lw=1.2):
    """Draw a directed edge with style depending on sign."""
    x0, y0 = pos_src
    x1, y1 = pos_dst
    dx, dy = x1 - x0, y1 - y0
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 0.01:
        return
    # Shrink to avoid overlap with node
    shrink = 0.18 / dist
    xs, ys = x0 + dx * shrink, y0 + dy * shrink
    xe, ye = x1 - dx * shrink, y1 - dy * shrink

    if sign > 0:
        color, ls, alpha = 'black', '-', 0.8
    else:
        color, ls, alpha = 'red', '--', 0.8

    ax.annotate('', xy=(xe, ye), xytext=(xs, ys),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, alpha=alpha, linestyle=ls,
                                connectionstyle=f'arc3,rad={curve}'))


def draw_subplot(ax, positions, edges, communities, comm_colors,
                 comm_ellipses, title, node_labels, gauge_text=None):
    """Draw one balance-type example on a given axes."""

    # Draw community ellipses
    for (cx, cy, w, h, color) in comm_ellipses:
        ell = Ellipse((cx, cy), w, h, angle=0,
                      facecolor=color, alpha=0.10, edgecolor=color,
                      linestyle='--', linewidth=1.2)
        ax.add_patch(ell)

    # Draw edges
    for (src, dst, sign, curve) in edges:
        draw_edge(ax, positions[src], positions[dst], sign, curve=curve)

    # Draw nodes
    for ci, (comm, color) in enumerate(zip(communities, comm_colors)):
        for node in comm:
            x, y = positions[node]
            ax.plot(x, y, 'o', markersize=14, color=color,
                    markeredgecolor='white', markeredgewidth=1.0, zorder=5)
            ax.text(x, y, node_labels[node], fontsize=8, ha='center', va='center',
                    color='white', fontweight='bold', zorder=6)

    # Community labels
    for (cx, cy, _, _, color) in comm_ellipses:
        # Label is added separately below
        pass

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_aspect('equal')
    ax.axis('off')

    if gauge_text:
        ax.text(0.5, -0.02, gauge_text, transform=ax.transAxes,
                fontsize=7.5, ha='center', va='top', color='gray')


def main():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    comm_blue = '#1f77b4'
    comm_red = '#d62728'
    comm_green = '#2ca02c'

    # ================================================================
    # (a) Structural Balance
    # ================================================================
    ax = axes[0]
    # V1 = {v1, v2, v3} top, V2 = {v4, v5, v6} bottom
    pos_a = {
        0: (-1.0,  1.0),  # v1
        1: ( 0.0,  1.4),  # v2
        2: ( 1.0,  1.0),  # v3
        3: (-1.0, -1.0),  # v4
        4: ( 0.0, -1.4),  # v5
        5: ( 1.0, -1.0),  # v6
    }
    labels_a = {i: f'$v_{i+1}$' for i in range(6)}
    comms_a = [[0, 1, 2], [3, 4, 5]]
    colors_a = [comm_blue, comm_red]

    # Intra-community: all positive
    # Inter-community: all negative
    edges_a = [
        # V1 intra (positive)
        (0, 1, +1, 0.15), (1, 2, +1, 0.15), (0, 2, +1, 0.25),
        # V2 intra (positive)
        (3, 4, +1, 0.15), (4, 5, +1, 0.15), (3, 5, +1, 0.25),
        # Inter-community (negative)
        (0, 3, -1, 0.15), (1, 4, -1, 0.10), (2, 5, -1, 0.15),
    ]
    ellipses_a = [
        (0.0,  1.15, 3.0, 1.4, comm_blue),
        (0.0, -1.15, 3.0, 1.4, comm_red),
    ]
    draw_subplot(ax, pos_a, edges_a, comms_a, colors_a, ellipses_a,
                 '(a) Structural Balance',
                 labels_a,
                 gauge_text=r'$d_i \in \{+1, -1\}$')
    # Community labels
    ax.text(-1.65, 1.15, r'$\mathcal{V}_1$', fontsize=11, color=comm_blue, fontweight='bold')
    ax.text(-1.65, -1.15, r'$\mathcal{V}_2$', fontsize=11, color=comm_red, fontweight='bold')
    ax.set_xlim(-2.1, 1.8)
    ax.set_ylim(-2.3, 2.3)

    # ================================================================
    # (b) Quasi-Structural Balance (QSB but not SB)
    # ================================================================
    ax = axes[1]
    pos_b = dict(pos_a)  # same layout
    labels_b = dict(labels_a)
    comms_b = [[0, 1, 2], [3, 4, 5]]
    colors_b = [comm_blue, comm_red]

    edges_b = [
        # V1 intra: mostly positive, but v1->v3 is NEGATIVE
        (0, 1, +1, 0.15), (1, 2, +1, 0.15),
        (0, 2, -1, 0.25),  # ← negative intra-community edge!
        # V2 intra (positive)
        (3, 4, +1, 0.15), (4, 5, +1, 0.15), (3, 5, +1, 0.25),
        # Inter-community (negative)
        (0, 3, -1, 0.15), (1, 4, -1, 0.10), (2, 5, -1, 0.15),
    ]
    ellipses_b = list(ellipses_a)
    draw_subplot(ax, pos_b, edges_b, comms_b, colors_b, ellipses_b,
                 '(b) Quasi-Structural Balance',
                 labels_b,
                 gauge_text=r'$d_i \in \{+1, -1\}$')
    ax.text(-1.65, 1.15, r'$\mathcal{V}_1$', fontsize=11, color=comm_blue, fontweight='bold')
    ax.text(-1.65, -1.15, r'$\mathcal{V}_2$', fontsize=11, color=comm_red, fontweight='bold')
    # Annotation for the negative intra-community edge
    ax.annotate(r'$a_{13}<0$', xy=(0.55, 1.05), fontsize=7, color='red',
                ha='center', style='italic')
    ax.set_xlim(-2.1, 1.8)
    ax.set_ylim(-2.3, 2.3)

    # ================================================================
    # (c) RVGB (but not QSB): three communities, each with 3 nodes.
    #     Every community has one intra-community negative edge but
    #     maintains positive-path connectivity.
    #     V1={v1,v2,v3} d=+1, V2={v4,v5,v6} d=+2, V3={v7,v8,v9} d=-1
    # ================================================================
    ax = axes[2]
    # Triangle layout: V1 top-left, V2 top-right, V3 bottom
    pos_c = {
        0: (-1.5,  1.3),   # v1
        1: (-0.8,  1.6),   # v2
        2: (-0.4,  1.0),   # v3
        3: ( 0.4,  1.6),   # v4
        4: ( 0.8,  1.0),   # v5
        5: ( 1.5,  1.3),   # v6
        6: (-0.6, -0.8),   # v7
        7: ( 0.0, -1.3),   # v8
        8: ( 0.6, -0.8),   # v9
    }
    labels_c = {i: f'$v_{i+1}$' for i in range(9)}
    comms_c = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    colors_c = [comm_blue, comm_red, comm_green]

    edges_c = [
        # V1 intra: v1->v2 +, v2->v3 +, v1->v3 NEGATIVE
        (0, 1, +1, 0.15),
        (1, 2, +1, 0.15),
        (0, 2, -1, 0.30),   # ← negative intra-community edge
        # V2 intra: v4->v5 +, v5->v6 +, v4->v6 NEGATIVE
        (3, 4, +1, 0.15),
        (4, 5, +1, 0.15),
        (3, 5, -1, 0.30),   # ← negative intra-community edge
        # V3 intra: v7->v8 +, v8->v9 +, v7->v9 NEGATIVE
        (6, 7, +1, 0.15),
        (7, 8, +1, 0.15),
        (6, 8, -1, 0.30),   # ← negative intra-community edge
        # V1 <-> V2: POSITIVE (d_id_j = 1*2 > 0) — violates QSB!
        (1, 3, +1, 0.12),   # v2 -> v4
        (2, 4, +1, 0.12),   # v3 -> v5
        # V1 <-> V3: NEGATIVE (d_id_j = 1*(-1) < 0)
        (0, 6, -1, 0.12),
        (2, 7, -1, 0.15),
        # V2 <-> V3: NEGATIVE (d_id_j = 2*(-1) < 0)
        (5, 8, -1, 0.12),
        (4, 7, -1, 0.15),
    ]
    ellipses_c = [
        (-0.90,  1.30, 2.0, 1.4, comm_blue),
        ( 0.90,  1.30, 2.0, 1.4, comm_red),
        ( 0.00, -1.00, 2.2, 1.4, comm_green),
    ]
    draw_subplot(ax, pos_c, edges_c, comms_c, colors_c, ellipses_c,
                 '(c) Real-Valued Gauge Balance',
                 labels_c,
                 gauge_text=r'$d_i \in \{+1, +2, -1\}$')
    ax.text(-2.10, 1.30, r'$\mathcal{V}_1$', fontsize=11, color=comm_blue, fontweight='bold')
    ax.text( 1.70, 1.30, r'$\mathcal{V}_2$', fontsize=11, color=comm_red, fontweight='bold')
    ax.text(-1.35, -1.00, r'$\mathcal{V}_3$', fontsize=11, color=comm_green, fontweight='bold')
    # Annotation for positive inter-community edge
    ax.annotate(r'$a_{ij}>0$', xy=(0.0, 1.75), fontsize=7, color='black',
                ha='center', style='italic')
    ax.set_xlim(-2.5, 2.2)
    ax.set_ylim(-2.0, 2.3)

    # ── Global legend ──
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=1.2, linestyle='-',
               label='Positive edge ($a_{ij}>0$)'),
        Line2D([0], [0], color='red', linewidth=1.2, linestyle='--',
               label='Negative edge ($a_{ij}<0$)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=9, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    filepath = os.path.join(SAVE_DIR, 'fig_balance_examples.pdf')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filepath}")


if __name__ == '__main__':
    main()
