"""
Experiment 6 (revised, R1): Slashdot Zoo validation with before/after analysis.
==============================================================================
Pipeline (identical extraction and partition as the original Experiment 6):
  1. extract the 149-node strongly connected subgraph around an adversarial hub;
  2. spectral bipartition, gauge d in {+1, -2};
  3. remove the RVGB-violating edges;
  4. NEW: restrict to the largest strongly connected component of the pruned
     graph so that every follower keeps at least one in-neighbour and the
     positive spanning tree condition (C1) can be checked;
  5. report connectivity / spectral / convergence parameters BEFORE and AFTER,
     run the dynamics on both, and regenerate Fig. 7.
Outputs: fig7a_R1.pdf, fig7b_R1.pdf, fig7c_R1.pdf, exp6_R1.json
"""
import os, sys, json
import numpy as np
import scipy.sparse as sp
from collections import deque
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import RVGB_slashdot_experiment as SE
from revision_analysis import (scc_info, wcc_info, signed_triangle_balance, compat_fraction,
                               local_search_partition, spectral_summary)

def params_exact(A, d, theta, leader, h=1):
    """Convergence parameters g, sigma of Theorem 3, eqs. (16)-(17), with Z_min = min_i Z_i over the followers."""
    n = A.shape[0]; ratio = d[None, :] / d[:, None]; mask = A != 0; np.fill_diagonal(mask, False)
    At = np.where(mask, A * ratio, 0.0); Z = At.sum(axis=1)
    F = [i for i in range(n) if i != leader]
    alpha = np.abs(A[mask]).min()
    same_neg = (A < 0) & mask & (d[:, None] * d[None, :] > 0)
    beta = np.abs(A[same_neg]).max() if same_neg.any() else 0.0
    Ntil = same_neg.sum(axis=1); Ntil_max = int(Ntil[F].max())
    gam = np.abs(ratio[mask]); gmin, gmax = gam.min(), gam.max()
    th = np.asarray(theta, float); thmin, thmax = th[F].min(), th[F].max()
    Zmin, Zmax = Z[F].min(), Z[F].max()
    nonpos = int((Z[F] <= 0).sum())
    g = 1 + 2 * (1 - thmin) * beta * gmax * Ntil_max / Zmin if Zmin > 0 else np.inf
    sigma = min(thmin, (1 - thmax) * alpha * gmin / Zmin) if Zmin > 0 else 0.0
    Qt = np.where(mask, At / np.where(Z[:, None] != 0, Z[:, None], 1), 0.0)
    Gam = np.diag(th) + (np.eye(n) - np.diag(th)) @ Qt; Gam[leader] = 0; Gam[leader, leader] = 1
    Phi = Gam[np.ix_(F, F)]
    g_emp = float(np.abs(Phi).sum(axis=1).max())
    # positive spanning tree from the leader in G(Q~)
    Pos = Qt > 0; depth = -np.ones(n, int); depth[leader] = 0; q = deque([leader])
    while q:
        u = q.popleft()
        for v in np.where(Pos[:, u])[0]:
            if depth[v] < 0: depth[v] = depth[u] + 1; q.append(v)
    reach = int((depth >= 0).sum()); delta = int(depth.max()) if reach == n else -1
    rho = g ** (delta * h) - sigma ** (delta * h) if delta > 0 and np.isfinite(g) else None
    rad = float(max(abs(np.linalg.eigvals(np.abs(Phi)))))
    return dict(alpha=float(alpha), beta=float(beta), gamma_min=float(gmin), gamma_max=float(gmax), Ntilde_max=Ntil_max,
                Z_min=float(Zmin), Z_max=float(Zmax), followers_with_Z_nonpositive=nonpos,
                g=float(g) if np.isfinite(g) else None, sigma=float(sigma), g_emp=g_emp,
                tree_reach=reach, delta=delta, rho_C3=(float(rho) if rho is not None else None),
                spectral_radius_absPhi=rad)

def structure(A):
    M = sp.csr_matrix(A); nscc, big, _ = scc_info(M); nwcc, _ = wcc_info(M)
    tri = signed_triangle_balance(M); spec = spectral_summary(A)
    indeg = (A != 0).sum(axis=1)
    return dict(n=int(A.shape[0]), m=int((A != 0).sum()), pos=int((A > 0).sum()), neg=int((A < 0).sum()),
                strongly_connected=bool(nscc == 1), n_scc=int(nscc), largest_scc=int(big), n_wcc=int(nwcc),
                min_in_degree=int(indeg.min()), mean_in_degree=float(indeg.mean()), max_in_degree=int(indeg.max()),
                reciprocity=float(((A != 0) & (A.T != 0)).sum() / (A != 0).sum()),
                balanced_triangles=float(tri['balanced_fraction']), **spec)

def main():
    SE.download_dataset(); edges = SE.load_edge_list()
    A0, n0, ids0, _ = SE.extract_dense_subgraph(edges, target_size=150)
    comm0 = SE.sign_consistent_partition(A0)
    d0, ratio0, A1 = SE.assign_gauge_and_verify(A0, comm0, d_values=(1.0, -2.0))
    V1_0, V2_0 = comm0
    leader0 = V2_0[np.argmax(np.sum(np.abs(A1[:, V2_0]), axis=0))]
    theta0 = np.full(n0, 0.3); theta0[leader0] = 1.0
    out = {'extracted_n': n0, 'extracted_m': int((A0 != 0).sum()), 'compat': ratio0, 'V1': len(V1_0), 'V2': len(V2_0)}
    # refined partition (greedy flips) for reference only
    d_ref = local_search_partition(sp.csr_matrix(A0), np.sign(d0)); fr_ref, _, _ = compat_fraction(sp.csr_matrix(A0), d_ref)
    out['compat_after_local_search'] = float(fr_ref)
    # BEFORE: unpruned 149-node subgraph
    out['before'] = dict(**structure(A0), **params_exact(A0, d0, theta0, leader0))
    # AFTER (pruned, all 149 nodes)
    out['after_pruned_149'] = dict(**structure(A1), **params_exact(A1, d0, theta0, leader0))
    # AFTER + largest SCC
    _, _, lab = scc_info(sp.csr_matrix(A1)); comp = np.argmax(np.bincount(lab)); keep = np.where(lab == comp)[0]
    A2 = A1[np.ix_(keep, keep)]; d2 = d0[keep]; n2 = len(keep)
    V1 = [k for k in range(n2) if d2[k] > 0]; V2 = [k for k in range(n2) if d2[k] < 0]
    leader = V2[np.argmax(np.sum(np.abs(A2[:, V2]), axis=0))]
    theta = np.full(n2, 0.3); theta[leader] = 1.0
    out['after_scc'] = dict(**structure(A2), **params_exact(A2, d2, theta, leader), V1=len(V1), V2=len(V2),
                            removed_nodes=int(n0 - n2), removed_edges_total=int((A0 != 0).sum() - (A2 != 0).sum()))
    # dynamics: before (fixed gauge, violating edges kept) vs after
    rng = np.random.RandomState(42); x0_full = rng.uniform(-1, 1, n0) * d0; x0_full[leader0] = d0[leader0] * 0.5
    Xb, Xsb = SE.simulate_rvgb(A0, d0, theta0, x0_full, 300, leader_idx=leader0)
    out['dynamics_before'] = dict(final_abs_max=float(np.abs(Xb[-1]).max()), ratio=float(abs(Xb[-1, V2_0].mean() / Xb[-1, V1_0].mean())),
                                  std_V1=float(Xb[-1, V1_0].std()), std_V2=float(Xb[-1, V2_0].std()))
    x0 = x0_full[keep].copy(); x0[leader] = d2[leader] * 0.5
    T = 300
    X, Xs = SE.simulate_rvgb(A2, d2, theta, x0, T, leader_idx=leader)
    out['dynamics_after'] = dict(mean_V1=float(X[-1, V1].mean()), mean_V2=float(X[-1, V2].mean()),
                                 ratio=float(abs(X[-1, V2].mean() / X[-1, V1].mean())),
                                 std_V1=float(X[-1, V1].std()), std_V2=float(X[-1, V2].std()),
                                 max_dev_from_pattern=float(np.abs(Xs[-1] - Xs[-1, leader]).max()),
                                 ratio_t100=float(abs(X[100, V2].mean() / X[100, V1].mean())))
    # figures (same plotting code; files renamed)
    SE.plot_slashdot_experiment(A2, A2, X, [V1, V2], d2, n2, ratio0, leader_idx=leader)
    for a, b in [('fig_exp6a.pdf', 'fig7a_R1.pdf'), ('fig_exp6b.pdf', 'fig7b_R1.pdf'), ('fig_exp6c.pdf', 'fig7c_R1.pdf')]:
        os.replace(os.path.join(HERE, a), os.path.join(HERE, b))
    json.dump(out, open(os.path.join(HERE, 'exp6_R1.json'), 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))

if __name__ == '__main__':
    main()
