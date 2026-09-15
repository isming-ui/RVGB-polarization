"""
Revision analysis for TCSS-2026-07-1138 (R1).
==============================================
Answers the following reviewer requests numerically:

  R1-3 / R2-3 : Slashdot subgraph before/after removal of RVGB-violating edges
                (connectivity, spectral properties, convergence parameters g, sigma),
                representativeness across alternative seeds, and RVGB compatibility
                of the ENTIRE Slashdot Zoo network.
  R3-2 / R3-3 : empirical runtime scaling of (a) RVGB/cycle checks, (b) one
                asynchronous update, dense vs. sparse, for n up to 1e5, and a
                sparse-matrix simulation on the full Slashdot network.

Outputs: results_revision.json + printed report.  Run from the code/ folder:
    python3 revision_analysis.py
"""
import os, sys, time, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import RVGB_slashdot_experiment as SE   # reuse the exact extraction of Experiment 6

RESULTS = {}
def log(*a):
    print(*a, flush=True)

# ---------------------------------------------------------------------------
# Generic helpers (sparse, O(n+m))
# ---------------------------------------------------------------------------
def edges_to_csr(edges, n):
    src = np.array([e[0] for e in edges]); dst = np.array([e[1] for e in edges])
    w = np.array([e[2] for e in edges], dtype=float)
    # A[src,dst] = sign: the rater src is influenced by the rated user dst (same convention as Experiment 6 code)
    A = sp.csr_matrix((w, (src, dst)), shape=(n, n))
    A.sum_duplicates()
    return A

def scc_info(A):
    ncomp, labels = sp.csgraph.connected_components(A, directed=True, connection='strong')
    sizes = np.bincount(labels)
    return ncomp, int(sizes.max()), labels

def wcc_info(A):
    ncomp, labels = sp.csgraph.connected_components(A, directed=True, connection='weak')
    sizes = np.bincount(labels)
    return ncomp, int(sizes.max())

def bfs_sign_gauge(A):
    """Sign gauge d in {+1,-1}^n from a BFS spanning forest of the underlying
    undirected signed graph (construction used in the proof of Theorem 2).
    Returns d and the number of edges whose sign is inconsistent with d
    (== number of non-tree edges closing a negative cycle). O(n+m)."""
    n = A.shape[0]
    S = (A + A.T).tocsr()          # symmetrized signed pattern
    S.data = np.sign(S.data)
    d = np.zeros(n, dtype=int)
    for root in range(n):
        if d[root] != 0: continue
        d[root] = 1
        q = deque([root])
        while q:
            u = q.popleft()
            lo, hi = S.indptr[u], S.indptr[u+1]
            for v, s in zip(S.indices[lo:hi], S.data[lo:hi]):
                if d[v] == 0 and s != 0:
                    d[v] = d[u] * int(np.sign(s)); q.append(v)
    return d

def compat_fraction(A, d):
    """Fraction of edges (j->i) with sgn(a_ij) = sgn(d_j/d_i)  (RVGB compatibility)."""
    Ac = A.tocoo()
    ok = np.sign(Ac.data) == np.sign(d[Ac.col] / d[Ac.row])
    return ok.mean(), int(ok.sum()), int((~ok).sum())

def local_search_partition(A, d, max_sweeps=30):
    """Greedy node flips (d_i -> -d_i) to increase compatibility. Returns refined d."""
    n = A.shape[0]
    S = (A + A.T).tocsr(); S.data = np.sign(S.data)
    d = d.astype(float).copy()
    for sweep in range(max_sweeps):
        flips = 0
        for i in range(n):
            lo, hi = S.indptr[i], S.indptr[i+1]
            if lo == hi: continue
            nb, s = S.indices[lo:hi], S.data[lo:hi]
            # frustration of node i: edges with s * d_i * d_j < 0
            fr = np.sum(s * d[i] * d[nb] < 0); ok = np.sum(s * d[i] * d[nb] > 0)
            if fr > ok:
                d[i] = -d[i]; flips += 1
        if flips == 0: break
    return d

def signed_triangle_balance(A, max_nodes=None):
    """Fraction of balanced (positive-product) triangles in the underlying
    undirected signed graph. Counted via sparse trace identities."""
    S = (A + A.T).tocsr(); S.data = np.sign(S.data)
    S.eliminate_zeros()
    P = S.copy(); P.data = (P.data > 0).astype(float); P.eliminate_zeros()
    N = S.copy(); N.data = (N.data < 0).astype(float); N.eliminate_zeros()
    def tr3(X, Y, Z):   # trace(XYZ) = sum_{ik} (XY)_{ik} Z_{ki}
        XY = (X @ Y).tocsr()
        return float((XY.multiply(Z.T.tocsr())).sum())
    T0 = tr3(P, P, P) / 6.0
    T1 = tr3(P, P, N) / 2.0
    T2 = tr3(P, N, N) / 2.0
    T3 = tr3(N, N, N) / 6.0
    tot = T0 + T1 + T2 + T3
    return dict(T0=T0, T1=T1, T2=T2, T3=T3, balanced_fraction=(T0 + T2) / tot if tot else float('nan'))

# ---------------------------------------------------------------------------
# Convergence parameters of Theorem 3 (given A, d, theta, leader)
# ---------------------------------------------------------------------------
def convergence_parameters(A_dense, d, theta, leader, h=1):
    n = A_dense.shape[0]
    A = A_dense
    ratio = d[None, :] / d[:, None]                 # d_j / d_i
    mask = (A != 0); np.fill_diagonal(mask, False)
    At = np.where(mask, A * ratio, 0.0)             # gauge-transformed weights
    Z = At.sum(axis=1)                              # Z_i
    Npos = ((A > 0) & mask).sum(axis=1)
    Nneg_hat = ((A < 0) & mask & (d[:, None] * d[None, :] < 0)).sum(axis=1)
    Nneg_tilde = ((A < 0) & mask & (d[:, None] * d[None, :] > 0)).sum(axis=1)
    followers = [i for i in range(n) if i != leader]
    alpha = np.abs(A[mask]).min()
    same_neg = (A < 0) & mask & (d[:, None] * d[None, :] > 0)
    beta = np.abs(A[same_neg]).max() if same_neg.any() else 0.0
    gam = np.abs(ratio[mask]); gmin, gmax = gam.min(), gam.max()
    Npos_min = Npos[followers].min(); Nhat_min = Nneg_hat[followers].min(); Ntil_max = Nneg_tilde[followers].max()
    Zmin_bound = alpha * gmin * (Npos_min + Nhat_min) - beta * gmax * Ntil_max
    th = np.asarray(theta, float); thmin, thmax = th[followers].min(), th[followers].max()
    g = 1 + 2 * (1 - thmin) * beta * gmax * Ntil_max / Zmin_bound if Zmin_bound > 0 else np.inf
    sigma = min(thmin, (1 - thmax) * alpha * gmin / Zmin_bound) if Zmin_bound > 0 else 0.0
    # empirical counterparts computed directly from Phi(t) at a synchronous step
    Qt = np.where(mask, At / np.where(Z[:, None] != 0, Z[:, None], 1), 0.0)
    Gam = np.diag(th) + (np.eye(n) - np.diag(th)) @ Qt
    Gam[leader] = 0; Gam[leader, leader] = 1
    F = [i for i in range(n) if i != leader]
    Phi = Gam[np.ix_(F, F)]
    g_emp = np.abs(Phi).sum(axis=1).max()
    y = Gam[F, leader]
    sigma_emp = y[y > 0].min() if (y > 0).any() else 0.0
    # positive spanning tree rooted at leader in G(Q~): depth via BFS on positive edges
    Pos = (Qt > 0)
    depth = -np.ones(n, int); depth[leader] = 0; q = deque([leader])
    while q:
        u = q.popleft()
        for v in np.where(Pos[:, u])[0]:      # edge u -> v exists iff Qt[v,u] > 0
            if depth[v] < 0: depth[v] = depth[u] + 1; q.append(v)
    delta = int(depth.max()) if (depth >= 0).all() else -1
    rho = g ** (delta * h) - sigma ** (delta * h) if delta > 0 and np.isfinite(g) else np.nan
    # spectral radius of |Phi| (empirical contraction)
    rad_absPhi = max(abs(np.linalg.eigvals(np.abs(Phi))))
    return dict(n=n, m=int(mask.sum()), alpha=float(alpha), beta=float(beta), gamma_min=float(gmin), gamma_max=float(gmax),
                Npos_min=int(Npos_min), Nhat_min=int(Nhat_min), Ntilde_max=int(Ntil_max),
                Z_min_bound=float(Zmin_bound), Z_min_actual=float(Z[followers].min()), Z_max_actual=float(Z[followers].max()),
                g=float(g), sigma=float(sigma), g_emp=float(g_emp), sigma_emp=float(sigma_emp),
                delta=delta, rho_C4=float(rho) if rho == rho else None, spectral_radius_absPhi=float(rad_absPhi),
                fraction_Z_positive=float((Z[followers] > 0).mean()))

def spectral_summary(A_dense):
    n = A_dense.shape[0]
    S = (A_dense + A_dense.T) / 2
    Dabs = np.diag(np.abs(S).sum(axis=1))
    Ls = Dabs - S                        # signed Laplacian
    Lu = Dabs - np.abs(S)                # unsigned Laplacian
    ev_s = np.linalg.eigvalsh(Ls); ev_u = np.linalg.eigvalsh(Lu)
    return dict(signed_laplacian_lambda_min=float(ev_s[0]),
                unsigned_laplacian_lambda2=float(ev_u[1]),
                spectral_radius_absA=float(max(abs(np.linalg.eigvals(np.abs(A_dense))))))

def simulate_sparse(A, d, theta, x0, T, leader, seed=0, h=1):
    """Sparse asynchronous simulation of (7) in transformed coordinates."""
    n = A.shape[0]
    Ac = A.tocoo()
    At = sp.csr_matrix((Ac.data * d[Ac.col] / d[Ac.row], (Ac.row, Ac.col)), shape=(n, n))
    Z = np.asarray(At.sum(axis=1)).ravel()
    inv = np.where(Z != 0, 1.0 / np.where(Z != 0, Z, 1), 0.0)
    Qt = sp.diags(inv) @ At
    xs = x0 / d
    rng = np.random.RandomState(seed)
    mv1, mv2 = np.zeros(n), np.zeros(n)
    for t in range(T):
        upd = rng.rand(n) < 1.0 if h == 1 else rng.rand(n) < 1.0 / h
        upd[leader] = False
        xn = theta * xs + (1 - theta) * (Qt @ xs)
        xs = np.where(upd, xn, xs)
        if t == T - 2: mv1 = xs.copy()
    return xs * d, xs

# ---------------------------------------------------------------------------
# PART A: full Slashdot network
# ---------------------------------------------------------------------------
def part_A_full_network(edges):
    log("\n=== PART A: entire Slashdot Zoo network ===")
    nodes = sorted(set([e[0] for e in edges]) | set([e[1] for e in edges]))
    idx = {v: k for k, v in enumerate(nodes)}; n = len(nodes)
    E = [(idx[s], idx[t], sg) for s, t, sg in edges if s != t]
    A = edges_to_csr(E, n)
    m = A.nnz
    npos = int((A.data > 0).sum()); nneg = int((A.data < 0).sum())
    nscc, big_scc, lab = scc_info(A); nwcc, big_wcc = wcc_info(A)
    log(f"nodes={n}, edges={m}, pos={npos} ({100*npos/m:.1f}%), neg={nneg} ({100*nneg/m:.1f}%)")
    log(f"strongly connected components: {nscc}, largest SCC = {big_scc} nodes ({100*big_scc/n:.1f}%)")
    log(f"weakly connected components: {nwcc}, largest WCC = {big_wcc} nodes ({100*big_wcc/n:.1f}%)")
    t0 = time.time(); d0 = bfs_sign_gauge(A); t_bfs = time.time() - t0
    frac0, ok0, bad0 = compat_fraction(A, d0)
    log(f"BFS sign gauge (Theorem-2 construction): {t_bfs:.2f}s; compatible edges {ok0}/{m} = {100*frac0:.1f}% -> cycle condition violated by {bad0} edges")
    t0 = time.time(); d1 = local_search_partition(A, d0); t_ls = time.time() - t0
    frac1, ok1, bad1 = compat_fraction(A, d1)
    log(f"after greedy local search ({t_ls:.1f}s): compatible {ok1}/{m} = {100*frac1:.1f}% (frustrated edges {bad1}, {100*bad1/m:.2f}%)")
    t0 = time.time(); tri = signed_triangle_balance(A); t_tri = time.time() - t0
    log(f"signed triangles ({t_tri:.1f}s): T0={tri['T0']:.0f} T1={tri['T1']:.0f} T2={tri['T2']:.0f} T3={tri['T3']:.0f}; balanced fraction = {100*tri['balanced_fraction']:.1f}%")
    # RVGB-projected full network: drop violating edges w.r.t. d1, largest SCC, sparse simulation
    Ac = A.tocoo()
    keep = np.sign(Ac.data) == np.sign(d1[Ac.col] / d1[Ac.row])
    Ar = sp.csr_matrix((Ac.data[keep], (Ac.row[keep], Ac.col[keep])), shape=(n, n))
    nscc_r, big_r, lab_r = scc_info(Ar)
    comp = np.argmax(np.bincount(lab_r)); sel = np.where(lab_r == comp)[0]
    As = Ar[sel][:, sel].tocsr(); ds = d1[sel].copy()
    # gauge values +1 / -2 as in Experiment 6 (community with more nodes gets +1)
    plus = ds > 0
    if plus.sum() < (~plus).sum(): plus = ~plus
    dv = np.where(plus, 1.0, -2.0)
    ns = As.shape[0]
    log(f"RVGB projection: removed {int((~keep).sum())} edges; largest SCC of projected graph = {ns} nodes, {As.nnz} edges; |V1|={int(plus.sum())}, |V2|={int((~plus).sum())}")
    # leader: highest in-degree node in V2
    indeg = np.asarray((As != 0).sum(axis=1)).ravel()
    cand = np.where(~plus)[0]; leader = cand[np.argmax(indeg[cand])]
    theta = np.full(ns, 0.3); theta[leader] = 1.0
    rng = np.random.RandomState(42); x0 = rng.uniform(-1, 1, ns) * dv; x0[leader] = dv[leader] * 0.5
    T = 2000
    # depth of the positive spanning tree rooted at the leader in G(Q~): BFS over edges u->v with Qt[v,u]>0, i.e. As[v,u]!=0
    order, pred = sp.csgraph.breadth_first_order(As.T.tocsr(), leader, directed=True, return_predecessors=True)
    dep = np.full(ns, -1); dep[leader] = 0
    for v in order[1:]: dep[v] = dep[pred[v]] + 1
    depth_full = int(dep.max()); log(f'positive spanning tree rooted at the leader: reaches {int((dep>=0).sum())}/{ns} nodes, depth = {depth_full}')
    t0 = time.time(); X, Xs = simulate_sparse(As, dv, theta, x0, T, leader); t_sim = time.time() - t0
    m1 = X[plus].mean(); m2 = X[~plus].mean()
    disp = np.abs(Xs - Xs[leader]).max()
    log(f"sparse simulation of full projected network: {T} steps in {t_sim:.2f}s ({1000*t_sim/T:.1f} ms/step); mean V1={m1:.4f}, mean V2={m2:.4f}, ratio |V2/V1|={abs(m2/m1):.3f}, max |x*_i - x*_leader| = {disp:.2e}")
    RESULTS['full_network'] = dict(n=n, m=m, pos=npos, neg=nneg, n_scc=nscc, largest_scc=big_scc, n_wcc=nwcc, largest_wcc=big_wcc,
        bfs_gauge_compat=frac0, bfs_gauge_violating=bad0, bfs_time_s=t_bfs,
        local_search_compat=frac1, local_search_violating=bad1, local_search_time_s=t_ls,
        triangles=tri, triangle_time_s=t_tri,
        projected_removed_edges=int((~keep).sum()), projected_largest_scc_n=ns, projected_largest_scc_m=int(As.nnz),
        projected_V1=int(plus.sum()), projected_V2=int((~plus).sum()),
        tree_depth=depth_full, sim_T=T, sim_time_s=t_sim, sim_ms_per_step=1000*t_sim/T, sim_mean_V1=float(m1), sim_mean_V2=float(m2), sim_ratio=float(abs(m2/m1)), sim_max_dispersion=float(disp))
    return A, n, idx, nodes, d1

# ---------------------------------------------------------------------------
# PART B: the 149-node subgraph of Experiment 6, before / after edge removal
# ---------------------------------------------------------------------------
def part_B_subgraph(edges, A_full, idx_map, d_full):
    log("\n=== PART B: Experiment-6 subgraph, before vs after removal of violating edges ===")
    A_orig, n, original_ids, node_map = SE.extract_dense_subgraph(edges, target_size=150)
    communities = SE.sign_consistent_partition(A_orig)
    d, ratio, A_rvgb = SE.assign_gauge_and_verify(A_orig, communities, d_values=(1.0, -2.0))
    V1, V2 = communities
    in_deg = np.sum(np.abs(A_rvgb[:, V2]), axis=0); leader = V2[np.argmax(in_deg)]
    theta = np.full(n, 0.3); theta[leader] = 1.0
    out = {}
    for name, M in [('before', A_orig), ('after', A_rvgb)]:
        Msp = sp.csr_matrix(M)
        nscc, big, _ = scc_info(Msp); nwcc, bigw = wcc_info(Msp)
        spec = spectral_summary(M)
        par = convergence_parameters(M, d, theta, leader)
        tri = signed_triangle_balance(Msp)
        indeg = (M != 0).sum(axis=1); outdeg = (M != 0).sum(axis=0)
        out[name] = dict(pos=int((M > 0).sum()), neg=int((M < 0).sum()),
                         strongly_connected=bool(nscc == 1), n_scc=int(nscc), largest_scc=big, n_wcc=int(nwcc),
                         min_in_degree=int(indeg.min()), mean_degree=float(indeg.mean()), max_in_degree=int(indeg.max()),
                         reciprocity=float(((M != 0) & (M.T != 0)).sum() / (M != 0).sum()),
                         balanced_triangle_fraction=tri['balanced_fraction'], **spec, **par)
        log(f"[{name}] " + json.dumps(out[name]))
    # dynamics on the UNPRUNED subgraph (violating edges kept, same gauge)
    rng = np.random.RandomState(42); x0 = rng.uniform(-1, 1, n) * d; x0[leader] = d[leader] * 0.5
    res = {}
    for name, M in [('before', A_orig), ('after', A_rvgb)]:
        Xf, Xsf = simulate_sparse(sp.csr_matrix(M), d, theta, x0, 300, leader)
        r = abs(Xf[V2].mean() / Xf[V1].mean())
        res[name] = dict(mean_V1=float(Xf[V1].mean()), mean_V2=float(Xf[V2].mean()), ratio=float(r),
                         std_V1=float(Xf[V1].std()), std_V2=float(Xf[V2].std()),
                         max_dev_from_pattern=float(np.abs(Xsf - Xsf[leader]).max()))
        log(f"dynamics on subgraph [{name}]: {json.dumps(res[name])}")
    out['dynamics'] = res
    # representativeness: same extraction from the 20 most adversarial hubs
    log("-- alternative seeds --")
    out_nb, in_nb, all_nodes = SE.build_adjacency_lists(edges)
    degree = defaultdict(int); negd = defaultdict(int)
    for s, t, sg in edges:
        degree[s] += 1; degree[t] += 1
        if sg < 0: negd[s] += 1; negd[t] += 1
    def score(v):
        dd = degree[v]; return 0 if dd < 10 else negd[v] / dd * np.sqrt(dd)
    seeds = sorted(all_nodes, key=score, reverse=True)[:20]
    alt = []
    for sd in seeds:
        try:
            A_s, n_s, ids_s = extract_from_seed(edges, sd, out_nb, in_nb, degree, score, 150)
            comm_s = SE.sign_consistent_partition(A_s)
            d_s = np.zeros(n_s); d_s[comm_s[0]] = 1.0; d_s[comm_s[1]] = -2.0
            fr, ok, bad = compat_fraction(sp.csr_matrix(A_s), d_s)
            alt.append(dict(seed=int(sd), n=int(n_s), m=int((A_s != 0).sum()), compat=float(fr), V1=len(comm_s[0]), V2=len(comm_s[1])))
        except Exception as ex:
            log(f"seed {sd} failed: {ex}")
    comp = np.array([a['compat'] for a in alt])
    log(f"alternative seeds: {len(alt)} subgraphs, compatibility mean={100*comp.mean():.1f}%, std={100*comp.std():.1f}%, min={100*comp.min():.1f}%, max={100*comp.max():.1f}%")
    out['alternative_seeds'] = dict(list=alt, mean=float(comp.mean()), std=float(comp.std()), min=float(comp.min()), max=float(comp.max()))
    # compatibility of the subgraph's nodes under the GLOBAL partition d_full
    sub_idx = np.array([idx_map[v] for v in original_ids])
    d_glob = d_full[sub_idx]
    fr_g, _, _ = compat_fraction(sp.csr_matrix(A_orig), d_glob)
    log(f"subgraph compatibility under the global (full-network) partition: {100*fr_g:.1f}%")
    out['compat_under_global_partition'] = float(fr_g)
    RESULTS['subgraph'] = out

def extract_from_seed(edges, seed, out_nb, in_nb, degree, score, target):
    visited = {seed}; q = deque([seed]); collect = target * 3
    while q and len(visited) < collect:
        u = q.popleft()
        nbs = sorted(set(list(out_nb[u].keys()) + list(in_nb[u].keys())), key=lambda x: -score(x))
        for v in nbs:
            if v not in visited and len(visited) < collect:
                visited.add(v); q.append(v)
    sub_edges = [(s, t, sg) for s, t, sg in edges if s in visited and t in visited]
    scc = SE.largest_scc(visited, sub_edges)
    if len(scc) > target:
        top = sorted(scc, key=lambda v: degree[v], reverse=True)[:target]
        scc = SE.largest_scc(set(top), [(s, t, sg) for s, t, sg in sub_edges if s in top and t in top])
    ids = sorted(scc); mp = {v: k for k, v in enumerate(ids)}; n = len(ids)
    A = np.zeros((n, n))
    for s, t, sg in edges:
        if s in mp and t in mp and s != t: A[mp[s], mp[t]] = sg   # same convention as Experiment 6 code
    return A, n, ids

# ---------------------------------------------------------------------------
# PART C: runtime scaling on synthetic RVGB networks (R3-2)
# ---------------------------------------------------------------------------
def synthetic_rvgb_sparse(n, avg_deg, k=3, rng=None):
    rng = rng or np.random.RandomState(0)
    cvals = np.array([1.0, 2.0, -1.0])[:k]
    d = cvals[rng.randint(0, k, n)]
    m = int(n * avg_deg)
    src = rng.randint(0, n, m); dst = rng.randint(0, n, m)
    ok = src != dst; src, dst = src[ok], dst[ok]
    w = rng.uniform(0.5, 1.5, src.size) * np.sign(d[src] / d[dst])   # sgn(a_ij)=sgn(d_j/d_i)
    A = sp.csr_matrix((w, (dst, src)), shape=(n, n)); A.sum_duplicates()
    return A, d

def part_C_scaling():
    log("\n=== PART C: runtime scaling (R3-2) ===")
    rows = []
    for n in [100, 1000, 10000, 100000]:
        A, d = synthetic_rvgb_sparse(n, 10)
        m = A.nnz
        t0 = time.time(); compat_fraction(A, d); t_check = time.time() - t0
        t0 = time.time(); bfs_sign_gauge(A); t_cycle = time.time() - t0
        Ac = A.tocoo(); At = sp.csr_matrix((Ac.data * d[Ac.col] / d[Ac.row], (Ac.row, Ac.col)), shape=(n, n))
        Z = np.asarray(At.sum(axis=1)).ravel(); inv = np.where(Z != 0, 1 / np.where(Z != 0, Z, 1), 0)
        Qt = sp.diags(inv) @ At; x = np.random.rand(n); th = 0.5
        t0 = time.time()
        for _ in range(10): x = th * x + (1 - th) * (Qt @ x)
        t_sparse = (time.time() - t0) / 10
        t0 = time.time()
        depth = sp.csgraph.breadth_first_order(Qt.T.tocsr(), 0, return_predecessors=False)
        t_tree = time.time() - t0
        t_dense = None; mem_dense = n * n * 8 / 1e9
        if n <= 10000:
            Qd = Qt.toarray(); x = np.random.rand(n)
            t0 = time.time()
            for _ in range(5): x = th * x + (1 - th) * (Qd @ x)
            t_dense = (time.time() - t0) / 5
        row = dict(n=n, m=int(m), t_rvgb_check_ms=1000*t_check, t_cycle_check_ms=1000*t_cycle,
                   t_sparse_step_ms=1000*t_sparse, t_dense_step_ms=None if t_dense is None else 1000*t_dense,
                   dense_memory_GB=mem_dense, t_bfs_tree_ms=1000*t_tree)
        rows.append(row); log(json.dumps(row))
    RESULTS['scaling'] = rows

if __name__ == '__main__':
    SE.download_dataset()
    edges = SE.load_edge_list()
    A_full, n_full, idx_map, nodes, d_full = part_A_full_network(edges)
    part_B_subgraph(edges, A_full, idx_map, d_full)
    part_C_scaling()
    with open(os.path.join(HERE, 'results_revision.json'), 'w') as f:
        json.dump(RESULTS, f, indent=1, default=float)
    log("\nSaved results_revision.json")
