# RVGB-polarization

Source code for the paper

> S. Yan, Y. Sun, L. Shi, and Y. Huang, "Opinion Polarization Over Signed Social Networks With Real-Valued Gauge Balance," submitted to *IEEE Transactions on Computational Social Systems* (manuscript TCSS-2026-07-1138).

All scripts are plain Python 3 (tested with Python 3.10, NumPy 2.2, SciPy 1.15, Matplotlib 3.10). Figures use LaTeX text rendering; set `rcParams['text.usetex'] = False` at the top of a script if LaTeX is not installed.

## Contents

| File | Purpose |
|---|---|
| `RVGB_simulation_50nodes.py` | Experiments 1–5: synthetic RVGB network generation (`build_rvgb_network`), gauge transformation, synchronous/asynchronous update rule (7) (`simulate_rvgb`), leaderless and three-community cases, failure case, malicious agent (`simulate_rvgb_malicious`). Produces `fig4a/b`, `fig5a/b`, `fig6a/b`. |
| `plot_topology.py` | Topologies of Experiments 1 and 2 (`fig2`, `fig3`). |
| `plot_balance_examples.py` | Fig. 1 (SB / QSB / RVGB examples). |
| `RVGB_slashdot_experiment.py` | Experiment 6, original pipeline: download of Slashdot Zoo (`soc-sign-Slashdot081106`) from SNAP, extraction of the 149-node strongly connected subgraph around an adversarial hub, spectral bipartition, gauge assignment, removal of RVGB-violating edges, simulation and plotting. |
| `experiment6_R1.py` | Experiment 6, revised (R1): same extraction, then restriction to the strongly connected core after edge removal; computes all quantities of Table I (before/after) and regenerates Fig. 8 (`fig7b_R1.pdf`, `fig7c_R1.pdf`) and a draft of Fig. 7 (the published Fig. 7 was finished by hand from this draft); writes `exp6_R1.json`. |
| `letter_unbounded_bias.py` | Supplementary robustness check (not in the paper): Experiment 5 network under bounded time-varying, linearly growing and state-dependent biases. |
| `revision_analysis.py` | (a) RVGB compatibility of the **entire** Slashdot Zoo network (breadth-first sign gauge of Theorem 2, greedy refinement, signed-triangle census), sparse simulation on the projected 24,421-node network; (b) representativeness over the 20 most adversarial hubs; (c) running-time measurements of Table II. Writes `results_revision.json`. |

## Reproducing the results

```bash
pip install -r requirements.txt
python3 RVGB_simulation_50nodes.py      # Experiments 1-5
python3 plot_topology.py                # Figs. 2-3
python3 RVGB_slashdot_experiment.py     # downloads the dataset (~10 MB) and runs the original Experiment 6
python3 experiment6_R1.py               # Table I and Fig. 8 (revised Experiment 6)
python3 revision_analysis.py            # full-network analysis and Table II (about 1 minute)
```

The Slashdot Zoo dataset is downloaded automatically from https://snap.stanford.edu/data/soc-sign-Slashdot081106.html.

## Conventions

The adjacency matrix follows the paper: `A[i, j] != 0` iff agent `i` is influenced by agent `j`. For the Slashdot data, a record `(src, dst, sign)` means that user `src` tagged `dst` as friend (+1) or foe (-1), and is stored as `A[src, dst] = sign` (the rater is influenced by the rated user).

## Data

All numbers reported in Tables I–II and in Section VII-B of the paper are regenerated exactly by the scripts above (all random seeds are fixed). Each script writes its figures and a JSON file with the computed quantities next to itself.

## License

MIT License, see `LICENSE`.
