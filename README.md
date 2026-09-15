# RVGB-polarization

Source code for the paper

> S. Yan, Y. Sun, L. Shi, and Y. Huang, "Opinion Polarization Over Signed Social Networks With Real-Valued Gauge Balance," submitted to *IEEE Transactions on Computational Social Systems* (manuscript TCSS-2026-07-1138).

All scripts are plain Python 3 (tested with Python 3.10, NumPy 2.2, SciPy 1.15, Matplotlib 3.10; `generate_fig7_ppt_R1.py` additionally needs python-pptx 1.0). Figures use LaTeX text rendering; set `rcParams['text.usetex'] = False` at the top of a script if LaTeX is not installed.

## Contents

| File | Purpose |
|---|---|
| `RVGB_simulation_50nodes.py` | Experiments 1–5: synthetic RVGB network generation (`build_rvgb_network`), gauge transformation, synchronous/asynchronous update rule (7) (`simulate_rvgb`), leaderless and three-community cases, failure case, malicious agent (`simulate_rvgb_malicious`). Produces `fig4a/b`, `fig5a/b`, `fig6a/b`. |
| `plot_topology.py` | Topologies of Experiments 1 and 2 (`fig2`, `fig3`). |
| `plot_balance_examples.py` | Fig. 1 (SB / QSB / RVGB examples). |
| `RVGB_slashdot_experiment.py` | Experiment 6, original pipeline: download of Slashdot Zoo (`soc-sign-Slashdot081106`) from SNAP, extraction of the 149-node strongly connected subgraph around an adversarial hub, spectral bipartition, gauge assignment, removal of RVGB-violating edges, simulation and plotting. |
| `experiment6_R1.py` | Experiment 6, revised (R1): same extraction, then restriction to the strongly connected core after edge removal; computes all quantities of Table I (before/after) and regenerates `fig7b_R1.pdf`, `fig7c_R1.pdf` (Fig. 8) and a matplotlib draft of Fig. 7; writes `exp6_R1.json`. |
| `generate_fig7_ppt_R1.py` | Fig. 7 (revised): rebuilds the editable PowerPoint figure `Fig7_R1.pptx` from the original hand-finished `Fig7.pptx` (same node positions, node style, edge style and legend), keeping only the 140-node strongly connected core and redrawing its edges; `fig7a_R1.pdf` is the slide exported from PowerPoint and cropped. |
| `letter_unbounded_bias.py` | Supplementary robustness check (not in the paper): Experiment 5 network under bounded time-varying, linearly growing and state-dependent biases; writes `letter_unbounded_bias.json`. |
| `Fig7.pptx`, `Fig7_R1.pptx` | Editable PowerPoint sources of Fig. 7 (original 149-node subgraph, and the 140-node core used in the paper). |
| `revision_analysis.py` | (a) RVGB compatibility of the **entire** Slashdot Zoo network (breadth-first sign gauge of Theorem 2, greedy refinement, signed-triangle census), sparse simulation on the projected 24,421-node network; (b) representativeness over the 20 most adversarial hubs; (c) running-time measurements of Table II. Writes `results_revision.json`. |

## Reproducing the results

```bash
pip install -r requirements.txt
python3 RVGB_simulation_50nodes.py      # Experiments 1-5
python3 plot_topology.py                # Figs. 2-3
python3 RVGB_slashdot_experiment.py     # downloads the dataset (~10 MB) and runs the original Experiment 6
python3 experiment6_R1.py               # Table I and Fig. 8 (revised Experiment 6)
python3 generate_fig7_ppt_R1.py         # Fig. 7 as an editable PowerPoint file
python3 revision_analysis.py            # full-network analysis and Table II (about 1 minute)
```

The Slashdot Zoo dataset is downloaded automatically from https://snap.stanford.edu/data/soc-sign-Slashdot081106.html.

## Conventions

The adjacency matrix follows the paper: `A[i, j] != 0` iff agent `i` is influenced by agent `j`. For the Slashdot data, a record `(src, dst, sign)` means that user `src` tagged `dst` as friend (+1) or foe (-1), and is stored as `A[src, dst] = sign` (the rater is influenced by the rated user).

## Data

All numbers reported in Tables I–II and in Section VII-B of the paper are regenerated exactly by the scripts above (all random seeds are fixed). The JSON files in this repository (`exp6_R1.json`, `results_revision.json`, `letter_unbounded_bias.json`) are the outputs of the last run.

## License

MIT License, see `LICENSE`.
