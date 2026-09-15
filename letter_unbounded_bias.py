"""
Numerical illustration for the response to Reviewer 1, Comment 2 (Remark 9):
behaviour of the Experiment 5 network under biases that are not covered by
Assumption 1.  Same topology, gauge, theta, initial opinions and malicious
agent as Experiment 5 (seed 100); the bias w_f(t) injected by v_f is
  (a) constant 0.3               (Experiment 5, scenario (a), for reference)
  (b) bounded time-varying 0.3 sin(0.1 t)
  (c) linearly growing 0.01 t
  (d) state-dependent kappa * max_i |x_i(t)| for kappa = 0.05 and 0.5.
Reported: max_i |x_i(t) - d_i x*_nom| at selected t.  Letter only.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import RVGB_simulation_50nodes as S


def simulate(A, d, theta, x0, T, leader, f, bias):
    """bias(t, x) -> w_f(t) in original coordinates."""
    n = A.shape[0]
    W = S.normalize_weights(S.gauge_transform(A, d), leader)
    Xs = np.zeros((T + 1, n)); Xs[0] = x0 / d
    for t in range(T):
        x = Xs[t] * d
        Xs[t + 1] = Xs[t].copy()
        for i in range(n):
            if i == leader: continue
            Xs[t + 1, i] = theta[i] * Xs[t, i] + (1 - theta[i]) * np.dot(W[i], Xs[t])
        Xs[t + 1, f] += bias(t, x) / d[f]
    return Xs * d[None, :]


def main():
    n = 50; leader = n - 1; f = 0
    V1 = list(range(25)); V2 = list(range(25, 50)); d = np.ones(n); d[25:] = -2.0
    rng = np.random.RandomState(100)
    A = S.build_rvgb_network(n, [V1, V2], d, p_intra=0.25, p_inter=0.10, w_range=(0.5, 1.5), leader_idx=leader, rng=rng)
    theta = np.full(n, 0.5); theta[leader] = 1.0
    x0 = rng.uniform(-1, 1, n)
    T = 400
    Xn = simulate(A, d, theta, x0, T, leader, f, lambda t, x: 0.0)
    xstar = Xn[-1] / d; nominal = d * xstar[leader]
    cases = {
        'constant 0.3': lambda t, x: 0.3,
        'sin 0.3 sin(0.1 t)': lambda t, x: 0.3 * np.sin(0.1 * t),
        'linear 0.01 t': lambda t, x: 0.01 * t,
        'state kappa=0.05': lambda t, x: 0.05 * np.abs(x).max(),
        'state kappa=0.5': lambda t, x: 0.5 * np.abs(x).max(),
    }
    out = {}
    for name, b in cases.items():
        X = simulate(A, d, theta, x0, T, leader, f, b)
        dev = np.abs(X - nominal[None, :]).max(axis=1)
        out[name] = {f't={t}': float(dev[t]) for t in (100, 200, 300, 400)}
        if name.startswith('linear'):
            out[name]['slope_200_400'] = float((dev[400] - dev[200]) / 200)
        print(f'{name:22s}', ' '.join(f'{k}:{v:9.3g}' for k, v in out[name].items()))
    json.dump(out, open(os.path.join(HERE, 'letter_unbounded_bias.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
