"""
Regression tests for the EV-charging QAOA project.

Run from the project root:
    python -m pytest -q tests        # or
    python tests/test_model.py       # plain-Python runner, no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as m  # noqa: E402  (import after sys.path update)

BRUTE_P25 = m.brute_force(m.DEFAULT_PENALTY)
FEASIBLE = [r for r in BRUTE_P25 if r["feasible"]]


def test_feasible_set_size():
    assert len(FEASIBLE) == 9


def test_global_optimum_is_staggered_schedule():
    best = min(BRUTE_P25, key=lambda r: r["total"])
    assert best["label"] == "011110"
    assert best["total"] == 25.0
    assert best["tariff"] == 15.0 and best["grid"] == 10.0


def test_greedy_baseline_cost():
    greedy = m.greedy_baseline(BRUTE_P25)
    assert greedy["label"] == "011011"
    assert greedy["total"] == 31.0


def test_thermal_coefficients_are_12_10_8():
    # |111111> makes every thermal product 1, so the thermal term equals the sum of coefficients.
    assert m.evaluate_schedule(np.ones(6), 0.0)["grid"] == 12.0 + 10.0 + 8.0


def test_penalty_threshold_is_between_10_and_13():
    threshold = m.penalty_threshold()
    assert 10.0 < threshold <= 12.5


def test_qubo_ising_encoding_matches_brute_force():
    operator, offset, _ = m.encode_penalty_ising(m.DEFAULT_PENALTY)
    m.check_encoding(operator, offset, m.DEFAULT_PENALTY)


def test_objective_hamiltonian_matches_brute_force():
    operator, offset = m.encode_objective_ising()
    m.check_encoding(operator, offset, None)


def test_exact_eigensolver_agrees_with_brute_force():
    operator, offset, _ = m.encode_penalty_ising(m.DEFAULT_PENALTY)
    assert abs(m.exact_ground_state(operator, offset) - 25.0) < 1e-6


def test_xy_ansatz_stays_in_feasible_subspace():
    """Every amplitude outside the 9 feasible schedules must be exactly zero."""
    from qiskit.quantum_info import Statevector

    cost_op, _ = m.encode_objective_ising()
    ansatz, gammas, betas = m.build_xy_ansatz(cost_op, reps=2)
    values = {gammas[0]: 0.37, gammas[1]: -1.1, betas[0]: 0.9, betas[1]: 0.21}
    state = Statevector(ansatz.assign_parameters(values))
    probs = state.probabilities()
    feasible_idx = [i for i in range(64) if m.evaluate_schedule(m.bits_from_index(i), 0.0)["feasible"]]
    leaked = probs.sum() - probs[feasible_idx].sum()
    assert leaked < 1e-10, f"probability leaked outside feasible subspace: {leaked}"


def test_weight_sweep_endpoints():
    sweep = m.weight_sweep(BRUTE_P25)
    assert sweep[0]["label"] == "011101"  # lambda = 0: pure thermal minimum (thermal 8)
    assert sweep[-1]["label"] == "011011"  # lambda = 1: pure tariff minimum (tariff 13)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
