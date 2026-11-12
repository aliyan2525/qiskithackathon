"""
Grid-Aware EV Fleet Charging vs. Transformer Thermal Degradation
================================================================

Qiskit Fall Fest 2026 Hackathon - The Quantum Optimization Challenge
Domain 6: Energy & Smart Infrastructure

Pipeline (one source of truth for the model: THERMAL_COEFFS / TARIFF_VECTOR):

  1. DOMAIN LAYER
     - 2 commercial EVs (EV_A, EV_B), 3 overnight slots each.
     - Tariff/urgency vector c = [8, 2, 4, 7, 2, 5].
     - Thermal stress f_grid(x) = 12 x_A1 x_B1 + 10 x_A2 x_B2 + 8 x_A3 x_B3.
     - Hard SoC equalities: sum(x_A) = 2, sum(x_B) = 2.

  2. QUANTUM ENCODING LAYER
     - Penalty QUBO (QuadraticProgramToQubo, penalty P) -> 6-qubit Ising H_P.
     - Objective-only Ising H_C (no penalty) -> cost layer of the XY-mixer ansatz.
     - Encoding check: the Ising diagonal equals the brute-force energy for all
       64 basis states (catches coefficient, sign and qubit-order mistakes).

  3. HYBRID VARIATIONAL LOOP (Qiskit V2 primitives)
     A. Penalty QAOA:      QAOAAnsatz(H_P, p) from |+>^6, constraints via penalty.
     B. Constraint-preserving QAOA: XY-mixer ansatz from a feasible basis state.
        Hamming weight per EV register is conserved, so every sampled state is
        feasible by construction; the penalty is not needed.
     - COBYLA on StatevectorEstimator; final sampling with StatevectorSampler.

  4. VALIDATION & CLASSICAL BENCHMARKING
     - 64-state brute force, NumPyMinimumEigensolver cross-check (asserted).
     - Naive greedy tariff-only baseline.
     - Penalty-weight sweep (classical, exact), with the exact feasibility threshold.
     - Pareto frontier and weighted-sum weight sweep over the feasible schedules.
     - NISQ transpilation onto a linear coupling map.

Run:
    python main.py [--reps 2] [--restarts 5] [--shots 8192] [--penalty 25.0]

Outputs:
    charts/*.png              - Matplotlib figures
    results/results.json      - machine-readable result summary (incl. runtimes)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate, QAOAAnsatz, XXPlusYYGate
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.transpiler import CouplingMap
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# Global configuration (single source of truth for the model)
# ----------------------------------------------------------------------------
N_SLOTS = 3
NUM_QUBITS = 6  # 2 EVs x 3 slots
SOC_REQUIRED = 2  # each EV must charge in exactly 2 of 3 slots

# Variable order: x = [x_A1, x_A2, x_A3, x_B1, x_B2, x_B3]  (qubit k <-> x_k)
EV_NAMES = ["EV_A", "EV_B"]
SLOT_NAMES = ["Evening Peak", "Midnight Deep Off-Peak", "Pre-Dawn Renewable Ramp"]
VARIABLE_NAMES = ["xA1", "xA2", "xA3", "xB1", "xB2", "xB3"]

# Objective 1: linear ToU tariff & battery-urgency cost
TARIFF_VECTOR = [8.0, 2.0, 4.0, 7.0, 2.0, 5.0]

# Objective 2: thermal stress. Coefficient of the product x_i * x_j itself.
# (QuadraticProgram stores exactly this coefficient for an (i, j) pair, so the
#  same numbers feed the QP, the brute force and the report.)
THERMAL_COEFFS = {(0, 3): 12.0, (1, 4): 10.0, (2, 5): 8.0}

# Constraint-preserving mixer: XY rings inside each EV's 3-slot register.
XY_RINGS = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
# A feasible starting point: x_A = (0,1,1), x_B = (0,1,1)  (2 slots each).
FEASIBLE_INIT_BITS = [0, 1, 1, 0, 1, 1]

DEFAULT_PENALTY = 25.0
DEFAULT_REPS = 2
DEFAULT_RESTARTS = 5
DEFAULT_SHOTS = 8192
SEED = 42
PENALTY_SWEEP = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 1000.0]
WEIGHT_SWEEP_STEPS = 11

ROOT = Path(__file__).resolve().parent
CHARTS_DIR = ROOT / "charts"
RESULTS_DIR = ROOT / "results"

console = Console()
TIMINGS: dict[str, float] = {}


class Timer:
    """Context manager that records wall-clock seconds under a name."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        TIMINGS[self.name] = round(time.perf_counter() - self.t0, 3)


# ----------------------------------------------------------------------------
# 1. DOMAIN LAYER
# ----------------------------------------------------------------------------
def build_quadratic_program(with_constraints: bool = True) -> QuadraticProgram:
    """Constrained scheduling problem (or its objective alone, for the cost Hamiltonian)."""
    name = "grid_aware_ev_charging" if with_constraints else "grid_aware_ev_objective"
    qp = QuadraticProgram(name)
    for var in VARIABLE_NAMES:
        qp.binary_var(var)

    qp.minimize(
        linear=dict(zip(VARIABLE_NAMES, TARIFF_VECTOR)),
        quadratic={
            (VARIABLE_NAMES[i], VARIABLE_NAMES[j]): w for (i, j), w in THERMAL_COEFFS.items()
        },
    )
    if with_constraints:
        qp.linear_constraint({v: 1 for v in VARIABLE_NAMES[:3]}, "==", SOC_REQUIRED, "soc_ev_a")
        qp.linear_constraint({v: 1 for v in VARIABLE_NAMES[3:]}, "==", SOC_REQUIRED, "soc_ev_b")
    return qp


def penalty_expression(bits: np.ndarray, penalty: float) -> float:
    """P * [(sum_A - 2)^2 + (sum_B - 2)^2]  (slack-free equality penalty)."""
    viol_a = bits[:3].sum() - SOC_REQUIRED
    viol_b = bits[3:].sum() - SOC_REQUIRED
    return penalty * (viol_a**2 + viol_b**2)


def evaluate_schedule(bits: np.ndarray, penalty: float) -> dict:
    """Decompose the full objective for one candidate schedule."""
    tariff = float(np.dot(TARIFF_VECTOR, bits))
    grid = float(sum(w * bits[i] * bits[j] for (i, j), w in THERMAL_COEFFS.items()))
    penalty_term = penalty_expression(bits, penalty)
    return {
        "tariff": tariff,
        "grid": grid,
        "penalty_term": penalty_term,
        "total": tariff + grid + penalty_term,
        "feasible": bool(bits[:3].sum() == SOC_REQUIRED and bits[3:].sum() == SOC_REQUIRED),
    }


def bits_from_index(index: int) -> np.ndarray:
    """Basis index -> bit vector; bit k is variable k (= qubit k, little-endian)."""
    return np.array([(index >> k) & 1 for k in range(NUM_QUBITS)], dtype=float)


def brute_force(penalty: float) -> list[dict]:
    """Exhaustively evaluate all 2^6 = 64 candidate schedules."""
    records = []
    for index in range(2**NUM_QUBITS):
        bits = bits_from_index(index)
        rec = evaluate_schedule(bits, penalty)
        rec["bits"] = bits
        rec["label"] = "".join(str(int(b)) for b in bits)  # order x_A1..x_B3
        records.append(rec)
    return records


# ----------------------------------------------------------------------------
# 2. QUANTUM ENCODING LAYER
# ----------------------------------------------------------------------------
def pauli_weight(pauli) -> int:
    """Number of non-identity factors in a Pauli label."""
    return sum(1 for ch in str(pauli) if ch != "I")


def encode_penalty_ising(penalty: float):
    """Constrained problem -> unconstrained Ising H_P (6 Z + 9 ZZ terms)."""
    qp = build_quadratic_program(with_constraints=True)
    qubo = QuadraticProgramToQubo(penalty=penalty).convert(qp)
    operator, offset = qubo.to_ising()

    n_z = sum(1 for p in operator.paulis if pauli_weight(p) == 1)
    n_zz = sum(1 for p in operator.paulis if pauli_weight(p) == 2)
    assert n_z == NUM_QUBITS, f"expected 6 single-qubit Z terms, got {n_z}"
    assert n_zz == 9, f"expected 9 ZZ interaction terms, got {n_zz}"
    return operator, offset, qubo


def encode_objective_ising():
    """Objective-only Ising H_C (no penalty), used as the cost Hamiltonian of the XY ansatz."""
    qp = build_quadratic_program(with_constraints=False)
    qubo = QuadraticProgramToQubo(penalty=1.0).convert(qp)
    operator, offset = qubo.to_ising()
    return operator, offset


def check_encoding(operator, offset: float, penalty: float | None) -> None:
    """Assert the Ising diagonal equals the brute-force energy for every basis state.

    penalty=None checks the objective-only Hamiltonian (no constraint term).
    Qiskit's little-endian convention (qubit k = bit k of the index) is exercised here.
    """
    diag = np.real(np.diag(operator.to_matrix()))
    for index in range(2**NUM_QUBITS):
        bits = bits_from_index(index)
        rec = evaluate_schedule(bits, penalty or 0.0)
        target = rec["tariff"] + rec["grid"] + (rec["penalty_term"] if penalty is not None else 0.0)
        if abs(diag[index] + offset - target) > 1e-9:
            raise RuntimeError(
                f"Encoding mismatch at state {index}: Ising={diag[index] + offset:.6f} "
                f"vs brute-force={target:.6f}"
            )


# ----------------------------------------------------------------------------
# 3. HYBRID VARIATIONAL LOOP (Qiskit V2 primitives)
# ----------------------------------------------------------------------------
def optimize_cobyla(energy_fn, num_params: int, reps: int, restarts: int, seed: int) -> dict:
    """Multi-start COBYLA. energy_fn(x) -> expectation value. Restart 0 uses a fixed start."""
    rng = np.random.default_rng(seed)
    best = {"energy": math.inf, "x": None}
    histories: list[list[float]] = []
    for restart in range(restarts):
        history: list[float] = []
        x0 = np.full(num_params, 0.1) if restart == 0 else rng.uniform(-math.pi, math.pi, size=num_params)

        def objective(x: np.ndarray) -> float:
            energy = energy_fn(x)
            history.append(energy)
            return energy

        res = minimize(
            objective,
            x0=x0,
            method="COBYLA",
            options={"maxiter": 400, "rhobeg": 0.5, "tol": 1e-6},
        )
        histories.append(history)
        if res.fun < best["energy"]:
            best = {"energy": float(res.fun), "x": list(res.x)}
    best["histories"] = histories
    return best


def sample_counts(circuit: QuantumCircuit, shots: int, seed: int) -> dict:
    """Measure a bound circuit with StatevectorSampler; keys in variable order x_A1..x_B3."""
    measured = circuit.copy()
    measured.measure_all()
    sampler = StatevectorSampler(seed=seed, default_shots=shots)
    raw = sampler.run([measured]).result()[0].data.meas.get_counts()
    # Qiskit prints q5..q0 (little-endian); reverse so index 0 is x_A1.
    return {bits[::-1]: n for bits, n in raw.items()}


def run_penalty_qaoa(operator, reps: int, restarts: int, shots: int, seed: int) -> dict:
    """Variant A: QAOAAnsatz on the penalty Hamiltonian, starting from |+>^6."""
    estimator = StatevectorEstimator(default_precision=0.0)
    ansatz = QAOAAnsatz(operator, reps=reps)

    def energy(x) -> float:
        return float(estimator.run([(ansatz, operator, list(x))]).result()[0].data.evs)

    best = optimize_cobyla(energy, 2 * reps, reps, restarts, seed)
    bound = ansatz.assign_parameters(best["x"])
    counts = sample_counts(bound, shots, seed)
    return _package_variant("penalty", best, bound, counts, shots)


def build_xy_ansatz(cost_operator, reps: int):
    """Constraint-preserving ansatz: feasible basis state, cost layer, XY-ring mixer layer.

    Parameters: gammas (cost phases) and betas (mixer angles), 2*reps in total.
    """
    gammas = ParameterVector("gamma", reps)
    betas = ParameterVector("beta", reps)
    qc = QuantumCircuit(NUM_QUBITS, name="xy_qaoa")
    for q, bit in enumerate(FEASIBLE_INIT_BITS):
        if bit:
            qc.x(q)
    for layer in range(reps):
        qc.append(PauliEvolutionGate(cost_operator, time=gammas[layer]), list(range(NUM_QUBITS)))
        for a, b in XY_RINGS:
            # Weight-preserving 2-qubit XY rotation: acts only on {|01>, |10>}.
            qc.append(XXPlusYYGate(2 * betas[layer]), [a, b])
    return qc, gammas, betas


def run_xy_qaoa(cost_operator, reps: int, restarts: int, shots: int, seed: int) -> dict:
    """Variant B: constraint-preserving XY-mixer QAOA; the penalty is not used."""
    estimator = StatevectorEstimator(default_precision=0.0)
    ansatz, gammas, betas = build_xy_ansatz(cost_operator, reps)

    def bind(x) -> QuantumCircuit:
        values = {}
        for layer in range(reps):
            values[gammas[layer]] = float(x[layer])
            values[betas[layer]] = float(x[reps + layer])
        return ansatz.assign_parameters(values)

    def energy(x) -> float:
        return float(estimator.run([(bind(x), cost_operator)]).result()[0].data.evs)

    best = optimize_cobyla(energy, 2 * reps, reps, restarts, seed)
    bound = bind(best["x"])
    counts = sample_counts(bound, shots, seed)
    return _package_variant("xy", best, bound, counts, shots)


def _package_variant(name: str, best: dict, bound: QuantumCircuit, counts: dict, shots: int) -> dict:
    total = sum(counts.values())
    return {
        "name": name,
        "bound_circuit": bound,
        "optimal_params": best["x"],
        "optimal_energy": best["energy"],
        "histories": best["histories"],
        "counts": counts,
        "probabilities": {bits: n / total for bits, n in counts.items()},
        "shots": shots,
    }


# ----------------------------------------------------------------------------
# 4. VALIDATION & CLASSICAL BENCHMARKING
# ----------------------------------------------------------------------------
def exact_ground_state(operator, offset: float) -> float:
    """Exact ground-state energy from Qiskit's NumPyMinimumEigensolver (+ offset)."""
    result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(operator)
    return float(np.real(result.eigenvalue)) + offset


def greedy_baseline(brute: list[dict]) -> dict:
    """Naive scheduler: meets the SoC constraints, minimizes tariff cost alone."""
    feasible = [r for r in brute if r["feasible"]]
    return min(feasible, key=lambda r: r["tariff"])


def penalty_sensitivity(penalties: list[float]) -> list[dict]:
    """Classical (exact, brute-force) penalty sweep."""
    rows = []
    for p in penalties:
        brute = brute_force(p)
        feasible = [r for r in brute if r["feasible"]]
        best_feasible = min(feasible, key=lambda r: r["total"])
        global_min = min(brute, key=lambda r: r["total"])
        rows.append(
            {
                "penalty": p,
                "global_opt_label": global_min["label"],
                "global_opt_cost": global_min["total"],
                "global_opt_feasible": global_min["feasible"],
                "best_feasible_label": best_feasible["label"],
                "best_feasible_cost": best_feasible["total"],
                "n_feasible": len(feasible),
            }
        )
    return rows


def penalty_threshold(step: float = 0.1, max_p: float = 50.0) -> float:
    """Smallest P (on a grid of `step`) for which the global optimum is feasible."""
    for p in np.arange(step, max_p + step / 2, step):
        p = round(float(p), 6)
        if min(brute_force(p), key=lambda r: r["total"])["feasible"]:
            return p
    raise RuntimeError("no feasible global optimum found up to max_p")


def weight_sweep(brute: list[dict], steps: int = WEIGHT_SWEEP_STEPS) -> list[dict]:
    """Weighted-sum sweep  lambda * tariff + (1 - lambda) * thermal  over feasible schedules."""
    feasible = [r for r in brute if r["feasible"]]
    rows = []
    for lam in np.linspace(0.0, 1.0, steps):
        best = min(feasible, key=lambda r: lam * r["tariff"] + (1 - lam) * r["grid"])
        rows.append(
            {
                "lambda_tariff": round(float(lam), 3),
                "label": best["label"],
                "tariff": best["tariff"],
                "grid": best["grid"],
            }
        )
    return rows


def pareto_frontier(brute: list[dict]) -> dict:
    """Non-dominated feasible schedules in (tariff, thermal stress) space."""
    feasible = [r for r in brute if r["feasible"]]
    points = [(r["tariff"], r["grid"], r["label"]) for r in feasible]
    pareto = []
    for i, (t, g, lbl) in enumerate(points):
        dominated = any(
            (t2 <= t and g2 <= g) and (t2 < t or g2 < g)
            for j, (t2, g2, _) in enumerate(points)
            if j != i
        )
        if not dominated:
            pareto.append((t, g, lbl))
    pareto.sort()
    return {"points": points, "pareto": pareto}


def nisq_transpilation(bound_circuit: QuantumCircuit, seed: int) -> dict:
    """Transpile to rz/sx/x/cx, with all-to-all connectivity and with a linear coupling map."""
    basis = ["rz", "sx", "x", "cx"]
    unrouted = transpile(bound_circuit, basis_gates=basis, optimization_level=3, seed_transpiler=seed)
    routed = transpile(
        bound_circuit,
        basis_gates=basis,
        coupling_map=CouplingMap.from_line(NUM_QUBITS),
        optimization_level=3,
        seed_transpiler=seed,
    )
    u_ops, r_ops = unrouted.count_ops(), routed.count_ops()
    return {
        "raw_cx": int(u_ops.get("cx", 0)),
        "raw_depth": int(unrouted.depth()),
        "native_cx": int(r_ops.get("cx", 0)),
        "native_depth": int(routed.depth()),
        "native_size": int(routed.size()),
    }


def package_versions() -> dict:
    out = {}
    for pkg in ["qiskit", "qiskit-algorithms", "qiskit-optimization", "numpy", "scipy", "matplotlib"]:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not installed"
    return out


# ----------------------------------------------------------------------------
# Chart generation
# ----------------------------------------------------------------------------
def make_charts(results: dict) -> None:
    CHARTS_DIR.mkdir(exist_ok=True)
    brute = results["brute_force"]
    variants = results["variants"]
    feasible_map = {r["label"]: r["feasible"] for r in brute}
    optimal_label = results["optimal_label"]
    exact = results["classical_min_energy"]

    # 1. Convergence (both variants) -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, key, title in zip(
        axes,
        ["penalty", "xy"],
        ["Penalty QAOA (QAOAAnsatz)", "XY-mixer QAOA (feasible subspace)"],
    ):
        for i, hist in enumerate(variants[key]["histories"]):
            ax.plot(hist, lw=1.1, alpha=0.85, label=f"restart {i + 1}")
        ax.axhline(exact, color="black", ls="--", lw=1.2, label="exact ground state")
        ax.set_xlabel("COBYLA evaluation")
        ax.set_ylabel(r"$\langle H \rangle$ (objective energy)")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "qaoa_convergence.png", dpi=150)
    plt.close(fig)

    # 2. Sampled distributions (top 12, both variants) ------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in zip(
        axes, ["penalty", "xy"], ["Penalty QAOA", "XY-mixer QAOA"]
    ):
        top = sorted(variants[key]["probabilities"].items(), key=lambda kv: kv[1], reverse=True)[:12]
        labels = [k for k, _ in top]
        probs = [v for _, v in top]
        colors = [
            "#2e7d32" if lbl == optimal_label else "#1976d2" if feasible_map[lbl] else "#b71c1c"
            for lbl in labels
        ]
        ax.bar(range(len(labels)), probs, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("sampling probability")
        ax.set_xlabel("bitstring (x_A1 x_A2 x_A3 x_B1 x_B2 x_B3)")
        ax.set_title(f"{title} ({variants[key]['shots']} shots)")
    fig.suptitle("green: optimum   blue: feasible   red: infeasible", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "qaoa_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Variant comparison ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["Penalty QAOA", "XY-mixer QAOA"]
    p_opt = [variants["penalty"]["p_optimum"], variants["xy"]["p_optimum"]]
    p_feas = [variants["penalty"]["p_feasible"], variants["xy"]["p_feasible"]]
    x = np.arange(2)
    ax.bar(x - 0.18, p_opt, 0.36, color="#2e7d32", label="P(global optimum)")
    ax.bar(x + 0.18, p_feas, 0.36, color="#1976d2", label="P(feasible schedule)")
    for xi, (o, f) in enumerate(zip(p_opt, p_feas)):
        ax.text(xi - 0.18, o + 0.01, f"{o:.1%}", ha="center", fontsize=8)
        ax.text(xi + 0.18, f + 0.01, f"{f:.1%}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("probability (sampled)")
    ax.set_title("Constraint handling: penalty vs. constraint-preserving mixer")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "variant_comparison.png", dpi=150)
    plt.close(fig)

    # 4. Penalty sensitivity (classical, exact) -------------------------------
    sweep = results["penalty_sweep"]
    ps = [r["penalty"] for r in sweep]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ps, [r["global_opt_cost"] for r in sweep], "o-", color="#b71c1c",
            label="global minimum of penalised objective")
    ax.plot(ps, [r["best_feasible_cost"] for r in sweep], "s-", color="#2e7d32",
            label="best feasible schedule cost")
    ax.axvline(results["config"]["penalty"], color="gray", ls=":", lw=1)
    ax.axvline(results["penalty_threshold"], color="#6a1b9a", ls="--", lw=1,
               label=f"feasibility threshold P* = {results['penalty_threshold']:g}")
    ax.set_xlabel("penalty weight P")
    ax.set_ylabel("objective value")
    ax.set_xscale("log")
    ax.set_title("Penalty weight sensitivity (exact brute force)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "penalty_sensitivity.png", dpi=150)
    plt.close(fig)

    # 5. Pareto frontier and weight sweep -------------------------------------
    pareto = results["pareto"]
    pts = pareto["points"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70, color="#90a4ae",
               label="feasible schedules (9)")
    pfront = pareto["pareto"]
    ax.step(
        [p[0] for p in pfront] + [max(p[0] for p in pfront) + 2],
        [p[1] for p in pfront] + [pfront[-1][1]],
        where="post", color="#1565c0", lw=2, label="Pareto frontier",
    )
    for t, g, lbl in pts:
        ax.annotate(lbl, (t, g), fontsize=7, textcoords="offset points", xytext=(4, 3))
    chosen = {r["label"] for r in results["weight_sweep"]}
    for t, g, lbl in pts:
        if lbl in chosen:
            ax.scatter([t], [g], s=160, facecolors="none", edgecolors="#ef6c00", lw=1.8)
    opt = next(p for p in pts if p[2] == results["optimal_label"])
    ax.scatter([opt[0]], [opt[1]], s=260, facecolors="none", edgecolors="#2e7d32", lw=2.5,
               label=f"global optimum |{results['optimal_label']}>")
    ax.scatter([], [], s=160, facecolors="none", edgecolors="#ef6c00", lw=1.8,
               label="chosen by weighted-sum sweep")
    ax.set_xlabel("tariff & urgency cost (objective 1)")
    ax.set_ylabel("transformer thermal stress (objective 2)")
    ax.set_title("Pareto frontier: energy cost vs. transformer thermal stress")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "pareto_frontier.png", dpi=150)
    plt.close(fig)

    console.print(f"[green]Charts written to[/green] {CHARTS_DIR}")


# ----------------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------------
def _summarize_variant(name: str, variant: dict, brute_by_label: dict, opt_label: str) -> dict:
    probs = variant["probabilities"]
    cost_of = {lbl: rec["total"] for lbl, rec in brute_by_label.items()}
    best_sampled = min(probs, key=lambda l: cost_of[l])
    variant["p_optimum"] = probs.get(opt_label, 0.0)
    variant["p_feasible"] = sum(p for l, p in probs.items() if brute_by_label[l]["feasible"])
    return {
        "name": name,
        "optimal_energy": variant["optimal_energy"],
        "best_sampled_label": best_sampled,
        "best_sampled_cost": cost_of[best_sampled],
        "p_optimum": variant["p_optimum"],
        "p_feasible": variant["p_feasible"],
        "top_counts": dict(sorted(variant["counts"].items(), key=lambda kv: kv[1], reverse=True)[:16]),
        "optimal_params": variant["optimal_params"],
        "histories": variant["histories"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="QAOA depth p")
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS, help="multi-start count")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help="final sampling shots")
    parser.add_argument("--penalty", type=float, default=DEFAULT_PENALTY, help="penalty weight P")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    t_start = time.perf_counter()

    console.print(
        Panel.fit(
            "[bold]Grid-Aware EV Fleet Charging vs. Transformer Thermal Degradation[/bold]\n"
            "Qiskit Fall Fest 2026 - Quantum Optimization Challenge (Domain 6)",
            border_style="cyan",
        )
    )

    # --- Domain + encoding layers -----------------------------------------------
    with Timer("encoding"):
        operator, offset, _qubo = encode_penalty_ising(args.penalty)
        cost_op, cost_offset = encode_objective_ising()
        check_encoding(operator, offset, args.penalty)
        check_encoding(cost_op, cost_offset, None)
    n_z = sum(1 for p in operator.paulis if pauli_weight(p) == 1)
    n_zz = sum(1 for p in operator.paulis if pauli_weight(p) == 2)
    console.print(
        f"[cyan]Encoding layer:[/cyan] QUBO -> Ising | {n_z} Z + {n_zz} ZZ terms | "
        f"64-state encoding check passed (penalty and objective Hamiltonians)"
    )

    # --- Classical brute force -----------------------------------------------------
    brute = brute_force(args.penalty)
    brute_by_label = {r["label"]: r for r in brute}
    n_feasible = sum(1 for r in brute if r["feasible"])
    global_min = min(brute, key=lambda r: r["total"])
    best_feasible = min((r for r in brute if r["feasible"]), key=lambda r: r["total"])
    greedy = greedy_baseline(brute)
    opt_label = best_feasible["label"]
    if global_min["label"] != opt_label:
        raise RuntimeError("penalised global optimum is infeasible at this P; increase --penalty")

    table = Table(title="Classical 64-state brute force", title_style="bold")
    for col in ["Schedule (x_A1..x_B3)", "Tariff", "Thermal", "Penalty", "Total", "Status"]:
        table.add_column(col)
    for rec, name in [(best_feasible, "Global optimum"), (greedy, "Naive greedy (tariff-only)")]:
        table.add_row(
            f"|{rec['label']}>  {name}",
            f"{rec['tariff']:.1f}",
            f"{rec['grid']:.1f}",
            f"{rec['penalty_term']:.1f}",
            f"{rec['total']:.1f}",
            "[green]feasible[/green]" if rec["feasible"] else "[red]INFEASIBLE[/red]",
        )
    console.print(table)
    console.print(f"Feasible states: {n_feasible}/64 ({100 * n_feasible / 64:.2f}%)")

    # --- Exact eigensolver (asserted against brute force) ------------------------------
    with Timer("exact_eigensolver"):
        classical_energy = exact_ground_state(operator, offset)
    if abs(classical_energy - global_min["total"]) > 1e-6:
        raise RuntimeError(
            f"NumPyMinimumEigensolver gave {classical_energy}, brute force gives {global_min['total']}"
        )
    console.print(
        f"[cyan]Exact baseline:[/cyan] NumPyMinimumEigensolver = {classical_energy:.2f} "
        f"== brute force {global_min['total']:.2f}"
    )

    # --- Hybrid loops ---------------------------------------------------------------------
    console.print(
        f"[cyan]Hybrid loops:[/cyan] p={args.reps}, {args.restarts} COBYLA restarts, "
        f"{args.shots} shots per variant"
    )
    with Timer("qaoa_penalty"):
        penalty_variant = run_penalty_qaoa(operator, args.reps, args.restarts, args.shots, args.seed)
    with Timer("qaoa_xy"):
        xy_variant = run_xy_qaoa(cost_op, args.reps, args.restarts, args.shots, args.seed)

    variants = {
        "penalty": _summarize_variant("penalty", penalty_variant, brute_by_label, opt_label),
        "xy": _summarize_variant("xy", xy_variant, brute_by_label, opt_label),
    }
    # Charts need probabilities/shots/histories; keep the full variant dicts for them.
    chart_variants = {
        "penalty": {**variants["penalty"], "probabilities": penalty_variant["probabilities"], "shots": args.shots},
        "xy": {**variants["xy"], "probabilities": xy_variant["probabilities"], "shots": args.shots},
    }

    qaoa_table = Table(title="QAOA results (sampled)", title_style="bold")
    qaoa_table.add_column("Metric")
    qaoa_table.add_column("Penalty QAOA")
    qaoa_table.add_column("XY-mixer QAOA")
    qaoa_table.add_row(
        "Optimized objective (expectation)",
        f"{variants['penalty']['optimal_energy'] + offset:.2f}",
        f"{variants['xy']['optimal_energy'] + cost_offset:.2f}",
    )
    qaoa_table.add_row(
        "P(global optimum |" + opt_label + ">)",
        f"{variants['penalty']['p_optimum']:.4f}",
        f"{variants['xy']['p_optimum']:.4f}",
    )
    qaoa_table.add_row(
        "P(feasible schedule)",
        f"{variants['penalty']['p_feasible']:.4f}",
        f"{variants['xy']['p_feasible']:.4f}",
    )
    qaoa_table.add_row(
        "Best sampled bitstring",
        f"|{variants['penalty']['best_sampled_label']}> (cost {variants['penalty']['best_sampled_cost']:.1f})",
        f"|{variants['xy']['best_sampled_label']}> (cost {variants['xy']['best_sampled_cost']:.1f})",
    )
    console.print(qaoa_table)

    # --- Classical sensitivity, Pareto and weight sweep ------------------------------------
    with Timer("classical_sweeps"):
        sweep = penalty_sensitivity(PENALTY_SWEEP)
        threshold = penalty_threshold()
        pareto = pareto_frontier(brute)
        wsweep = weight_sweep(brute)

    sweep_table = Table(title="Penalty weight (P) sensitivity (exact)", title_style="bold")
    for col in ["P", "Global minimum", "Cost", "Feasible?", "Best feasible cost"]:
        sweep_table.add_column(col)
    for r in sweep:
        sweep_table.add_row(
            f"{r['penalty']:g}",
            f"|{r['global_opt_label']}>",
            f"{r['global_opt_cost']:.1f}",
            "[green]yes[/green]" if r["global_opt_feasible"] else "[red]NO[/red]",
            f"{r['best_feasible_cost']:.1f}",
        )
    console.print(sweep_table)
    console.print(f"Feasibility threshold: global minimum is feasible for P >= {threshold:g} (grid step 0.1)")
    console.print(
        f"Pareto analysis: {len(pareto['points'])} feasible schedules, {len(pareto['pareto'])} Pareto-optimal"
    )

    # --- NISQ transpilation --------------------------------------------------------------------
    with Timer("transpilation"):
        transp_penalty = nisq_transpilation(penalty_variant["bound_circuit"], args.seed)
        transp_xy = nisq_transpilation(xy_variant["bound_circuit"], args.seed)
    transp_table = Table(title="NISQ transpilation (rz/sx/x/cx basis)", title_style="bold")
    for col in ["Metric", "Penalty: all-to-all", "Penalty: linear map", "XY: all-to-all", "XY: linear map"]:
        transp_table.add_column(col)
    transp_table.add_row(
        "CX gates", str(transp_penalty["raw_cx"]), str(transp_penalty["native_cx"]),
        str(transp_xy["raw_cx"]), str(transp_xy["native_cx"]),
    )
    transp_table.add_row(
        "Depth", str(transp_penalty["raw_depth"]), str(transp_penalty["native_depth"]),
        str(transp_xy["raw_depth"]), str(transp_xy["native_depth"]),
    )
    console.print(transp_table)

    # --- Persist results ---------------------------------------------------------------------------
    TIMINGS["total"] = round(time.perf_counter() - t_start, 3)
    results = {
        "config": {
            "reps": args.reps,
            "restarts": args.restarts,
            "shots": args.shots,
            "penalty": args.penalty,
            "seed": args.seed,
            "optimizer": "COBYLA (maxiter 400, rhobeg 0.5, tol 1e-6)",
        },
        "versions": package_versions(),
        "timings_seconds": TIMINGS,
        "ising": {"n_z_terms": n_z, "n_zz_terms": n_zz, "offset": offset, "objective_offset": cost_offset},
        "n_feasible": n_feasible,
        "classical_min_energy": global_min["total"],
        "exact_eigensolver_energy": classical_energy,
        "optimal_label": opt_label,
        "optimal_cost": best_feasible["total"],
        "greedy_label": greedy["label"],
        "greedy_cost": greedy["total"],
        "greedy_tariff": greedy["tariff"],
        "greedy_thermal": greedy["grid"],
        "variants": variants,
        "penalty_sweep": sweep,
        "penalty_threshold": threshold,
        "weight_sweep": wsweep,
        "pareto": {
            "points": [list(p) for p in pareto["points"]],
            "pareto": [list(p) for p in pareto["pareto"]],
        },
        "transpilation": {"penalty": transp_penalty, "xy": transp_xy},
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    console.print(f"[green]Results written to[/green] {RESULTS_DIR / 'results.json'}")

    make_charts(
        {
            "brute_force": brute,
            "variants": chart_variants,
            "classical_min_energy": classical_energy,
            "optimal_label": opt_label,
            "penalty_sweep": sweep,
            "penalty_threshold": threshold,
            "weight_sweep": wsweep,
            "pareto": pareto,
            "config": results["config"],
        }
    )
    console.print(f"[dim]Total runtime: {TIMINGS['total']:.1f} s[/dim]")


if __name__ == "__main__":
    main()
