"""
Report builder for the Grid-Aware EV Fleet Charging project.

Reads results/results.json and charts/*.png produced by main.py and writes the
.docx submission report. Every number in the report comes from results.json.

Run (after main.py):
    python build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent
CHARTS_DIR = ROOT / "charts"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
OUTPUT = ROOT / "report" / "Grid_Aware_EV_Charging_QAOA_Report.docx"

LIMITATIONS = [
    (
        "Penalty weight: a feasibility threshold, not a tuning knob",
        "Below the threshold P* found by the exact sweep, the penalised global minimum is an "
        "infeasible schedule (it under-charges at least one EV). Above P*, the optimum is the "
        "same feasible schedule. The penalty-QAOA results depend on P; the XY-mixer variant does "
        "not use a penalty at all.",
    ),
    (
        "All-to-all ZZ connectivity vs. hardware SWAP overhead",
        "Penalty terms couple every pair of variables inside each EV's register, and the thermal "
        "term couples the two registers. On a linear coupling map, routing these gates adds CX "
        "gates. The transpilation table quantifies this for both circuits; it is a linear-map "
        "proxy, not a measurement on IBM hardware.",
    ),
    (
        "Scale and quantum advantage",
        "The instance has 9 feasible schedules, so classical brute force finishes instantly. "
        "This is a benchmark instance that checks the encoding and the algorithm end to end. "
        "No speedup is claimed. The approach scales by reusing the same encoding for larger "
        "fleets, where the feasible set grows combinatorially.",
    ),
    (
        "Classical optimizer behaviour",
        "COBYLA is gradient-free and the QAOA energy landscape can have local minima. "
        "Five multi-starts are used and every restart's history is saved in results.json and "
        "plotted. We do not claim which specific states trap the optimizer.",
    ),
    (
        "Mixer design and feasibility",
        "The XY-mixer ansatz keeps the Hamming weight of each EV register fixed, so sampled "
        "schedules are always feasible for this equality-constrained instance. Extending to "
        "inequality constraints (for example, a transformer capacity cap) would need slack "
        "qubits or a different mixer, which is outside this submission.",
    ),
]


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)


def add_figure(doc: Document, path: Path, caption: str, width: float = 6.2) -> None:
    if not path.exists():
        doc.add_paragraph(f"[missing figure: {path.name}]")
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in cap.runs:
        run.italic = True
        run.font.size = Pt(9)


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def main() -> None:
    results = json.loads((RESULTS_DIR / "results.json").read_text())
    OUTPUT.parent.mkdir(exist_ok=True)

    cfg = results["config"]
    var = results["variants"]
    pen, xy = var["penalty"], var["xy"]
    transp = results["transpilation"]
    opt = results["optimal_label"]
    greedy = results["greedy_label"]
    overload_greedy = results["greedy_thermal"]
    tariff_greedy = results["greedy_tariff"]

    doc = Document()

    title = doc.add_heading("Grid-Aware EV Fleet Charging vs. Transformer Thermal Degradation", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "Qiskit Fall Fest 2026 Hackathon - The Quantum Optimization Challenge\n"
        "Domain 6: Energy & Smart Infrastructure\n"
        "Hybrid quantum-classical QAOA with Qiskit V2 primitives"
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Abstract ------------------------------------------------------------------
    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "We schedule overnight charging for two commercial EVs across three Time-of-Use windows. "
        "The objective combines tariff/urgency cost with quadratic distribution-transformer thermal "
        "stress, under hard state-of-charge equalities. The problem has 6 binary variables. We solve "
        "it two ways with QAOA: (A) a penalty formulation, and (B) a constraint-preserving XY-mixer "
        f"ansatz. Both are checked against a 64-state brute force and Qiskit's NumPyMinimumEigensolver. "
        f"The exact optimum is |{opt}> with cost {results['optimal_cost']:.1f}; the tariff-only greedy "
        f"schedule |{greedy}> costs {results['greedy_cost']:.1f}. The XY-mixer ansatz returns feasible "
        f"schedules with probability {xy['p_feasible']:.2f}, versus {pen['p_feasible']:.2f} for the penalty "
        f"formulation. Optimum probabilities are {xy['p_optimum']:.3f} (XY) and {pen['p_optimum']:.3f} (penalty) "
        f"at p = {cfg['reps']}."
    )

    # 1. Problem ---------------------------------------------------------------
    doc.add_heading("1. Problem identification and intended users", level=1)
    doc.add_paragraph(
        "Urban commercial EV depots and municipal microgrid operators schedule overnight charging "
        "under time-of-use tariffs. When several heavy-duty EVs charge in the same cheap window, "
        "transformer current rises and resistive (I^2R) heating grows quadratically in the "
        "overlap, which accelerates insulation ageing. The operator must balance (1) electricity "
        "and battery-urgency cost against (2) transformer thermal stress, while every vehicle still "
        "reaches its overnight state of charge (SoC)."
    )

    # 2. Formulation -------------------------------------------------------------
    doc.add_heading("2. Formulation, QUBO and Ising mapping", level=1)
    doc.add_paragraph(
        "Decision variables: x = [x_A1, x_A2, x_A3, x_B1, x_B2, x_B3] in {0,1}^6, where x_{i,t} = 1 "
        "if EV i charges in slot t."
    )
    add_table(
        doc,
        ["Component", "Expression"],
        [
            ["Objective 1 (tariff and urgency)", "c^T x,  c = [8, 2, 4, 7, 2, 5]"],
            ["Objective 2 (thermal stress)", "12 x_A1 x_B1 + 10 x_A2 x_B2 + 8 x_A3 x_B3"],
            ["SoC constraint, EV_A", "x_A1 + x_A2 + x_A3 = 2"],
            ["SoC constraint, EV_B", "x_B1 + x_B2 + x_B3 = 2"],
        ],
    )
    doc.add_paragraph(
        f"Penalty formulation (P = {cfg['penalty']:g}): minimise c^T x + thermal + "
        "P[(sum_A - 2)^2 + (sum_B - 2)^2]. Substituting x_k = (1 - Z_k)/2 gives a diagonal 6-qubit "
        f"Ising Hamiltonian with {results['ising']['n_z_terms']} single-qubit and "
        f"{results['ising']['n_zz_terms']} two-qubit Z terms. The XY-mixer variant uses the same "
        "objective without the penalty, as its cost Hamiltonian. Before any quantum run, the code "
        "checks that the Ising diagonal equals the brute-force energy for all 64 basis states."
    )

    # 3. Architecture ------------------------------------------------------------
    doc.add_heading("3. Hybrid quantum-classical architecture", level=1)
    add_figure(doc, DOCS_DIR / "architecture.png", "Figure 1. Architecture: domain, encoding, hybrid loop, validation.", width=6.4)
    doc.add_paragraph(
        "Variant A (penalty QAOA) starts from |+>^6 and uses QAOAAnsatz on the penalty Hamiltonian. "
        "Variant B (XY-mixer QAOA) starts from the feasible basis state |011011>. Each layer applies "
        "the objective phase, then XY rotations on the pairs (0,1), (1,2), (2,0) inside each EV's "
        "register. Those rotations conserve Hamming weight, so the state never leaves the 9-state "
        "feasible subspace. COBYLA optimizes (gamma, beta) against StatevectorEstimator in both "
        "cases, and StatevectorSampler draws the final shots."
    )

    # 4. Results ---------------------------------------------------------------------
    doc.add_heading("4. Results and classical reference", level=1)
    doc.add_paragraph(
        f"Of the 64 basis states, {results['n_feasible']} ({pct(results['n_feasible'] / 64)}) satisfy both SoC "
        f"constraints. The exact optimum is |{opt}>, with tariff 15.0, thermal stress 10.0 and total "
        f"{results['optimal_cost']:.1f}. NumPyMinimumEigensolver returns {results['exact_eigensolver_energy']:.1f}, "
        "which matches the brute force. The greedy tariff-only schedule "
        f"|{greedy}> has tariff {tariff_greedy:.1f} but thermal stress {overload_greedy:.1f}, for a total of "
        f"{results['greedy_cost']:.1f}."
    )
    doc.add_heading("4.1 QAOA performance (sampled)", level=2)
    add_table(
        doc,
        ["Metric", "Penalty QAOA (A)", "XY-mixer QAOA (B)"],
        [
            ["QAOA depth p / restarts / shots", f"{cfg['reps']} / {cfg['restarts']} / {cfg['shots']}", f"{cfg['reps']} / {cfg['restarts']} / {cfg['shots']}"],
            ["Optimized expectation (objective)", f"{pen['optimal_energy'] + results['ising']['offset']:.2f}", f"{xy['optimal_energy'] + results['ising']['objective_offset']:.2f}"],
            [f"P(global optimum |{opt}>)", pct(pen["p_optimum"]), pct(xy["p_optimum"])],
            ["P(feasible schedule)", pct(pen["p_feasible"]), pct(xy["p_feasible"])],
            ["Best sampled bitstring", f"|{pen['best_sampled_label']}> (cost {pen['best_sampled_cost']:.1f})", f"|{xy['best_sampled_label']}> (cost {xy['best_sampled_cost']:.1f})"],
        ],
    )
    doc.add_paragraph(
        "The quantum results are sampled, so the probabilities vary with the seed and the number of "
        "shots. Seed, restarts and shots are fixed in results.json, so the run is reproducible."
    )
    add_figure(doc, CHARTS_DIR / "variant_comparison.png", "Figure 2. Feasibility and optimum probability: penalty vs. XY-mixer.")
    add_figure(doc, CHARTS_DIR / "qaoa_convergence.png", "Figure 3. COBYLA convergence for each multi-start, both variants.")
    add_figure(doc, CHARTS_DIR / "qaoa_distribution.png", "Figure 4. Top-12 sampled bitstrings for each variant.")

    doc.add_heading("4.2 Penalty weight sensitivity (exact)", level=2)
    doc.add_paragraph(
        f"The exact sweep finds that the penalised global minimum is feasible for P >= {results['penalty_threshold']:g} "
        "(grid step 0.1). Below that, the global minimum under-charges at least one EV. This "
        "is classical brute force, not QAOA; QAOA was run only at the default P."
    )
    add_table(
        doc,
        ["P", "Penalised global minimum", "Cost", "Feasible?", "Best feasible cost"],
        [
            [
                f"{r['penalty']:g}",
                f"|{r['global_opt_label']}>",
                f"{r['global_opt_cost']:.1f}",
                "yes" if r["global_opt_feasible"] else "no",
                f"{r['best_feasible_cost']:.1f}",
            ]
            for r in results["penalty_sweep"]
        ],
    )
    add_figure(doc, CHARTS_DIR / "penalty_sensitivity.png", f"Figure 5. Penalty sweep and feasibility threshold P* = {results['penalty_threshold']:g}.")

    doc.add_heading("4.3 Pareto frontier and weight sweep", level=2)
    doc.add_paragraph(
        "Over the feasible schedules, the Pareto frontier in (tariff, thermal) space contains the "
        "non-dominated schedules listed below. The weighted-sum sweep chooses, for each tariff weight "
        "lambda, the schedule minimising lambda * tariff + (1 - lambda) * thermal. It recovers only "
        "points on the convex hull of the frontier."
    )
    add_table(
        doc,
        ["lambda (tariff weight)", "Chosen schedule", "Tariff", "Thermal"],
        [[f"{w['lambda_tariff']:.2f}", f"|{w['label']}>", f"{w['tariff']:.1f}", f"{w['grid']:.1f}"] for w in results["weight_sweep"]],
    )
    add_figure(doc, CHARTS_DIR / "pareto_frontier.png", "Figure 6. Feasible schedules, Pareto frontier, and schedules chosen by the weight sweep.")

    doc.add_heading("4.4 NISQ transpilation overhead", level=2)
    add_table(
        doc,
        ["Metric", "Penalty: all-to-all", "Penalty: linear map", "XY: all-to-all", "XY: linear map"],
        [
            ["CX gates", str(transp["penalty"]["raw_cx"]), str(transp["penalty"]["native_cx"]), str(transp["xy"]["raw_cx"]), str(transp["xy"]["native_cx"])],
            ["Depth", str(transp["penalty"]["raw_depth"]), str(transp["penalty"]["native_depth"]), str(transp["xy"]["raw_depth"]), str(transp["xy"]["native_depth"])],
        ],
    )

    # 5. Limitations ------------------------------------------------------------
    doc.add_heading("5. Honest discussion of limitations and trade-offs", level=1)
    for i, (name, text) in enumerate(LIMITATIONS, 1):
        doc.add_heading(f"5.{i} {name}", level=2)
        doc.add_paragraph(text)

    # 6. Reproducibility -----------------------------------------------------------
    doc.add_heading("6. Reproducibility", level=1)
    versions = ", ".join(f"{k} {v}" for k, v in results["versions"].items())
    doc.add_paragraph(
        "Run `python main.py` and then `python build_report.py`. Seed 42 is fixed, and the tests "
        "in tests/test_model.py cover the classical model and the feasibility of the XY ansatz. "
        f"Package versions for this run: {versions}. Wall-clock timings are in results.json."
    )

    doc.save(OUTPUT)
    print(f"Report written to {OUTPUT}")


if __name__ == "__main__":
    main()
