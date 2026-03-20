"""Generate the LaTeX comparison report and compile to PDF."""

import subprocess
from pathlib import Path

import numpy as np


PDFLATEX = "/usr/local/texlive/2026/bin/universal-darwin/pdflatex"


def _fmt(x, decimals=4):
    """Format a float for LaTeX tables."""
    if x is None:
        return "---"
    return f"{x:.{decimals}f}"


def _figure_block(filename: str, caption: str, label: str, width: str = "0.85") -> str:
    """Generate a LaTeX figure block."""
    return rf"""
\begin{{figure}}[htbp]
\centering
\includegraphics[width={width}\textwidth]{{{filename}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""


def generate_report(
    all_results: dict | None,
    excited_results: dict | None,
    noise_results: dict | None,
    output_dir: Path,
):
    """Generate comparison_report.tex and compile to PDF."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build tables from results
    noiseless_table = _build_noiseless_table(all_results) if all_results else "No noiseless results available."
    excited_table = _build_excited_table(excited_results) if excited_results else "No excited-state results available."
    noise_table = _build_noise_table(noise_results) if noise_results else "No noise results available."

    tex = rf"""\documentclass[11pt,a4paper]{{article}}

\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{physics}}
\usepackage{{hyperref}}
\usepackage{{booktabs}}
\usepackage{{geometry}}
\usepackage{{enumitem}}
\usepackage{{xcolor}}
\usepackage{{graphicx}}
\usepackage{{subcaption}}

\geometry{{margin=1in}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue, citecolor=blue}}

\title{{Comparison of VQE, VQITE, and TTITE\\
  for Charmonium Spectroscopy\\[0.5em]
  \large Apples-to-Apples Benchmark on the Woloshyn Hamiltonian}}

\author{{quarkonium-simulations project}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
We compare three quantum-computing methods---the Variational Quantum
Eigensolver (VQE), Variational Quantum Imaginary-Time Evolution (VQITE),
and Trotter--Taylor Imaginary-Time Evolution (TTITE)---on the same
2-qubit charmonium Hamiltonian with spin-dependent interactions.
All three methods use the same direct qubit encoding and the same
physical parameters, enabling a controlled comparison of accuracy,
convergence speed, and noise sensitivity.
\end{{abstract}}

\tableofcontents
\newpage

%=============================================================================
\section{{Setup}}
%=============================================================================

\subsection{{The Hamiltonian}}

We use the Woloshyn charmonium Hamiltonian (arXiv:2301.10828v2), which
describes a charm--anticharm system with the Cornell potential plus a
spin-dependent contact term:
\begin{{equation}}
  V(r) = -\frac{{a}}{{r}} + b\,r + V_s(r)\,\vec{{S}}_c \cdot \vec{{S}}_{{\bar{{c}}}}
\end{{equation}}

The Hamiltonian is expanded in a harmonic oscillator basis truncated
to $N=4$ radial states, yielding a $4 \times 4$ matrix encoded on
\textbf{{2 qubits}} via direct state mapping:
$\ket{{00}} \leftrightarrow n\!=\!0$,
$\ket{{01}} \leftrightarrow n\!=\!1$,
$\ket{{10}} \leftrightarrow n\!=\!2$,
$\ket{{11}} \leftrightarrow n\!=\!3$.

Three spin channels are studied:
${{}}^1S_0$ ($\eta_c$, spin singlet),
${{}}^3S_1$ ($J/\psi$, spin triplet), and
${{}}^1P_1$ ($h_c$, $P$-wave singlet).

The Pauli decomposition of each channel has 10 terms:
\begin{{equation}}
  H = c_0\,II + c_1\,IZ + c_2\,ZI + c_3\,ZZ + c_4\,IX + c_5\,XI
    + c_6\,ZX + c_7\,XZ + c_8\,XX + c_9\,YY
\end{{equation}}

\subsection{{The Three Methods}}

\paragraph{{VQE (Variational Quantum Eigensolver)}}
Minimizes the energy expectation value over a parameterized ansatz:
\begin{{equation}}
  E(\vec{{\theta}}) = \mel{{\psi(\vec{{\theta}})}}{{H}}{{\psi(\vec{{\theta}})}} \geq E_0
\end{{equation}}
using the COBYLA classical optimizer. The ansatz is a 3-parameter
$R_Y$--CNOT circuit on 2 qubits.

\paragraph{{VQITE (Variational Quantum Imaginary-Time Evolution)}}
Evolves variational parameters along the imaginary-time gradient:
\begin{{equation}}
  \sum_j A_{{ij}}\,\dot{{\theta}}_j = C_i, \qquad
  C_i = -\frac{{\partial E}}{{\partial\theta_i}}, \quad
  A_{{ij}} = \text{{Re}}\!\left[\frac{{\partial\bra{{\psi}}}}{{\partial\theta_i}}
    \frac{{\partial\ket{{\psi}}}}{{\partial\theta_j}}\right]
\end{{equation}}
where $A$ is the quantum metric tensor. Uses the same 3-parameter ansatz
as VQE, with parameters updated via forward Euler.

\paragraph{{TTITE (Trotter--Taylor Imaginary-Time Evolution)}}
Implements imaginary-time evolution directly (no ansatz) via:
\begin{{equation}}
  e^{{-H\tau}} \approx \prod_{{i=1}}^m \left(\alpha_i\,I + \beta_i\,h_i\right)^L
\end{{equation}}
where $\alpha_i$ and $\beta_i$ are Taylor expansion coefficients and each
factor is applied via an ancilla-qubit Linear Combination of Unitaries (LCU)
circuit with postselection.

\subsection{{Shared Ansatz}}

All variational methods (VQE, VQITE) use the same hardware-efficient ansatz:
\begin{{equation}}
  \begin{{aligned}}
  &q_0: \quad R_Y(\theta_0) - \bullet - \\
  &q_1: \quad R_Y(\theta_1) - \oplus - R_Y(\theta_2)
  \end{{aligned}}
\end{{equation}}
With 3 parameters, this circuit spans all real superpositions of the 4 basis states.

%=============================================================================
\section{{Noiseless Results}}
%=============================================================================

\subsection{{Ground-State Energy}}

{noiseless_table}

{_figure_block("energy_summary.png",
               "Noiseless ground-state energy comparison across all channels and methods.",
               "fig:energy_summary")}

{_figure_block("convergence_comparison.png",
               "Convergence of each method. VQE: energy vs optimizer iteration; "
               "VQITE: energy vs imaginary-time step; TTITE: energy vs imaginary time $\\\\tau$.",
               "fig:convergence")}

{_figure_block("circuit_cost.png",
               "Computation cost: wall-clock time (left) and number of iterations/steps (right).",
               "fig:cost")}

%=============================================================================
\section{{Excited States}}
%=============================================================================

The penalty method is used for both VQE and VQITE:
\begin{{equation}}
  E_{{\text{{eff}}}}(\vec{{\theta}}) = \mel{{\psi(\vec{{\theta}})}}{{H}}{{\psi(\vec{{\theta}})}}
  + \alpha \sum_{{j<k}} \bigl|\braket{{\phi_j|\psi(\vec{{\theta}})}}\bigr|^2
\end{{equation}}
with $\alpha = 10$. TTITE does not support excited states.

{excited_table}

{_figure_block("excited_states.png",
               "Energy levels from VQE and VQITE (with penalty method) compared to exact diagonalization. "
               "TTITE finds only the ground state.",
               "fig:excited")}

%=============================================================================
\section{{Noise Analysis}}
%=============================================================================

All methods run on \texttt{{AerSimulator}} with a depolarizing noise model
at rates $p \in \{{0, 0.01, 0.05\}}$, with 3 repeats per configuration:
\begin{{itemize}}[nosep]
  \item \textbf{{VQE}}: Every energy evaluation runs shot-based Pauli
    measurement circuits on a noisy \texttt{{AerSimulator}}. The COBYLA
    optimizer sees noisy cost function values.
  \item \textbf{{VQITE}}: Energy gradient via parameter-shift rule on
    noisy circuits. Metric tensor via overlap circuits on the same noisy
    backend. Both gradient and metric are corrupted by gate noise.
  \item \textbf{{TTITE}}: A single continuous circuit chains all
    $\sim\!2250$ LCU steps with ancilla \texttt{{reset}} between steps.
    Gate noise accumulates across the entire circuit. The \texttt{{reset}}
    instruction does \emph{{not}} postselect---if the ancilla measures
    $\ket{{1}}$ (failed LCU), the corrupted work state carries forward,
    modeling realistic hardware behavior.
\end{{itemize}}

{noise_table}

{_figure_block("noise_sweep.png",
               "Energy error vs depolarizing rate for all three methods. "
               "Error bars show standard deviation over repeated runs.",
               "fig:noise")}

{_figure_block("fidelity_vs_noise.png",
               "Ground-state fidelity vs depolarizing rate.",
               "fig:fidelity")}

%=============================================================================
\section{{Discussion}}
%=============================================================================

\subsection{{Accuracy}}

In the noiseless limit (shot noise only, no gate errors), all three methods
converge to the exact ground-state energy within shot-noise precision
($\sim\!0.01$--$0.03$~fm$^{{-1}}$). The key differences emerge under gate noise:
\begin{{itemize}}[nosep]
  \item \textbf{{VQE is most noise-resilient.}} Its shallow ansatz circuit
    (depth 3, one CNOT) accumulates minimal gate error per evaluation.
    COBYLA partially adapts to the noisy landscape. At 1\% depolarizing
    noise, energy errors remain $\sim\!0.1$--$0.3$~fm$^{{-1}}$.
  \item \textbf{{VQITE degrades significantly under noise.}} The
    parameter-shift gradient and overlap-circuit metric tensor are both
    corrupted by shot and gate noise, causing erratic parameter updates.
    At 1\% noise, errors reach $\sim\!0.5$--$2$~fm$^{{-1}}$ with high
    variance between runs.
  \item \textbf{{TTITE is catastrophically destroyed by noise.}} The
    continuous circuit of $\sim\!2250$ LCU steps accumulates gate errors
    at every operation. Additionally, failed postselections (ancilla
    measuring $\ket{{1}}$) mix the ``wrong'' LCU branch into the work
    state. At 1\% noise, the state collapses to the maximally mixed
    state (fidelity $\to 1/d = 0.25$).
\end{{itemize}}

\subsection{{Convergence Speed}}

VQE convergence depends on the optimizer and landscape geometry;
COBYLA typically requires $\sim\!50$--$100$ function evaluations but
can get trapped in local minima with shot noise. VQITE converges
monotonically via natural gradient in $\sim\!30$--$40$ steps, though
each step requires $\sim\!30$ circuit evaluations (gradient + metric tensor).
TTITE converges deterministically in imaginary time ($\sim\!250$ Trotter
segments) with the fastest wall-clock time ($<\!1$~s noiseless).

\subsection{{Practical Considerations}}

\begin{{center}}
\begin{{tabular}}{{@{{}}l c c c@{{}}}}
\toprule
Criterion & VQE & VQITE & TTITE \\
\midrule
Ansatz required & Yes & Yes & No \\
Ancilla qubits & 0 & 0 & 1 \\
Circuit depth & Shallow (3) & Shallow (3) & Deep ($\sim\!2250$) \\
Noise resilience & Best & Moderate & Worst \\
Excited states & Yes (penalty) & Yes (penalty) & No \\
Convergence guarantee & No & Conditional & Yes (noiseless) \\
\bottomrule
\end{{tabular}}
\end{{center}}

\subsection{{Conclusion}}

For near-term noisy quantum hardware, \textbf{{VQE is the clear winner}}:
its shallow circuits minimize noise exposure, and the classical optimizer
provides robustness to shot noise. VQITE offers stronger theoretical
convergence guarantees but is undermined by the circuit cost of computing
the quantum metric tensor under noise. TTITE is exact in principle but
fundamentally impractical on noisy hardware---its deep circuit and
postselection requirements make it a fault-tolerant-era algorithm.

\end{{document}}
"""

    tex_path = output_dir / "comparison_report.tex"
    with open(tex_path, "w") as f:
        f.write(tex)

    # Compile PDF (2 passes for references)
    for pass_num in range(2):
        result = subprocess.run(
            [PDFLATEX, "-interaction=nonstopmode", "comparison_report.tex"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 and pass_num == 1:
            print(f"  pdflatex warning (pass {pass_num + 1}): see {output_dir}/comparison_report.log")

    pdf_path = output_dir / "comparison_report.pdf"
    if pdf_path.exists():
        print(f"  Report: {pdf_path}")
    else:
        print(f"  PDF compilation may have failed. Check {output_dir}/comparison_report.log")


def _build_noiseless_table(all_results: dict) -> str:
    """Build LaTeX table of noiseless ground-state results."""
    channels = list(all_results.keys())
    methods = list(next(iter(all_results.values())).keys())

    rows = []
    for ch in channels:
        for method_name in methods:
            mr = all_results[ch][method_name]
            rows.append(
                f"    {ch} & {method_name} & {_fmt(mr.energy)} & "
                f"{_fmt(mr.exact_energy)} & {_fmt(mr.error, 6)} & {_fmt(mr.fidelity, 6)} & "
                f"{mr.wall_time:.2f} \\\\"
            )
        rows.append("    \\midrule")

    # Remove last midrule
    if rows and rows[-1].strip() == "\\midrule":
        rows.pop()

    table_body = "\n".join(rows)

    return rf"""
\begin{{center}}
\small
\begin{{tabular}}{{@{{}}l l r r r r r@{{}}}}
\toprule
Channel & Method & $E_0$ [fm$^{{-1}}$] & Exact & Error & Fidelity & Time [s] \\
\midrule
{table_body}
\bottomrule
\end{{tabular}}
\end{{center}}
"""


def _build_excited_table(excited_results: dict) -> str:
    """Build LaTeX table of excited-state results."""
    if not excited_results:
        return ""

    rows = []
    for ch in excited_results:
        for method_name, mr_list in excited_results[ch].items():
            for mr in mr_list:
                k = getattr(mr, "metadata", {}).get("state_index", None) or getattr(mr, "state_index", "?")
                rows.append(
                    f"    {ch} & {method_name} & {k} & "
                    f"{_fmt(mr.energy)} & {_fmt(mr.exact_energy)} & {_fmt(mr.error, 4)} \\\\"
                )
        rows.append("    \\midrule")

    if rows and rows[-1].strip() == "\\midrule":
        rows.pop()

    table_body = "\n".join(rows)

    return rf"""
\begin{{center}}
\small
\begin{{tabular}}{{@{{}}l l c r r r@{{}}}}
\toprule
Channel & Method & State & Energy & Exact & Error \\
\midrule
{table_body}
\bottomrule
\end{{tabular}}
\end{{center}}
"""


def _build_noise_table(noise_results: dict) -> str:
    """Build LaTeX table of noise sweep results (mean +/- std)."""
    if not noise_results:
        return ""

    # Use the first channel as representative
    ch = next(iter(noise_results))
    sweep = noise_results[ch]

    rows = []
    for rate in sorted(sweep.noise_levels):
        for method_name in ["VQE", "VQITE", "TTITE"]:
            e_list = sweep.energies.get(method_name, {}).get(rate, [])
            if e_list:
                mean_e = np.mean(e_list)
                std_e = np.std(e_list) if len(e_list) > 1 else 0.0
                error = mean_e - sweep.exact_energy
                rows.append(
                    f"    {rate:.3f} & {method_name} & "
                    f"${_fmt(mean_e)} \\pm {_fmt(std_e)}$ & {_fmt(error, 4)} \\\\"
                )
        rows.append("    \\midrule")

    if rows and rows[-1].strip() == "\\midrule":
        rows.pop()

    table_body = "\n".join(rows)

    return rf"""
Representative results for channel {ch}:
\begin{{center}}
\small
\begin{{tabular}}{{@{{}}l l r r@{{}}}}
\toprule
Noise rate & Method & Energy [fm$^{{-1}}$] & Error \\
\midrule
{table_body}
\bottomrule
\end{{tabular}}
\end{{center}}
"""
