import argparse
import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_bytes().decode("utf-8", errors="replace"))

def load_delphi_logs(paths: list[str]) -> list[dict]:
    logs = [load_json(p) for p in paths]
    console.print(f"[cyan]{len(logs)} log(s) Delphi carregado(s).[/cyan]")
    return logs



def get_trajectory(log: dict) -> list[float]:
    return log["trajectory"]["cosine_divergence_mean"]

def get_revision_rates(log: dict) -> list[float | None]:
    return log["trajectory"]["revision_rate"]

def baseline_divergence(baseline_log: dict) -> float:
    """Converte similaridade cosseno do baseline para divergência."""
    sim = baseline_log["metrics"]["cosine_similarity"]["mean"]
    return 1.0 - sim

def aggregate_trajectories(logs: list[dict]) -> dict:
    """Agrega múltiplos runs: média e DP por rodada."""
    max_rounds = max(len(get_trajectory(l)) for l in logs)
    matrix = np.full((len(logs), max_rounds), np.nan)
    for i, log in enumerate(logs):
        traj = get_trajectory(log)
        matrix[i, :len(traj)] = traj
    return {
        "mean":  np.nanmean(matrix, axis=0).tolist(),
        "std":   np.nanstd(matrix, axis=0).tolist(),
        "n_runs": len(logs),
        "max_rounds": max_rounds,
    }

def detect_sycophancy(log: dict) -> list[dict]:
    """
    Identifica casos suspeitos de sycophancy:
    posição marcada como 'mudou' mas justificativa vaga ou ausente.
    """
    suspects = []
    for r in log["rounds"]:
        if not r.get("revisions"):
            continue
        for persona, rev in r["revisions"].items():
            if rev["position_changed"]:
                just = rev.get("change_justification") or ""
                # heurística: justificativa curta (< 30 palavras) é suspeita
                if len(just.split()) < 30:
                    suspects.append({
                        "round": r["round"],
                        "persona": persona,
                        "justification": just,
                    })
    return suspects



def print_comparison_table(baseline_div: float, agg: dict):
    t = Table(
        title="Delphi × Baseline — Divergência Cosseno por Rodada",
        box=box.SIMPLE_HEAVY,
    )
    t.add_column("Condição", style="cyan", min_width=32)
    t.add_column("d_cos (média)", justify="right")
    t.add_column("DP", justify="right")
    t.add_column("Δ vs. Baseline", justify="right")

    t.add_row(
        "Baseline (consulta paralela)",
        f"{baseline_div:.4f}",
        "—",
        "—",
        style="dim",
    )
    for i, (mean, std) in enumerate(zip(agg["mean"], agg["std"])):
        label = f"Delphi — Rodada {i}"
        delta = mean - baseline_div
        delta_str = f"{delta:+.4f}"
        style = "green" if delta < 0 else "red" if delta > 0 else ""
        t.add_row(label, f"{mean:.4f}", f"±{std:.4f}", delta_str, style=style)

    console.print(t)


def print_trajectory_chart(agg: dict, baseline_div: float):
    console.print("\n[bold]Trajetória de Divergência (escala: 1 █ = 0.005):[/bold]")
    scale = 200
    baseline_bar = "░" * int(baseline_div * scale)
    console.print(f"  Baseline  : {baseline_div:.4f}  {baseline_bar}")
    for i, mean in enumerate(agg["mean"]):
        bar = "█" * max(1, int(mean * scale))
        console.print(f"  Rodada {i}  : {mean:.4f}  {bar}")


def print_revision_analysis(logs: list[dict]):
    console.print("\n[bold]Taxa de Revisão de Posição por Rodada:[/bold]")
    # usa o primeiro log como referência para estrutura
    for log in logs[:1]:
        for r in log["rounds"]:
            revs = r.get("revisions")
            if not revs:
                continue
            total = len(revs)
            changed = sum(1 for v in revs.values() if v["position_changed"])
            bar = "■" * changed + "□" * (total - changed)
            console.print(f"  Rodada {r['round']}: {changed}/{total} [{bar}]")

            for persona, rev in revs.items():
                if rev["position_changed"]:
                    just = rev.get("change_justification", "—") or "—"
                    console.print(
                        f"    [yellow]{persona}[/yellow] mudou: "
                        f"{just[:120]}{'...' if len(just) > 120 else ''}"
                    )


def print_synthesis_summary(log: dict):
    console.print("\n[bold]Sínteses Intermediárias por Rodada:[/bold]")
    for r in log["rounds"]:
        if not r.get("synthesis"):
            continue
        s = r["synthesis"]
        console.rule(f"[yellow]Rodada {r['round']}[/yellow]")
        console.print(f"  [green]Convergências:[/green]")
        for c in s["convergencias"]:
            console.print(f"    • {c}")
        console.print(f"  [red]Divergências:[/red]")
        for d in s["divergencias"]:
            console.print(f"    • {d}")
        if s["dimensoes_nao_abordadas"]:
            console.print(f"  [blue]Não abordadas:[/blue]")
            for d in s["dimensoes_nao_abordadas"]:
                console.print(f"    • {d}")


def print_sycophancy_report(logs: list[dict]):
    console.rule("[bold]Análise de Sycophancy[/bold]")
    all_suspects = []
    for log in logs:
        all_suspects.extend(detect_sycophancy(log))

    if not all_suspects:
        console.print("[green]Nenhum caso suspeito de sycophancy detectado.[/green]")
        return

    console.print(
        f"[yellow]{len(all_suspects)} caso(s) suspeito(s) "
        f"(revisão com justificativa < 30 palavras):[/yellow]"
    )
    for s in all_suspects:
        console.print(
            f"  Rodada {s['round']} / {s['persona']}: "
            f"\"{s['justification'][:100]}\""
        )


def print_stop_reasons(logs: list[dict]):
    console.print("\n[bold]Motivos de Parada:[/bold]")
    from collections import Counter
    reasons = Counter(log.get("stop_reason", "desconhecido") for log in logs)
    for reason, count in reasons.items():
        console.print(f"  {reason}: {count} run(s)")


def compare(delphi_paths: list[str], baseline_path: str):
    delphi_logs  = load_delphi_logs(delphi_paths)
    baseline_log = load_json(baseline_path)

    baseline_div = baseline_divergence(baseline_log)
    agg = aggregate_trajectories(delphi_logs)

    console.rule("[bold blue]COMPARAÇÃO DELPHI × BASELINE")
    console.print(f"  Baseline: {baseline_path}")
    console.print(f"  Delphi:   {len(delphi_logs)} run(s)")
    console.print(f"  Questão:  {delphi_logs[0]['question'][:80]}...")

    print_comparison_table(baseline_div, agg)
    print_trajectory_chart(agg, baseline_div)
    print_revision_analysis(delphi_logs)
    print_synthesis_summary(delphi_logs[0])  # síntese detalhada do primeiro run
    print_sycophancy_report(delphi_logs)
    print_stop_reasons(delphi_logs)

    # Exportar resumo JSON
    summary = {
        "baseline_divergence": baseline_div,
        "delphi_trajectory": agg,
        "delta_r0_vs_baseline": agg["mean"][0] - baseline_div,
        "delta_final_vs_baseline": agg["mean"][-1] - baseline_div,
        "delta_final_vs_r0": agg["mean"][-1] - agg["mean"][0],
        "n_delphi_runs": len(delphi_logs),
    }
    out_path = Path(delphi_paths[0]).parent / "comparison_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    console.print(f"\n[bold green]Resumo salvo em: {out_path}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara Delphi vs. Baseline")
    parser.add_argument(
        "--delphi",
        nargs="+",
        required=True,
        help="Um ou mais arquivos JSON do ciclo Delphi",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Arquivo JSON do baseline (run_baseline.py)",
    )
    args = parser.parse_args()
    compare(args.delphi, args.baseline)
