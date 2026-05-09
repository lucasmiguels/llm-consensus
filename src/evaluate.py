import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

def load_logs(input_path: str) -> list[dict]:
    if os.path.isdir(input_path):
        paths = sorted(glob.glob(os.path.join(input_path, "*.json")))
    else:
        paths = sorted(glob.glob(input_path))
    if not paths:
        print(f"[ERRO] Nenhum JSON encontrado em: {input_path}", file=sys.stderr)
        sys.exit(1)
    logs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
            data["_source_file"] = os.path.basename(p)
            logs.append(data)
    return logs

def _sim(log: dict, round_idx: Optional[int] = None) -> dict:
    """Retorna métricas de similaridade semântica de um log."""
    if log.get("experiment_type") == "baseline" or round_idx is None:
        return log.get("metrics", {}).get("semantic_similarity", {})
    rounds = log.get("rounds", [])
    if round_idx < len(rounds):
        return rounds[round_idx].get("metrics", {}).get("semantic_similarity", {})
    return {}


def _numeric(log: dict, round_idx: Optional[int] = None) -> dict:
    if log.get("experiment_type") == "baseline" or round_idx is None:
        return log.get("metrics", {}).get("numeric_estimates", {})
    rounds = log.get("rounds", [])
    if round_idx < len(rounds):
        return rounds[round_idx].get("metrics", {}).get("numeric_estimates", {})
    return {}


def summarize_baseline(log: dict) -> dict:
    sim = _sim(log)
    num = _numeric(log)
    return {
        "arquivo": log["_source_file"],
        "tipo": "baseline",
        "questao": log.get("question", "")[:55] + "…",
        "rodada": 0,
        "agentes": len(log.get("responses", [])),
        "sim_media": sim.get("mean"),
        "dp_estimativas": num.get("std_dev"),
        "media_estimativas": num.get("mean"),
        "delta_sim": None,
        "delta_dp": None,
        "taxa_revisao": None,
    }


def summarize_delphi(log: dict) -> list[dict]:
    rows = []
    rounds = log.get("rounds", [])
    sim_r0 = _sim(log, 0).get("mean") if rounds else None
    dp_r0 = _numeric(log, 0).get("std_dev") if rounds else None

    for t, rd in enumerate(rounds):
        sim = rd.get("metrics", {}).get("semantic_similarity", {})
        num = rd.get("metrics", {}).get("numeric_estimates", {})
        sim_t = sim.get("mean")
        dp_t = num.get("std_dev")

        rows.append({
            "arquivo": log["_source_file"],
            "tipo": "delphi",
            "questao": log.get("question", "")[:55] + "…",
            "rodada": t + 1,
            "agentes": 3,
            "sim_media": sim_t,
            "dp_estimativas": dp_t,
            "media_estimativas": num.get("mean"),
            "delta_sim": (sim_t - sim_r0) if (sim_t is not None and sim_r0 is not None) else None,
            "delta_dp": (dp_t - dp_r0) if (dp_t is not None and dp_r0 is not None) else None,
            "taxa_revisao": None,  # avaliação manual
        })
    return rows


def _fmt(v, fmt=".4f", fallback="—") -> str:
    if v is None:
        return fallback
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return fallback


def print_baseline_table(rows: list[dict], console: Console) -> None:
    t = Table(title="Resultados — Baseline", box=box.ROUNDED, show_lines=True)
    t.add_column("Arquivo", style="cyan", width=38)
    t.add_column("Questão", width=30)
    t.add_column("Sim. Média", justify="right")
    t.add_column("DP Numér.", justify="right")
    t.add_column("Média Numér.", justify="right")
    for r in rows:
        t.add_row(
            r["arquivo"],
            r["questao"],
            _fmt(r["sim_media"]),
            _fmt(r["dp_estimativas"], ".2f"),
            _fmt(r["media_estimativas"], ".1f"),
        )
    console.print(t)


def print_delphi_table(rows: list[dict], console: Console) -> None:
    t = Table(title="Resultados — Delphi (por rodada)", box=box.ROUNDED, show_lines=True)
    t.add_column("Arquivo", style="cyan", width=30)
    t.add_column("Rodada", justify="right")
    t.add_column("Sim. Média", justify="right")
    t.add_column("ΔSim", justify="right")
    t.add_column("DP Numér.", justify="right")
    t.add_column("ΔDP", justify="right")
    t.add_column("Taxa Revisão", justify="right")
    for r in rows:
        t.add_row(
            r["arquivo"],
            str(r["rodada"]),
            _fmt(r["sim_media"]),
            _fmt(r["delta_sim"], "+.4f"),
            _fmt(r["dp_estimativas"], ".2f"),
            _fmt(r["delta_dp"], "+.2f"),
            _fmt(r["taxa_revisao"], ".0%"),
        )
    console.print(t)


def print_comparison_table(baseline_rows: list[dict], delphi_rows: list[dict], console: Console) -> None:
    """Tabela de comparação Delphi final vs Baseline por questão."""
    if not baseline_rows or not delphi_rows:
        return

    # Agrupa Delphi: última rodada de cada arquivo
    from itertools import groupby
    delphi_final: dict[str, dict] = {}
    for row in delphi_rows:
        key = row["arquivo"]
        if key not in delphi_final or row["rodada"] > delphi_final[key]["rodada"]:
            delphi_final[key] = row

    # Baseline por questão (média se múltiplas replicações)
    b_sim = [r["sim_media"] for r in baseline_rows if r["sim_media"] is not None]
    b_dp  = [r["dp_estimativas"] for r in baseline_rows if r["dp_estimativas"] is not None]
    d_sim = [r["sim_media"] for r in delphi_final.values() if r["sim_media"] is not None]
    d_dp  = [r["dp_estimativas"] for r in delphi_final.values() if r["dp_estimativas"] is not None]

    t = Table(title="Comparação Delphi Final × Baseline", box=box.ROUNDED)
    t.add_column("Condição", style="bold")
    t.add_column("Sim. Semântica Média", justify="right")
    t.add_column("DP Estimativas (p.p.)", justify="right")
    t.add_column("n", justify="right")

    t.add_row(
        "Baseline",
        _fmt(np.mean(b_sim) if b_sim else None),
        _fmt(np.mean(b_dp) if b_dp else None, ".2f"),
        str(len(baseline_rows)),
    )
    t.add_row(
        "Delphi (rodada final)",
        _fmt(np.mean(d_sim) if d_sim else None),
        _fmt(np.mean(d_dp) if d_dp else None, ".2f"),
        str(len(delphi_final)),
    )

    console.print(t)


def save_csv(rows: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avaliação formal LLM-Consensus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Diretório com JSONs ou glob (ex: src/data/*.json).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Caminho para CSV de saída (opcional).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()
    console.rule("[bold blue]LLM-Consensus — Avaliação Formal[/bold blue]")

    logs = load_logs(args.input)
    console.print(f"Logs carregados: [bold]{len(logs)}[/bold] arquivo(s)\n")

    baseline_rows: list[dict] = []
    delphi_rows: list[dict] = []

    for log in logs:
        exp_type = log.get("experiment_type", "unknown")
        if exp_type == "baseline":
            baseline_rows.append(summarize_baseline(log))
        elif exp_type == "delphi":
            delphi_rows.extend(summarize_delphi(log))
        else:
            console.print(f"[yellow]Tipo desconhecido '{exp_type}' em {log['_source_file']} — ignorado.[/yellow]")

    if baseline_rows:
        print_baseline_table(baseline_rows, console)

    if delphi_rows:
        print_delphi_table(delphi_rows, console)
        print_comparison_table(baseline_rows, delphi_rows, console)
    elif baseline_rows:
        console.print("\n[dim]Nenhum log Delphi encontrado. Tabela de comparação disponível após execução de run_delphi.py.[/dim]")

    all_rows = baseline_rows + delphi_rows
    if args.output and all_rows:
        save_csv(all_rows, args.output)
        console.print(f"\n[bold]CSV salvo em:[/bold] {args.output}")
    elif args.output:
        console.print("[yellow]Nenhum dado para salvar.[/yellow]")


if __name__ == "__main__":
    main()
