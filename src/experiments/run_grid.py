import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

from experiments.shared import (
    MODEL,
    EMBEDDING_MODEL,
    PERSONA_NAMES,
    compute_cosine_sim,
    load_client,
    query_persona,
    save_json,
    ts_now,
)

TEMPERATURES = [0.0, 0.4, 0.8, 1.0, 1.2]

QUESTIONS: dict[str, str] = {
    "obstaculo_adocao": (
        "Qual é o maior obstáculo para adoção em larga escala de IA nas empresas ao redor do mundo?"
    ),
    "concentracao_renda": (
        "Como a IA afetará a concentração de renda globalmente nos próximos 20 anos?"
    ),
    "relacoes_poder": (
        "Como a IA mudará as relações de poder entre empregadores e trabalhadores nas próximas décadas?"
    ),
    "beneficiados_ia": (
        "Quais perfis de trabalhadores serão mais beneficiados pela IA globalmente e por quê?"
    ),
    "adocao_desigual": (
        "Por que a adoção de IA é tão desigual entre países desenvolvidos e em desenvolvimento?"
    ),
}

N_RUNS = 5
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "exp4"
)
TOTAL_RUNS = len(TEMPERATURES) * len(QUESTIONS) * N_RUNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 4 — Grid Search temperatura × questão")
    args = parser.parse_args()

    client = load_client()
    console = Console()
    console.rule("[bold blue]Exp 4 — Grid Search: Temperatura × Questão[/bold blue]")
    console.print(
        f"Grid: {len(TEMPERATURES)} temperaturas × {len(QUESTIONS)} questões × {N_RUNS} runs "
        f"= [bold]{TOTAL_RUNS}[/bold] runs totais\n"
        f"Métrica: cosseno (rode src/experiments/enrich_metrics.py para BERTScore + sentence-cosine)\n"
    )

    # grid_results[temp][q_key] = lista de divergências por cosseno
    grid_acc: dict[float, dict[str, list[float]]] = {
        t: {q: [] for q in QUESTIONS} for t in TEMPERATURES
    }
    run_counter = 0

    for temp in TEMPERATURES:
        for q_key, question in QUESTIONS.items():
            for run_idx in range(N_RUNS):
                run_counter += 1
                console.print(
                    f"[{run_counter:>3}/{TOTAL_RUNS}] temp={temp} | {q_key} | run {run_idx}…",
                    end=" ",
                )

                responses = [
                    query_persona(client, p, question, temperature=temp)
                    for p in PERSONA_NAMES
                ]
                labels = [r["persona"] for r in responses]
                texts = [r["response"] for r in responses]

                cosine_scores = compute_cosine_sim(texts, labels, client)
                cosine_div = 1.0 - cosine_scores["mean"]
                grid_acc[temp][q_key].append(cosine_div)

                run_data = {
                    "experiment": "exp4_grid",
                    "temperature": temp,
                    "question_key": q_key,
                    "question": question,
                    "run_index": run_idx,
                    "model": MODEL,
                    "embedding_model": EMBEDDING_MODEL,
                    "responses": responses,
                    "metrics": {
                        "cosine_similarity": cosine_scores,
                        "cosine_divergence": cosine_div,
                    },
                }

                fname = f"temp{str(temp).replace('.', '')}_{q_key}_run{run_idx}_{ts_now()}.json"
                save_json(run_data, os.path.join(OUTPUT_DIR, fname))
                console.print(f"cosine_div={cosine_div:.4f}")

    # Agregar por célula
    grid_summary: list[dict] = []
    for temp in TEMPERATURES:
        for q_key in QUESTIONS:
            vals = grid_acc[temp][q_key]
            grid_summary.append({
                "temperature": temp,
                "question_key": q_key,
                "cosine_divergence_mean": float(np.mean(vals)),
                "cosine_divergence_std": float(np.std(vals, ddof=0)),
                "cosine_divergence_runs": vals,
            })

    ranking = sorted(grid_summary, key=lambda x: x["cosine_divergence_mean"], reverse=True)
    best = ranking[0]

    summary = {
        "experiment": "exp4_grid",
        "temperatures": TEMPERATURES,
        "questions": list(QUESTIONS.keys()),
        "n_runs": N_RUNS,
        "model": MODEL,
        "grid": grid_summary,
    }
    save_json(summary, os.path.join(OUTPUT_DIR, "summary.json"))
    save_json(ranking, os.path.join(OUTPUT_DIR, "ranking.json"))

    # Tabela ranking top-10
    t = Table(title="Ranking — Combinações por Divergência de Cosseno (top 10)", box=box.ROUNDED)
    t.add_column("Pos", justify="right", width=4)
    t.add_column("Temp", justify="right", width=6)
    t.add_column("Questão", width=30)
    t.add_column("Divergência média", justify="right", width=18)
    t.add_column("± std", justify="right", width=8)

    for pos, row in enumerate(ranking[:10], 1):
        t.add_row(
            str(pos),
            str(row["temperature"]),
            row["question_key"][:30],
            f"{row['cosine_divergence_mean']:.4f}",
            f"{row['cosine_divergence_std']:.4f}",
        )

    console.print(t)
    console.print(
        f"\n[bold green]Melhor combinação (cosseno):[/bold green] "
        f"temperatura=[bold]{best['temperature']}[/bold]  "
        f"questão=[bold]{best['question_key']}[/bold]  "
        f"divergência={best['cosine_divergence_mean']:.4f} ±{best['cosine_divergence_std']:.4f}"
    )
    console.print(
        "\n[dim]Para ranking com BERTScore + sentence-cosine:[/dim]"
        "\n[dim]python src/experiments/enrich_metrics.py --input-dir src/data/exp4[/dim]"
    )
    console.print(f"[green]Arquivos em {OUTPUT_DIR}[/green]")


if __name__ == "__main__":
    main()
