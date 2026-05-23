import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

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

QUESTION = (
    "Como a inteligência artificial transformará a educação e o mercado de trabalho "
    "brasileiro nos próximos 10 anos? Considere tanto oportunidades quanto riscos."
)
QUESTION_KEY = "qualitativa_ia_educacao_trabalho"

TEMPERATURES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
N_RUNS = 5

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "exp1"
)


def main() -> None:
    client = load_client()
    console = Console()
    console.rule("[bold blue]Exp 1 — Temperatura[/bold blue]")
    console.print(f"Questão: {QUESTION}\n")

    summary_rows = []

    for temp in TEMPERATURES:
        console.print(f"[bold cyan]temperatura={temp}[/bold cyan]")
        for run_idx in range(N_RUNS):
            responses = [
                query_persona(client, p, QUESTION, temperature=temp)
                for p in PERSONA_NAMES
            ]
            labels = [r["persona"] for r in responses]
            texts = [r["response"] for r in responses]
            cosine_scores = compute_cosine_sim(texts, labels, client)

            run_data = {
                "experiment": "exp1_temperature",
                "question_key": QUESTION_KEY,
                "question": QUESTION,
                "temperature": temp,
                "run_index": run_idx,
                "model": MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "responses": responses,
                "metrics": {
                    "cosine_similarity": cosine_scores,
                },
            }

            fname = f"temp{str(temp).replace('.', '')}_run{run_idx}_{ts_now()}.json"
            save_json(run_data, os.path.join(OUTPUT_DIR, fname))

            summary_rows.append({
                "temperature": temp,
                "run_index": run_idx,
                "cosine_mean": cosine_scores["mean"],
                "cosine_pairs": {k: v for k, v in cosine_scores.items() if k != "mean"},
            })
            console.print(
                f"  run {run_idx} → cosine_mean={cosine_scores['mean']:.4f}"
            )

    summary = {
        "experiment": "exp1_temperature",
        "question_key": QUESTION_KEY,
        "question": QUESTION,
        "temperatures": TEMPERATURES,
        "n_runs": N_RUNS,
        "model": MODEL,
        "results": summary_rows,
    }
    save_json(summary, os.path.join(OUTPUT_DIR, "summary.json"))
    console.print(f"\n[green]Concluído. Summary em {OUTPUT_DIR}/summary.json[/green]")


if __name__ == "__main__":
    main()
