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

# temperatura=None → default do modelo (1.0 no gpt-4o-mini)
TEMPERATURE = None

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
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "exp2a"
)


def main() -> None:
    client = load_client()
    console = Console()
    console.rule("[bold blue]Exp 2a — Questões Qualitativas Polêmicas[/bold blue]")
    console.print(f"Temperatura: default do modelo (None)\n")

    summary_rows = []

    for q_key, question in QUESTIONS.items():
        console.print(f"[bold cyan]questão: {q_key}[/bold cyan]")
        for run_idx in range(N_RUNS):
            responses = [
                query_persona(client, p, question, temperature=TEMPERATURE)
                for p in PERSONA_NAMES
            ]
            labels = [r["persona"] for r in responses]
            texts = [r["response"] for r in responses]
            cosine_scores = compute_cosine_sim(texts, labels, client)

            run_data = {
                "experiment": "exp2a_qualitative",
                "question_key": q_key,
                "question": question,
                "temperature": TEMPERATURE,
                "run_index": run_idx,
                "model": MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "responses": responses,
                "metrics": {
                    "cosine_similarity": cosine_scores,
                },
            }

            fname = f"{q_key}_run{run_idx}_{ts_now()}.json"
            save_json(run_data, os.path.join(OUTPUT_DIR, fname))

            summary_rows.append({
                "question_key": q_key,
                "run_index": run_idx,
                "cosine_mean": cosine_scores["mean"],
                "cosine_pairs": {k: v for k, v in cosine_scores.items() if k != "mean"},
            })
            console.print(
                f"  run {run_idx} → cosine_mean={cosine_scores['mean']:.4f}"
            )

    summary = {
        "experiment": "exp2a_qualitative",
        "temperature": TEMPERATURE,
        "n_runs": N_RUNS,
        "model": MODEL,
        "questions": list(QUESTIONS.keys()),
        "results": summary_rows,
    }
    save_json(summary, os.path.join(OUTPUT_DIR, "summary.json"))
    console.print(f"\n[green]Concluído. Summary em {OUTPUT_DIR}/summary.json[/green]")


if __name__ == "__main__":
    main()
