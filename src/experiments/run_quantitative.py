import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pydantic import BaseModel
from rich.console import Console

from experiments.shared import (
    MODEL,
    EMBEDDING_MODEL,
    PERSONAS,
    PERSONA_LABELS,
    PERSONA_NAMES,
    PERSONA_PAIRS,
    compute_cosine_sim,
    load_client,
    save_json,
    ts_now,
)

# temperatura=None → default do modelo (1.0 no gpt-4o-mini)
TEMPERATURE = None

QUESTIONS: dict[str, str] = {
    "call_centers_anos": (
        "Em quantos anos 80% dos call centers globais terão sido substituídos por IA? "
        "Forneça um número específico com justificativa."
    ),
    "pib_crescimento_pct": (
        "Qual percentual do crescimento do PIB mundial nos próximos 10 anos será atribuível à IA? "
        "Forneça um número percentual específico com justificativa."
    ),
    "empregos_automatizados_pct": (
        "Qual percentual dos empregos formais globais serão automatizados pela IA nos próximos 15 anos? "
        "Forneça um número percentual específico com justificativa."
    ),
    "gap_desenvolvidos_anos": (
        "Em quantos anos países em desenvolvimento alcançarão países desenvolvidos em adoção de IA? "
        "Forneça um número específico com justificativa."
    ),
}

N_RUNS = 5

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "exp2b"
)


class QuantitativeResponse(BaseModel):
    estimate_pct: float
    justification: str


def query_persona_structured(
    client,
    persona_name: str,
    question: str,
    temperature: float | None = None,
) -> dict:
    kwargs: dict = dict(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": PERSONAS[persona_name]},
            {"role": "user", "content": question},
        ],
        response_format=QuantitativeResponse,
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    completion = client.beta.chat.completions.parse(**kwargs)
    parsed: QuantitativeResponse = completion.choices[0].message.parsed
    return {
        "persona": persona_name,
        "label": PERSONA_LABELS[persona_name],
        "estimate_pct": parsed.estimate_pct,
        "justification": parsed.justification,
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
        "temperature": temperature,
    }


def compute_numeric_metrics(responses: list[dict]) -> dict:
    estimates = {r["persona"]: r["estimate_pct"] for r in responses}
    values = list(estimates.values())
    return {
        **estimates,
        "mean": float(np.mean(values)),
        "std_dev": float(np.std(values, ddof=0)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    client = load_client()
    console = Console()
    console.rule("[bold blue]Exp 2b — Questões Quantitativas (Pydantic)[/bold blue]")
    console.print("Temperatura: default do modelo (None)\n")

    summary_rows = []

    for q_key, question in QUESTIONS.items():
        console.print(f"[bold cyan]questão: {q_key}[/bold cyan]")
        for run_idx in range(N_RUNS):
            responses = [
                query_persona_structured(client, p, question, temperature=TEMPERATURE)
                for p in PERSONA_NAMES
            ]

            labels = [r["persona"] for r in responses]
            justifications = [r["justification"] for r in responses]
            cosine_scores = compute_cosine_sim(justifications, labels, client)
            numeric_metrics = compute_numeric_metrics(responses)

            run_data = {
                "experiment": "exp2b_quantitative",
                "question_key": q_key,
                "question": question,
                "temperature": TEMPERATURE,
                "run_index": run_idx,
                "model": MODEL,
                "embedding_model": EMBEDDING_MODEL,
                "responses": responses,
                "metrics": {
                    "cosine_similarity_justifications": cosine_scores,
                    "numeric_estimates": numeric_metrics,
                },
            }

            fname = f"{q_key}_run{run_idx}_{ts_now()}.json"
            save_json(run_data, os.path.join(OUTPUT_DIR, fname))

            summary_rows.append({
                "question_key": q_key,
                "run_index": run_idx,
                "cosine_mean": cosine_scores["mean"],
                "std_dev": numeric_metrics["std_dev"],
                "estimates": {p: numeric_metrics[p] for p in PERSONA_NAMES},
            })
            console.print(
                f"  run {run_idx} → "
                f"cosine_mean={cosine_scores['mean']:.4f}  "
                f"std_dev={numeric_metrics['std_dev']:.2f}%  "
                f"estimates={[numeric_metrics[p] for p in PERSONA_NAMES]}"
            )

    summary = {
        "experiment": "exp2b_quantitative",
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
