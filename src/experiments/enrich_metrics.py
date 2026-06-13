import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from rich.console import Console
from rich.table import Table
from rich import box

from experiments.shared import PERSONA_NAMES, PERSONA_PAIRS, save_json

SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_sbert: SentenceTransformer | None = None


def get_sbert() -> SentenceTransformer:
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer(SBERT_MODEL)
    return _sbert


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s]


def sentence_cosine_pair(text_a: str, text_b: str) -> dict:
    model = get_sbert()
    sents_a = split_sentences(text_a)
    sents_b = split_sentences(text_b)
    if not sents_a or not sents_b:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    emb_a = model.encode(sents_a, convert_to_numpy=True)
    emb_b = model.encode(sents_b, convert_to_numpy=True)
    sim_matrix = sk_cosine(emb_a, emb_b)

    precision = float(sim_matrix.max(axis=1).mean())  # cada sentença A → melhor match em B
    recall = float(sim_matrix.max(axis=0).mean())     # cada sentença B → melhor match em A
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_sentence_cosine(responses: list[dict]) -> dict:
    labels = [r["persona"] for r in responses]
    texts = [r["response"] for r in responses]
    scores = {}
    for i, j in PERSONA_PAIRS:
        key = f"{labels[i]}__{labels[j]}"
        scores[key] = sentence_cosine_pair(texts[i], texts[j])
    scores["mean_f1"] = float(np.mean([v["f1"] for v in scores.values()]))
    return scores


def enrich_delphi_log(run: dict) -> dict:
    """Enrich a Delphi log by computing sentence-cosine for each round."""
    enriched = dict(run)
    enriched["rounds"] = []
    sent_div_per_round = []

    for r in run["rounds"]:
        round_copy = dict(r)
        responses_dict = r.get("responses", {})
        responses_list = [
            {"persona": p, "response": responses_dict[p]["response"]}
            for p in PERSONA_NAMES
            if p in responses_dict
        ]
        if len(responses_list) >= 2:
            sent_scores = compute_sentence_cosine(responses_list)
            sent_div = 1.0 - sent_scores["mean_f1"]
        else:
            sent_scores = {}
            sent_div = None

        round_copy["metrics"] = dict(r.get("metrics", {}))
        round_copy["metrics"]["sentence_cosine"] = sent_scores
        round_copy["metrics"]["sentence_cosine_divergence"] = sent_div
        enriched["rounds"].append(round_copy)
        if sent_div is not None:
            sent_div_per_round.append(sent_div)

    traj = dict(run.get("trajectory", {}))
    traj["sentence_cosine_divergence_mean"] = sent_div_per_round
    enriched["trajectory"] = traj
    return enriched


def load_runs(input_dir: str) -> list[tuple[str, dict]]:
    pattern = os.path.join(input_dir, "*.json")
    runs = []
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        if fname in ("summary.json", "ranking.json"):
            continue
        if fname.startswith("enriched"):
            continue
        with open(path, encoding="utf-8") as f:
            runs.append((fname, json.load(f)))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Enriquece experimentos com sentence cosine")
    parser.add_argument("--input-dir", required=True, help="Diretório de dados do experimento")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = os.path.join(input_dir, "enriched")

    console = Console()
    console.rule("[bold blue]Exp 3 — Enriquecimento de Métricas[/bold blue]")
    console.print(f"Entrada : {input_dir}")
    console.print(f"Saída   : {output_dir}")
    console.print(f"Métricas: sentence-cosine\n")

    runs = load_runs(input_dir)
    if not runs:
        console.print(f"[red]Nenhum arquivo de run encontrado em {input_dir}[/red]")
        sys.exit(1)
    console.print(f"{len(runs)} runs encontrados.\n")

    summary_rows = []

    for i, (fname, run) in enumerate(runs):
        # ── Delphi log ────────────────────────────────────────────────────────
        if run.get("experiment_type") == "delphi":
            console.print(f"[{i+1:>3}/{len(runs)}] {fname} [delphi]…", end=" ")
            enriched = enrich_delphi_log(run)
            save_json(enriched, os.path.join(output_dir, fname))
            traj = enriched["trajectory"]
            sent_divs = traj.get("sentence_cosine_divergence_mean", [])
            cos_divs = traj.get("cosine_divergence_mean", [])
            console.print(f"rounds={len(enriched['rounds'])}  sent_div_r0={sent_divs[0]:.4f}" if sent_divs else "no rounds")
            summary_rows.append({
                "file": fname,
                "question_key": run.get("question", "delphi")[:40],
                "temperature": run.get("temperature"),
                "run_index": None,
                "cosine_divergence": (1.0 - (1.0 - cos_divs[0])) if cos_divs else None,
                "sentence_cosine_divergence": sent_divs[0] if sent_divs else None,
            })
            continue

        # ── Regular experiment log ─────────────────────────────────────────────
        responses = run.get("responses", [])

        if responses and "justification" in responses[0]:
            for r in responses:
                r["response"] = r["justification"]

        console.print(f"[{i+1:>3}/{len(runs)}] {fname}…", end=" ")

        sent_scores = compute_sentence_cosine(responses)
        sent_div = 1.0 - sent_scores["mean_f1"]

        cosine_mean = run.get("metrics", {}).get("cosine_similarity", {}).get("mean")
        cosine_div = (1.0 - cosine_mean) if cosine_mean is not None else None

        enriched = dict(run)
        enriched["metrics"] = dict(run.get("metrics", {}))
        enriched["metrics"]["sentence_cosine"] = sent_scores
        enriched["metrics"]["sentence_cosine_divergence"] = sent_div

        save_json(enriched, os.path.join(output_dir, fname))

        row: dict = {
            "file": fname,
            "question_key": run.get("question_key", "—"),
            "temperature": run.get("temperature"),
            "run_index": run.get("run_index"),
            "cosine_divergence": cosine_div,
            "sentence_cosine_divergence": sent_div,
        }

        summary_rows.append(row)

        msg = f"sent_div={sent_div:.4f}"
        if cosine_div is not None:
            msg += f"  cosine_div={cosine_div:.4f}"
        console.print(msg)

    # Correlações entre métricas
    cosine_vals = np.array([r["cosine_divergence"] for r in summary_rows if r["cosine_divergence"] is not None])
    sent_vals = np.array([r["sentence_cosine_divergence"] for r in summary_rows])
    corr_cosine_sent = float(np.corrcoef(cosine_vals, sent_vals)[0, 1]) if len(cosine_vals) == len(sent_vals) else None

    summary = {
        "experiment": "exp3_enrich_metrics",
        "input_dir": input_dir,
        "sbert_model": SBERT_MODEL,
        "n_runs": len(summary_rows),
        "correlations": {
            "cosine_vs_sentence_cosine": corr_cosine_sent,
        },
        "results": summary_rows,
    }
    save_json(summary, os.path.join(output_dir, "summary.json"))

    # Tabela comparativa
    t = Table(title="Divergência por Métrica", box=box.ROUNDED)
    t.add_column("Arquivo", width=35)
    t.add_column("Cosseno div", justify="right")
    t.add_column("Sent-cosine div", justify="right")

    for r in summary_rows:
        cols = [
            r["file"][:35],
            f"{r['cosine_divergence']:.4f}" if r["cosine_divergence"] is not None else "—",
            f"{r['sentence_cosine_divergence']:.4f}",
        ]
        t.add_row(*cols)
    console.print(t)

    console.print("\n[bold]Correlações entre métricas:[/bold]")
    console.print(f"  Cosseno × Sentence-cosine : {corr_cosine_sent:.4f}" if corr_cosine_sent else "  —")

    console.print(f"\n[green]Concluído. Arquivos em {output_dir}[/green]")


if __name__ == "__main__":
    main()
