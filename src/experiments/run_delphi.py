import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from scipy.spatial.distance import cosine as cosine_distance

load_dotenv()
client = OpenAI()
console = Console()

GENERATION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL  = "text-embedding-3-small"
DEFAULT_TEMPERATURE = 0.8          # temperatura identificada como ótima no Exp 4
SYNTHESIS_TEMPERATURE = 0.2        # síntese: baixa temperatura para consistência
DEFAULT_MAX_ROUNDS = 4
CONVERGENCE_THRESHOLD = 0.05       # parar se d_cos_mean < threshold


import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from shared import PERSONAS


class AgentRevision(BaseModel):
    """Resposta estruturada de um agente em rodada de revisão (round >= 1)."""
    position_changed: bool
    change_justification: Optional[str]  
    response: str                        

class SynthesisOutput(BaseModel):
    """Saída do módulo de síntese intermediária."""
    convergencias: list[str]
    divergencias: list[str]
    dimensoes_nao_abordadas: list[str]
    feedback: str                        

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def pairwise_divergences(embeddings: dict[str, list[float]]) -> dict:
    """Calcula divergência cosseno para todos os pares e a média."""
    personas = list(embeddings.keys())
    results: dict[str, float] = {}
    for i in range(len(personas)):
        for j in range(i + 1, len(personas)):
            key = f"{personas[i]}__{personas[j]}"
            results[key] = float(cosine_distance(
                embeddings[personas[i]], embeddings[personas[j]]
            ))
    results["mean"] = float(np.mean(list(results.values())))
    return results


def call_agent_round0(persona: str, question: str, temperature: float) -> tuple[str, dict]:
    """Rodada 0: resposta independente, sem contexto externo."""
    resp = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": PERSONAS[persona]},
            {"role": "user",   "content": question},
        ],
    )
    tokens = {
        "input_tokens":  resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    return resp.choices[0].message.content, tokens


def call_synthesis(question: str, responses: dict[str, str], round_num: int) -> SynthesisOutput:
    """
    Módulo de síntese: um LLM separado (facilitador) analisa as respostas
    da rodada anterior e gera feedback estruturado em JSON.
    """
    responses_block = "\n\n".join(
        f"=== {p.upper()} ===\n{r}" for p, r in responses.items()
    )
    prompt = (
        f"Você é o facilitador de um processo Delphi, analisando as respostas da Rodada {round_num}.\n\n"
        f"QUESTÃO EM DEBATE:\n{question}\n\n"
        f"RESPOSTAS DOS PARTICIPANTES:\n{responses_block}\n\n"
        "Sua tarefa é produzir uma síntese estruturada que oriente a próxima rodada.\n"
        "Para cada campo:\n"
        "  convergencias: liste os pontos em que as três perspectivas concordam "
        "(mínimo 2 itens, máximo 5, seja específico).\n"
        "  divergencias: liste os pontos de desacordo substantivo entre perspectivas "
        "(mínimo 2 itens, máximo 5, cite qual persona defende cada posição).\n"
        "  dimensoes_nao_abordadas: aspectos relevantes da questão não cobertos por nenhuma resposta.\n"
        "  feedback: parágrafo de 3-5 frases para orientar a revisão, "
        "destacando as divergências mais importantes e convidando os participantes a aprofundá-las."
    )
    resp = client.beta.chat.completions.parse(
        model=GENERATION_MODEL,
        temperature=SYNTHESIS_TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
        response_format=SynthesisOutput,
    )
    return resp.choices[0].message.parsed


def call_agent_revision(
    persona: str,
    question: str,
    previous_response: str,
    synthesis: SynthesisOutput,
    temperature: float,
) -> tuple[AgentRevision, dict]:
    """
    Rodadas N >= 1: o agente recebe sua resposta anterior e a síntese do
    facilitador e decide manter, modificar ou expandir sua posição.
    """
    synthesis_block = (
        f"CONVERGÊNCIAS:\n" + "\n".join(f"  - {c}" for c in synthesis.convergencias) + "\n\n"
        f"DIVERGÊNCIAS:\n"  + "\n".join(f"  - {d}" for d in synthesis.divergencias)  + "\n\n"
        f"DIMENSÕES NÃO ABORDADAS:\n" + "\n".join(f"  - {d}" for d in synthesis.dimensoes_nao_abordadas) + "\n\n"
        f"FEEDBACK DO FACILITADOR:\n{synthesis.feedback}"
    )
    prompt = (
        f"QUESTÃO ORIGINAL:\n{question}\n\n"
        f"SUA RESPOSTA NA RODADA ANTERIOR:\n{previous_response}\n\n"
        f"SÍNTESE DO FACILITADOR:\n{synthesis_block}\n\n"
        "Com base na síntese:\n"
        "  • Você PODE manter sua posição (e aprofundá-la ou responder às divergências).\n"
        "  • Você PODE modificar ou expandir sua posição se algum argumento da síntese "
        "foi convincente — mas apenas se houver razão genuína, não apenas para concordar.\n"
        "  • Se mudar de posição, indique exatamente qual argumento ou divergência motivou "
        "a mudança em 'change_justification'.\n"
        "  • Se manter sua posição, deixe 'change_justification' como null.\n"
        "  • Em 'response', forneça sua resposta completa e revisada."
    )
    resp = client.beta.chat.completions.parse(
        model=GENERATION_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": PERSONAS[persona]},
            {"role": "user",   "content": prompt},
        ],
        response_format=AgentRevision,
    )
    tokens = {
        "input_tokens":  resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
    }
    return resp.choices[0].message.parsed, tokens


def run_delphi(
    question: str,
    output_dir: str = "data/delphi",
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log = {
        "experiment_type": "delphi",
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "model": GENERATION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "temperature": temperature,
        "max_rounds": max_rounds,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "rounds": [],
        "stop_reason": None,
    }

    current_responses: dict[str, str] = {}

    console.rule("[bold blue]RODADA 0 — Consulta Independente")
    round0_tokens = {}
    for persona in PERSONAS:
        console.print(f"  [{persona}] gerando resposta...", end="")
        text, tokens = call_agent_round0(persona, question, temperature)
        current_responses[persona] = text
        round0_tokens[persona] = tokens
        console.print(" ✓")

    embeddings = {p: get_embedding(r) for p, r in current_responses.items()}
    div_r0 = pairwise_divergences(embeddings)
    prev_divergences = div_r0

    log["rounds"].append({
        "round": 0,
        "responses": {
            p: {"response": current_responses[p], **round0_tokens[p]}
            for p in PERSONAS
        },
        "metrics": {"cosine_divergence": div_r0},
        "synthesis": None,
        "revisions": None,
    })
    _print_metrics_table(0, div_r0)

    stop_reason = f"max_rounds ({max_rounds})"

    for round_num in range(1, max_rounds + 1):
        console.rule(f"[bold yellow]RODADA {round_num} — Síntese + Revisão")

        # Critério: convergência atingida
        if round_num > 1 and prev_divergences["mean"] < CONVERGENCE_THRESHOLD:
            stop_reason = f"convergencia_atingida (d={prev_divergences['mean']:.4f})"
            console.print(f"[green]Convergência atingida. Encerrando.[/green]")
            break

        # Síntese
        console.print("  [facilitador] sintetizando respostas...", end="")
        synthesis = call_synthesis(question, current_responses, round_num - 1)
        console.print(" ✓")
        _print_synthesis(synthesis)

        # Revisão por cada agente
        new_responses: dict[str, str] = {}
        revision_logs: dict[str, dict] = {}
        any_changed = False

        for persona in PERSONAS:
            console.print(f"  [{persona}] revisando posição...", end="")
            revision, tokens = call_agent_revision(
                persona, question, current_responses[persona], synthesis, temperature
            )
            new_responses[persona] = revision.response
            revision_logs[persona] = {
                "position_changed":    revision.position_changed,
                "change_justification": revision.change_justification,
                **tokens,
            }
            if revision.position_changed:
                any_changed = True
            status = "✓ [yellow]mudou[/yellow]" if revision.position_changed else "✓ manteve"
            console.print(f" {status}")

        current_responses = new_responses
        embeddings = {p: get_embedding(r) for p, r in current_responses.items()}
        divergences = pairwise_divergences(embeddings)
        prev_divergences = divergences

        log["rounds"].append({
            "round": round_num,
            "responses": {
                p: {
                    "response": current_responses[p],
                    **revision_logs[p],
                }
                for p in PERSONAS
            },
            "metrics": {"cosine_divergence": divergences},
            "synthesis": synthesis.model_dump(),
            "revisions": revision_logs,
        })
        _print_metrics_table(round_num, divergences, revision_logs)

        # Critério: nenhum agente revisou
        if not any_changed:
            stop_reason = "sem_revisao"
            console.print("[green]Nenhum agente revisou posição. Encerrando.[/green]")
            break

    log["stop_reason"] = stop_reason
    log["trajectory"] = _build_trajectory(log["rounds"])

    output_path = Path(output_dir) / f"delphi_{timestamp}.json"
    output_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[bold green]Log salvo em: {output_path}[/bold green]")
    _print_final_trajectory(log["trajectory"])
    return log


def _print_metrics_table(round_num: int, divergences: dict, revisions: dict = None):
    t = Table(title=f"Rodada {round_num} — Divergência Cosseno")
    t.add_column("Par", style="cyan")
    t.add_column("Divergência", justify="right")
    for k, v in divergences.items():
        if k != "mean":
            t.add_row(k.replace("__", " × "), f"{v:.4f}")
    t.add_row("[bold]Média[/bold]", f"[bold]{divergences['mean']:.4f}[/bold]")
    console.print(t)

    if revisions:
        rt = Table(title="Revisões de Posição")
        rt.add_column("Persona")
        rt.add_column("Mudou?", justify="center")
        rt.add_column("Justificativa (resumo)")
        for p, r in revisions.items():
            just = (r.get("change_justification") or "—")[:90]
            rt.add_row(p, "Sim" if r["position_changed"] else "Não", just)
        console.print(rt)

def _print_synthesis(synthesis: SynthesisOutput):
    console.print("\n  [bold cyan]Síntese do facilitador:[/bold cyan]")
    console.print(f"  Convergências ({len(synthesis.convergencias)}): "
                  + "; ".join(synthesis.convergencias[:2]) + " ...")
    console.print(f"  Divergências  ({len(synthesis.divergencias)}): "
                  + "; ".join(synthesis.divergencias[:2]) + " ...")
    console.print()

def _print_final_trajectory(trajectory: dict):
    console.rule("[bold]Trajetória Final")
    console.print("[bold]Divergência cosseno por rodada:[/bold]")
    for i, d in enumerate(trajectory["cosine_divergence_mean"]):
        bar = "█" * max(1, int(d * 150))
        console.print(f"  Rodada {i}: {d:.4f}  {bar}")
    console.print("\n[bold]Taxa de revisão por rodada:[/bold]")
    for i, r in enumerate(trajectory["revision_rate"]):
        if r is not None:
            console.print(f"  Rodada {i}: {r:.0%}")

def _build_trajectory(rounds: list) -> dict:
    return {
        "cosine_divergence_mean": [
            r["metrics"]["cosine_divergence"]["mean"] for r in rounds
        ],
        "revision_rate": [
            (
                sum(1 for v in r["revisions"].values() if v["position_changed"]) / len(PERSONAS)
                if r.get("revisions") else None
            )
            for r in rounds
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ciclo Delphi com LLMs multiagente")
    parser.add_argument(
        "-q", "--question",
        default=(
            "Qual é o maior obstáculo para adoção em larga escala de "
            "IA nas empresas ao redor do mundo?"
        ),
        help="Questão para o ciclo Delphi (default: obstaculo_adocao)",
    )
    parser.add_argument("-o", "--output-dir", default="data/delphi")
    parser.add_argument("-r", "--rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    run_delphi(
        question=args.question,
        output_dir=args.output_dir,
        max_rounds=args.rounds,
        temperature=args.temperature,
    )
