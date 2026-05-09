import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
import numpy as np
import openai
from scipy.spatial.distance import cosine
from rich.console import Console
from rich.table import Table
from rich import box


MODEL = "gpt-4o-mini"
MAX_TOKENS = 1024
EMBEDDING_MODEL = "text-embedding-3-small"

PERSONAS: dict[str, str] = {
    "economista": (
        "Você é um economista especializado em mercado de trabalho brasileiro e automação. "
        "Analise a questão a seguir sob a perspectiva econômica, considerando: elasticidade "
        "do emprego, setores mais vulneráveis à automação, impactos distributivos e possíveis "
        "políticas de ajuste. Seja direto e use dados ou estimativas quando relevante. "
        "Se a questão pedir uma estimativa numérica, forneça um número específico com justificativa. "
        "Responda em português."
    ),
    "cientista_computacao": (
        "Você é um cientista da computação especializado em inteligência artificial e sistemas "
        "de automação. Analise a questão a seguir sob a perspectiva técnica, considerando: "
        "capacidades atuais e projetadas dos sistemas de IA, tipos de tarefas automatizáveis "
        "(rotineiras vs. cognitivas complexas), limitações técnicas e horizontes de tempo "
        "realistas para adoção em larga escala no contexto brasileiro. "
        "Se a questão pedir uma estimativa numérica, forneça um número específico com justificativa. "
        "Responda em português."
    ),
    "sociologo": (
        "Você é um sociólogo especializado em trabalho, tecnologia e desigualdade social no Brasil. "
        "Analise a questão a seguir sob a perspectiva sociológica, considerando: impactos "
        "diferenciados por classe, gênero e região, transformações nas relações de trabalho, "
        "resistências sociais, papel das instituições e efeitos sobre a coesão social. "
        "Se a questão pedir uma estimativa numérica, forneça um número específico com justificativa. "
        "Responda em português."
    ),
}

PERSONA_LABELS = {
    "economista": "Economista",
    "cientista_computacao": "Cientista da Computação",
    "sociologo": "Sociólogo",
}


def load_client() -> openai.OpenAI:
    """Carrega variáveis de ambiente e retorna cliente OpenAI."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            "[ERRO] Variável OPENAI_API_KEY não encontrada.\n"
            "Crie um arquivo .env na raiz do projeto com:\n"
            "  OPENAI_API_KEY=sk-...",
            file=sys.stderr,
        )
        sys.exit(1)
    return openai.OpenAI(api_key=api_key)


def query_agent(
    client: openai.OpenAI,
    persona_name: str,
    system_prompt: str,
    question: str,
) -> dict:
    """Executa uma chamada de API para um agente com persona específica."""
    ts = datetime.now(timezone.utc).isoformat()
    completion = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    response_text = completion.choices[0].message.content or ""
    return {
        "persona": persona_name,
        "label": PERSONA_LABELS[persona_name],
        "system_prompt": system_prompt,
        "question": question,
        "response": response_text,
        "model": MODEL,
        "timestamp": ts,
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
    }


def run_baseline(question: str, client: openai.OpenAI) -> list[dict]:
    """Consulta os três agentes de forma independente e retorna lista de respostas."""
    results = []
    for persona_name, system_prompt in PERSONAS.items():
        results.append(query_agent(client, persona_name, system_prompt, question))
    return results


def extract_number(text: str) -> float | None:
    """Extrai o primeiro valor percentual ou numérico do texto."""
    pattern = r"(\d+(?:[.,]\d+)?)\s*(?:%|por\s*cento)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        raw = match.group(1).replace(",", ".")
        return float(raw)
    return None


def compute_metrics(
    responses: list[dict],
    client: openai.OpenAI,
) -> dict:
    """Calcula similaridade semântica entre pares e desvio-padrão de estimativas numéricas."""
    texts = [r["response"] for r in responses]
    labels = [r["persona"] for r in responses]

    emb_response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embeddings = np.array([d.embedding for d in emb_response.data])

    pairs = [
        (labels[0], labels[1]),
        (labels[0], labels[2]),
        (labels[1], labels[2]),
    ]
    sim_scores = {}
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        key = f"{labels[i]}__{labels[j]}"
        sim_scores[key] = float(1.0 - cosine(embeddings[i], embeddings[j]))

    _ = pairs 
    mean_sim = float(np.mean(list(sim_scores.values())))

    numbers = {r["persona"]: extract_number(r["response"]) for r in responses}
    valid_nums = [v for v in numbers.values() if v is not None]
    numeric_metrics: dict = {k: v for k, v in numbers.items()}
    if len(valid_nums) >= 2:
        numeric_metrics["mean"] = float(np.mean(valid_nums))
        numeric_metrics["std_dev"] = float(np.std(valid_nums, ddof=0))
    else:
        numeric_metrics["mean"] = None
        numeric_metrics["std_dev"] = None

    return {
        "semantic_similarity": {**sim_scores, "mean": mean_sim},
        "numeric_estimates": numeric_metrics,
    }


def build_log(question: str, responses: list[dict], metrics: dict) -> dict:
    """Monta o dicionário completo do log de execução."""
    return {
        "experiment_type": "baseline",
        "question": question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "responses": responses,
        "metrics": metrics,
    }


def save_log(log: dict, output_dir: str = "data") -> str:
    """Salva o log em JSON e retorna o caminho do arquivo."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"baseline_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    return path


def print_summary_table(
    responses: list[dict],
    metrics: dict,
    console: Console,
) -> None:
    """Imprime tabelas resumidas com respostas e métricas."""
    # Tabela 1 — Respostas
    t1 = Table(title="Respostas dos Agentes", box=box.ROUNDED, show_lines=True)
    t1.add_column("Agente", style="bold cyan", width=22)
    t1.add_column("Resposta (resumo)", width=60)
    t1.add_column("Tokens", justify="right", width=8)
    for r in responses:
        preview = r["response"][:220].replace("\n", " ")
        if len(r["response"]) > 220:
            preview += "…"
        t1.add_row(
            r["label"],
            preview,
            str(r["input_tokens"] + r["output_tokens"]),
        )
    console.print(t1)

    # Tabela 2 — Similaridade semântica
    sim = metrics["semantic_similarity"]
    t2 = Table(title="Similaridade Semântica (cosseno)", box=box.ROUNDED)
    t2.add_column("Par", style="bold")
    t2.add_column("Score", justify="right")
    for key, val in sim.items():
        if key == "mean":
            continue
        a, b = key.split("__")
        t2.add_row(
            f"{PERSONA_LABELS[a]} × {PERSONA_LABELS[b]}",
            f"{val:.4f}",
        )
    t2.add_row("[bold]Média[/bold]", f"[bold]{sim['mean']:.4f}[/bold]")
    console.print(t2)

    # Tabela 3 — Estimativas numéricas (condicional)
    nums = metrics["numeric_estimates"]
    valid = {k: v for k, v in nums.items() if k not in ("mean", "std_dev") and v is not None}
    if valid:
        t3 = Table(title="Estimativas Numéricas (%)", box=box.ROUNDED)
        t3.add_column("Agente", style="bold")
        t3.add_column("Estimativa (%)", justify="right")
        for persona, val in valid.items():
            t3.add_row(PERSONA_LABELS[persona], f"{val:.1f}")
        if nums["mean"] is not None:
            t3.add_row("[bold]Média[/bold]", f"[bold]{nums['mean']:.1f}[/bold]")
        if nums["std_dev"] is not None:
            t3.add_row("[bold]Desvio-padrão[/bold]", f"[bold]{nums['std_dev']:.1f}[/bold]")
        console.print(t3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline LLM-Consensus: consulta paralela sem iteração."
    )
    parser.add_argument(
        "-q",
        "--question",
        required=True,
        help="Questão a ser enviada aos três agentes.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Diretório de saída para o log JSON (padrão: data).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    console.rule("[bold blue]LLM-Consensus — Baseline[/bold blue]")
    console.print(f"\n[bold]Questão:[/bold] {args.question}\n")

    client = load_client()

    with console.status("Consultando agentes…", spinner="dots"):
        responses = run_baseline(args.question, client)

    console.print("[green]Respostas recebidas.[/green]\n")

    with console.status("Calculando métricas…", spinner="dots"):
        metrics = compute_metrics(responses, client)

    log = build_log(args.question, responses, metrics)
    log_path = save_log(log, args.output_dir)

    print_summary_table(responses, metrics, console)
    console.print(f"\n[bold]Log salvo em:[/bold] {log_path}")


if __name__ == "__main__":
    main()
