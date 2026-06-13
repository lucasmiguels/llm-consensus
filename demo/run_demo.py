"""
demo/run_demo.py — Demo interativo do ciclo Delphi

Executa o ciclo Delphi com a configuração ótima identificada nos experimentos
(obstaculo_adocao, temperatura 0.8) e exibe a trajetória de convergência.

Uso:
  python demo/run_demo.py
  python demo/run_demo.py --question "Sua questão aqui" --rounds 3
  python demo/run_demo.py --temperature 1.0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()

BANNER = """
[bold cyan]LLM-Consensus[/bold cyan] — Demo do Ciclo Delphi
Três personas LLM (Economista, Cientista da Computação, Sociólogo) debatem
uma questão em rodadas iterativas com síntese e revisão de posição.
"""

DEFAULT_QUESTION = (
    "Qual é o maior obstáculo para adoção em larga escala de "
    "IA nas empresas ao redor do mundo?"
)


def main():
    parser = argparse.ArgumentParser(description="Demo do ciclo Delphi com LLMs")
    parser.add_argument(
        "-q", "--question",
        default=DEFAULT_QUESTION,
        help="Questão para debate (default: obstaculo_adocao)",
    )
    parser.add_argument(
        "-r", "--rounds",
        type=int,
        default=4,
        help="Número máximo de rodadas (default: 4)",
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.8,
        help="Temperatura de geração (default: 0.8, ótimo identificado no Exp 4)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold red]ERRO:[/bold red] OPENAI_API_KEY não encontrada.\n"
            "Crie um arquivo .env na raiz do projeto com:\n"
            "  OPENAI_API_KEY=sk-..."
        )
        sys.exit(1)

    console.print(Panel(BANNER.strip(), border_style="cyan"))
    console.print(f"\n[bold]Questão:[/bold] {args.question}")
    console.print(f"[bold]Temperatura:[/bold] {args.temperature}")
    console.print(f"[bold]Rodadas máx.:[/bold] {args.rounds}\n")

    from experiments.run_delphi import run_delphi

    log = run_delphi(
        question=args.question,
        output_dir="data/delphi",
        max_rounds=args.rounds,
        temperature=args.temperature,
    )

    traj = log["trajectory"]
    divs = traj["cosine_divergence_mean"]
    console.print("\n")
    console.print(Panel(
        f"[bold]Stop:[/bold] {log['stop_reason']}\n"
        f"[bold]Rodadas executadas:[/bold] {len(divs)}\n"
        f"[bold]Divergência Rodada 0:[/bold] {divs[0]:.4f}\n"
        f"[bold]Divergência final:[/bold]   {divs[-1]:.4f}  "
        f"({'↓' if divs[-1] < divs[0] else '↑'} {abs(divs[-1]-divs[0]):.4f})",
        title="Resultado",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
