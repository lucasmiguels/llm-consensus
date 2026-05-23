"""Utilitários compartilhados entre experimentos."""
import json
import os
from datetime import datetime, timezone

import numpy as np
import openai
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

MODEL = "gpt-4o-mini"
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

PERSONA_NAMES = list(PERSONAS.keys())
PERSONA_PAIRS = [(0, 1), (0, 2), (1, 2)]


def load_client() -> openai.OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrada. Crie .env com OPENAI_API_KEY=sk-...")
    return openai.OpenAI(api_key=api_key)


def query_persona(
    client: openai.OpenAI,
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
    )
    if temperature is not None:
        kwargs["temperature"] = temperature

    completion = client.chat.completions.create(**kwargs)
    response_text = completion.choices[0].message.content or ""
    return {
        "persona": persona_name,
        "label": PERSONA_LABELS[persona_name],
        "response": response_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_tokens": completion.usage.prompt_tokens,
        "output_tokens": completion.usage.completion_tokens,
        "temperature": temperature,
    }


def compute_cosine_sim(texts: list[str], labels: list[str], client: openai.OpenAI) -> dict:
    emb_response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embeddings = np.array([d.embedding for d in emb_response.data])

    scores = {}
    for i, j in PERSONA_PAIRS:
        key = f"{labels[i]}__{labels[j]}"
        scores[key] = float(1.0 - cosine(embeddings[i], embeddings[j]))

    scores["mean"] = float(np.mean([v for k, v in scores.items()]))
    return scores


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
