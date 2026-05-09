# Documentação dos Dados: LLM-Consensus

## Visão Geral

Os dados deste projeto são **gerados pelos próprios experimentos** via chamadas à API da
OpenAI. Não há dataset externo. Cada execução produz um arquivo JSON em `src/data/` com
respostas dos agentes e métricas pré-computadas.

Todos os arquivos JSON estão versionados no repositório.

## Estrutura de Diretórios

```
src/data/
├── baseline_YYYYMMDD_HHMMSS.json   # log de execução baseline
└── delphi_YYYYMMDD_HHMMSS.json     # log de execução Delphi (futuro)
```

## Arquivos Disponíveis

### Replicações formais — `src/data/` (temperatura 0.0)

| Arquivo | Questão |
|---------|---------|
| `baseline_20260509_104211.json` | Q1 — quantitativa |
| `baseline_20260509_104239.json` | Q1 — quantitativa |
| `baseline_20260509_104310.json` | Q1 — quantitativa |
| `baseline_20260509_110314.json` | Q2 — qualitativa  |
| `baseline_20260509_110358.json` | Q2 — qualitativa  |
| `baseline_20260509_110440.json` | Q2 — qualitativa  |


## Formato dos Arquivos

### Log de Baseline (`experiment_type: "baseline"`)

```json
{
  "experiment_type": "baseline",
  "question": "...",
  "timestamp": "ISO8601",
  "model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "responses": [
    {
      "persona": "economista | cientista_computacao | sociologo",
      "label": "...",
      "system_prompt": "...",
      "question": "...",
      "response": "...",
      "model": "...",
      "timestamp": "ISO8601",
      "input_tokens": 0,
      "output_tokens": 0
    }
  ],
  "metrics": {
    "semantic_similarity": {
      "economista__cientista_computacao": 0.0,
      "economista__sociologo": 0.0,
      "cientista_computacao__sociologo": 0.0,
      "mean": 0.0
    },
    "numeric_estimates": {
      "economista": null,
      "cientista_computacao": null,
      "sociologo": null,
      "mean": null,
      "std_dev": null
    }
  }
}
```

### Log Delphi (`experiment_type: "delphi"`) — estrutura planejada

```json
{
  "experiment_type": "delphi",
  "question": "...",
  "timestamp": "ISO8601",
  "model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "rounds": [
    {
      "round": 1,
      "responses": [ /* mesma estrutura do baseline */ ],
      "synthesis": {
        "convergencias": "...",
        "divergencias": "...",
        "dimensoes_nao_abordadas": "...",
        "feedback": "..."
      },
      "metrics": { /* mesma estrutura de metrics do baseline */ }
    }
  ],
  "final_metrics": { /* métricas da última rodada */ }
}
```

## Como Reproduzir os Dados

Para gerar novas execuções:

1. Configure `OPENAI_API_KEY` em `.env` na raiz do projeto.
2. Instale dependências: `pip install -r requirements.txt`
3. Execute:

```bash
# Q1 — Quantitativa
python src/run_baseline.py \
  -q "Qual percentual dos empregos formais brasileiros você estima que serão automatizados pela IA nos próximos 10 anos? Forneça um número percentual específico com justificativa." \
  --output-dir src/data/

# Q2 — Qualitativa
python src/run_baseline.py \
  -q "Como a inteligência artificial transformará a educação e o mercado de trabalho brasileiro nos próximos 10 anos? Considere tanto oportunidades quanto riscos." \
  --output-dir src/data/
```

## Como Avaliar os Dados

```bash
# Resumo de todos os logs
python src/evaluate.py --input src/data/

# Exportar CSV
python src/evaluate.py --input src/data/ --output results/metrics.csv
```

## Parâmetros do Protocolo Formal

| Parâmetro | Valor |
|-----------|-------|
| Modelo de linguagem | `gpt-4o-mini` |
| Temperatura | `0.0` |
| Máximo de tokens por resposta | `1024` |
| Modelo de embedding | `text-embedding-3-small` |
| Dimensão do embedding | `1536` |
| Métrica de similaridade | Cosseno |
| Replicações por condição | `3` |
| Máximo de rodadas Delphi | `4` |
