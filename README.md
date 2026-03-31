# Orquestração de Deliberação Estruturada em Sistemas Multiagente

**Disciplina:** Busca e Recuperação da Informação  

**Dupla:** Lucas Miguel · Mauro Jorge

**Repositório:** [https://github.com/lucasmiguels/llm-consensus](https://github.com/lucasmiguels/llm-consensus)

---

## Objetivo

Implementar e avaliar um framework de debate iterativo entre múltiplos LLMs inspirado no Método Delphi, capaz de conduzir rodadas de discussão estruturada, sintetizar convergências e divergências, e encerrar o processo com um relatório justificado de consenso ou divergência irredutível.

---

## Estrutura do Repositório

```
llm-consensus/
├── data/          # Questões de entrada e logs de rodadas
├── notebooks/     # Análise exploratória e visualizações
├── reports/       # Relatórios e documentos do projeto
└── src/           # Código-fonte do sistema
```

---

## Instruções de Execução (Preliminares)

### Pré-requisitos

- Python 3.10+
- Chave de API para ao menos um provedor de LLM (Anthropic, OpenAI ou equivalente)

### Instalação

```bash
git clone https://github.com/lucasmiguels/llm-consensus.git
cd llm-consensus
pip install -r requirements.txt
```

### Configuração

Crie um arquivo `.env` na raiz com suas chaves de API:

```
ANTHROPIC_API_KEY=sk-...
OPENAI_API_KEY=sk-...   # opcional
```

### Execução (em breve)

```bash
python src/run_debate.py --question "Qual o impacto da IA no mercado de trabalho brasileiro?" --rounds 4
```

---

## Referências

Nóbrega, L. et al. **AI-Delphi: Emulating Personas Toward Machine–Machine Collaboration**. 2025.
