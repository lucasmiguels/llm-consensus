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

## Baseline (Entrega 2)

O baseline executa três agentes com personas distintas (economista, cientista da computação, sociólogo) sobre a mesma questão, de forma independente e sem rodadas de feedback. Serve como referência para comparação com o sistema iterativo Delphi.

### Execução via script

```bash
# Questão quantitativa
python src/run_baseline.py -q "Qual percentual dos empregos formais brasileiros você estima que serão automatizados pela IA nos próximos 10 anos? Forneça um número percentual específico com justificativa."

# Questão qualitativa
python src/run_baseline.py -q "Como a inteligência artificial transformará a educação e o mercado de trabalho brasileiro nos próximos 10 anos? Considere tanto oportunidades quanto riscos."
```

### Execução via notebook

```bash
jupyter notebook notebooks/baseline.ipynb
```

Selecione a questão na célula 3 e execute todas as células em ordem.

### Saída

- **Terminal:** tabelas com respostas resumidas, similaridade semântica e estimativas numéricas.
- **Arquivo:** `data/baseline_YYYYMMDD_HHMMSS.json` com log completo (respostas, tokens, métricas).


---

## Referências

Nóbrega, L. et al. **AI-Delphi: Emulating Personas Toward Machine–Machine Collaboration**. 2025.
