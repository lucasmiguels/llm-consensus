# LLM-Consensus — Tornar Personas Relevantes

**Disciplina:** Busca e Recuperação da Informação  
**Dupla:** Lucas Miguel · Mauro Jorge Ernesto  
**Repositório:** https://github.com/lucasmiguels/llm-consensus  

---

## Resumo

Investigamos quais estratégias produzem divergência semântica substancial entre personas
de LLMs antes de qualquer processo iterativo, condição necessária para que deliberações
estruturadas (Método Delphi) sejam não-triviais. Avaliamos temperatura, formulação de
questão, tipo de resposta e sua interação usando um painel de três personas
(Economista, Cientista da Computação, Sociólogo) e duas métricas complementares de
divergência (cosseno global e sentence-cosine).

---

## Estrutura do Repositório

```
llm-consensus/
├── src/
│   ├── run_baseline.py                    # baseline original
│   ├── experiments/
│   │   ├── shared.py                      # personas, modelos, utilitários
│   │   ├── run_temperature.py             # Exp 1: efeito da temperatura
│   │   ├── run_qualitative.py             # Exp 2a: questões polêmicas
│   │   ├── run_quantitative.py            # Exp 2b: estimativas numéricas
│   │   ├── run_grid.py                    # Exp 4: grade temp × questão
│   │   ├── enrich_metrics.py              # Exp 3: sentence-cosine (local)
│   │   ├── run_delphi.py                  # Exp 5: ciclo Delphi iterativo
│   │   └── compare_delphi_baseline.py     # comparação Delphi × baseline
│   ├── plots/
│   │   └── generate_plots.py              # gera todas as figuras
│   └── data/                              # logs JSON de cada experimento
├── data/
│   └── delphi/                            # logs do ciclo Delphi
├── demo/
│   └── run_demo.py                        # demo interativo do ciclo Delphi
├── reports/
│   ├── figures/                           # figuras geradas
│   └── 05_artigo_final.tex                # artigo final
├── make_results.sh                        # reproduz todos os resultados
└── requirements.txt
```

---

## Pré-requisitos

- Python 3.10+
- Chave de API OpenAI com acesso a `gpt-4o-mini` e `text-embedding-3-small`

---

## Instalação

```bash
git clone https://github.com/lucasmiguels/llm-consensus.git
cd llm-consensus
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
```

### Configuração

Crie `.env` na raiz do projeto:

```
OPENAI_API_KEY=sk-...
```

---

## Demo Rápida

Executa o ciclo Delphi completo com a questão ótima identificada nos experimentos:

```bash
python demo/run_demo.py
```

Flags opcionais:

```bash
python demo/run_demo.py --question "Sua questão aqui" --rounds 3 --temperature 0.8
```

---

## Reprodução dos Resultados do Artigo

```bash
bash make_results.sh
```

O script executa todos os experimentos em ordem, enriquece as métricas e gera as
figuras usadas no artigo. Os logs ficam em `src/data/` e as figuras em `reports/figures/`.

**Atenção:** o script faz chamadas à API OpenAI. Custo estimado: ~US$ 0,40.

Para reproduzir apenas o ciclo Delphi (Exp 5):

```bash
python src/experiments/run_delphi.py
python src/experiments/compare_delphi_baseline.py \
  --delphi data/delphi/delphi_<timestamp>.json \
  --baseline src/data/exp4/temp08_obstaculo_adocao_run0_<timestamp>.json
```

---

## Experimentos Individuais

```bash
# Exp 1 — efeito da temperatura
python src/experiments/run_temperature.py

# Exp 2a — questões polêmicas qualitativas
python src/experiments/run_qualitative.py

# Exp 2b — estimativas numéricas
python src/experiments/run_quantitative.py

# Exp 3 — enriquecimento com sentence-cosine (sem custo de API)
python src/experiments/enrich_metrics.py --input-dir src/data/exp1
python src/experiments/enrich_metrics.py --input-dir src/data/exp2a

# Exp 4 — grade temperatura × questão (125 execuções)
python src/experiments/run_grid.py

# Gerar figuras
python src/plots/generate_plots.py
```

---

## Referências

Nóbrega, L. et al. **AI-Delphi: Emulating Personas Toward Machine–Machine Collaboration**. 2025.
