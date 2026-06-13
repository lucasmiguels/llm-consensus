# Custo estimado: ~US$ 0.40 em chamadas OpenAI

set -e  # abortar em qualquer erro

echo "========================================"
echo " LLM-Consensus — Reprodução de Resultados"
echo "========================================"

# Verificar chave de API
if [ -z "$OPENAI_API_KEY" ] && [ ! -f ".env" ]; then
    echo "ERRO: OPENAI_API_KEY não encontrada. Crie um arquivo .env com:"
    echo "  OPENAI_API_KEY=sk-..."
    exit 1
fi

echo ""
echo "[1/7] Exp 1 — Efeito da temperatura (35 execuções)..."
python src/experiments/run_temperature.py

echo ""
echo "[2/7] Exp 2a — Questões polêmicas qualitativas (25 execuções)..."
python src/experiments/run_qualitative.py

echo ""
echo "[3/7] Exp 2b — Estimativas numéricas (20 execuções)..."
python src/experiments/run_quantitative.py

echo ""
echo "[4/7] Exp 3 — Enriquecimento com sentence-cosine (local, sem custo de API)..."
python src/experiments/enrich_metrics.py --input-dir src/data/exp1
python src/experiments/enrich_metrics.py --input-dir src/data/exp2a

echo ""
echo "[5/7] Exp 4 — Grade temperatura × questão (125 execuções)..."
python src/experiments/run_grid.py
python src/experiments/enrich_metrics.py --input-dir src/data/exp4

echo ""
echo "[6/7] Exp 5 — Ciclo Delphi (1 execução)..."
python src/experiments/run_delphi.py
python src/experiments/enrich_metrics.py --input-dir data/delphi

# Comparar Delphi vs baseline (usa primeiro run disponível do exp4 temp=0.8)
BASELINE=$(ls src/data/exp4/temp08_obstaculo_adocao_run0_*.json 2>/dev/null | head -1)
DELPHI=$(ls data/delphi/delphi_*.json 2>/dev/null | grep -v enriched | head -1)

if [ -n "$BASELINE" ] && [ -n "$DELPHI" ]; then
    python src/experiments/compare_delphi_baseline.py \
        --delphi "$DELPHI" \
        --baseline "$BASELINE"
else
    echo "AVISO: arquivos de baseline ou Delphi não encontrados para comparação."
fi

echo ""
echo "[7/7] Gerando figuras..."
python src/plots/generate_plots.py

echo ""
echo "========================================"
echo " Concluído. Figuras em reports/figures/"
echo "========================================"
