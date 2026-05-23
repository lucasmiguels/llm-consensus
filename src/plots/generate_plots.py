import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "src", "data")
OUT  = os.path.join(ROOT, "reports", "figures")
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = sns.color_palette("muted")

QUESTION_LABELS = {
    "obstaculo_adocao":   "Obstáculo\nadoção",
    "concentracao_renda": "Concentração\nde renda",
    "relacoes_poder":     "Relações\nde poder",
    "beneficiados_ia":    "Beneficiados\npela IA",
    "adocao_desigual":    "Adoção\ndesigual",
}

PERSONA_LABELS = {
    "economista":          "Economista",
    "cientista_computacao": "Cient. Comp.",
    "sociologo":           "Sociólogo",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_runs(directory, skip=("summary.json", "ranking.json")):
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if os.path.basename(path) in skip:
            continue
        runs.append(load_json(path))
    return runs

def load_enriched_runs(directory):
    enriched_dir = os.path.join(directory, "enriched")
    return load_runs(enriched_dir, skip=("summary.json",))

def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# Plot 1 — Exp1: Divergência cosseno × temperatura (linha + erro)

def plot_exp1_temperature():
    print("Plot 1: Exp1 temperature...")
    summary = load_json(os.path.join(DATA, "exp1", "summary.json"))
    enriched_summary = load_json(os.path.join(DATA, "exp1", "enriched", "summary.json"))

    # aggregate cosine divergence
    temp_cosine = defaultdict(list)
    for r in summary["results"]:
        temp_cosine[r["temperature"]].append(1.0 - r["cosine_mean"])

    # aggregate sentence cosine divergence from enriched
    temp_sent = defaultdict(list)
    for r in enriched_summary["results"]:
        temp_sent[r["temperature"]].append(r["sentence_cosine_divergence"])

    temps = sorted(temp_cosine.keys())
    cos_means = [np.mean(temp_cosine[t]) for t in temps]
    cos_stds  = [np.std(temp_cosine[t], ddof=0) for t in temps]
    sent_means = [np.mean(temp_sent[t]) for t in temps]
    sent_stds  = [np.std(temp_sent[t], ddof=0) for t in temps]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(temps, cos_means, yerr=cos_stds, marker="o", capsize=4,
                label="Cosseno global", color=COLORS[0], linewidth=2)
    ax.errorbar(temps, sent_means, yerr=sent_stds, marker="s", capsize=4,
                label="Sentence-cosine", color=COLORS[1], linewidth=2, linestyle="--")

    ax.set_xlabel("Temperatura")
    ax.set_ylabel("Divergência média (1 − similaridade)")
    ax.set_title("Exp 1 — Divergência entre personas × Temperatura")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.FixedLocator(temps))
    plt.tight_layout()
    save("exp1_temperature.png")

# Plot 2 — Exp2a: divergência por questão (barras agrupadas: cosseno vs sent)

def plot_exp2a_questions():
    print("Plot 2: Exp2a questions...")
    summary   = load_json(os.path.join(DATA, "exp2a", "summary.json"))
    enriched  = load_json(os.path.join(DATA, "exp2a", "enriched", "summary.json"))

    q_cos  = defaultdict(list)
    q_sent = defaultdict(list)

    for r in summary["results"]:
        q_cos[r["question_key"]].append(1.0 - r["cosine_mean"])
    for r in enriched["results"]:
        q_sent[r["question_key"]].append(r["sentence_cosine_divergence"])

    questions = list(QUESTION_LABELS.keys())
    cos_m  = [np.mean(q_cos[q])  for q in questions]
    cos_e  = [np.std(q_cos[q], ddof=0)  for q in questions]
    sent_m = [np.mean(q_sent[q]) for q in questions]
    sent_e = [np.std(q_sent[q], ddof=0) for q in questions]

    x = np.arange(len(questions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, cos_m, width, yerr=cos_e, capsize=4,
           label="Cosseno global", color=COLORS[0], alpha=0.85)
    ax.bar(x + width/2, sent_m, width, yerr=sent_e, capsize=4,
           label="Sentence-cosine", color=COLORS[1], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([QUESTION_LABELS[q] for q in questions])
    ax.set_ylabel("Divergência média (1 − similaridade)")
    ax.set_title("Exp 2a — Divergência por questão (duas métricas)")
    ax.legend()
    plt.tight_layout()
    save("exp2a_questions.png")

# Plot 3 — Exp2a: scatter cosseno × sentence-cosine (correlação baixa)

def plot_exp2a_scatter():
    print("Plot 3: Exp2a metric scatter...")
    enriched = load_json(os.path.join(DATA, "exp2a", "enriched", "summary.json"))

    cos_vals  = [r["cosine_divergence"] for r in enriched["results"]]
    sent_vals = [r["sentence_cosine_divergence"] for r in enriched["results"]]
    q_keys    = [r["question_key"] for r in enriched["results"]]

    unique_qs = list(QUESTION_LABELS.keys())
    color_map = {q: COLORS[i] for i, q in enumerate(unique_qs)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for q in unique_qs:
        idx = [i for i, k in enumerate(q_keys) if k == q]
        ax.scatter([cos_vals[i] for i in idx],
                   [sent_vals[i] for i in idx],
                   color=color_map[q], label=QUESTION_LABELS[q],
                   s=70, alpha=0.85, zorder=3)

    rho = np.corrcoef(cos_vals, sent_vals)[0, 1]
    # regression line
    m, b = np.polyfit(cos_vals, sent_vals, 1)
    xs = np.linspace(min(cos_vals), max(cos_vals), 100)
    ax.plot(xs, m * xs + b, "k--", linewidth=1, alpha=0.5)

    ax.set_xlabel("Divergência — Cosseno global")
    ax.set_ylabel("Divergência — Sentence-cosine")
    ax.set_title(f"Exp 2a — Correlação entre métricas ($\\rho = {rho:.2f}$)")
    ax.legend(fontsize=9, title="Questão")
    plt.tight_layout()
    save("exp2a_metric_scatter.png")

# Plot 4 — Exp2b: estimativas numéricas por persona e questão

def plot_exp2b_numeric():
    print("Plot 4: Exp2b numeric estimates...")
    runs = load_runs(os.path.join(DATA, "exp2b"))
    # exclude call_centers_anos (bug)
    valid_questions = ["empregos_automatizados_pct", "pib_crescimento_pct", "gap_desenvolvidos_anos"]

    q_persona_vals = defaultdict(lambda: defaultdict(list))
    for run in runs:
        qk = run["question_key"]
        if qk not in valid_questions:
            continue
        for r in run["responses"]:
            q_persona_vals[qk][r["persona"]].append(r["estimate_pct"])

    q_labels = {
        "empregos_automatizados_pct": "Empregos\nautomatizados (%)",
        "pib_crescimento_pct":        "Crescimento\nPIB atrib. IA (%)",
        "gap_desenvolvidos_anos":     "Gap países\ndesenv. (anos)",
    }

    personas = ["economista", "cientista_computacao", "sociologo"]
    fig, axes = plt.subplots(1, len(valid_questions), figsize=(13, 5), sharey=False)

    for ax, qk in zip(axes, valid_questions):
        data_per_persona = [q_persona_vals[qk][p] for p in personas]
        positions = np.arange(len(personas))

        bp = ax.boxplot(data_per_persona, positions=positions, widths=0.5,
                        patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(positions)
        ax.set_xticklabels([PERSONA_LABELS[p] for p in personas], fontsize=9)
        ax.set_title(q_labels[qk], fontsize=10)
        ax.set_ylabel("Estimativa" if ax == axes[0] else "")

    fig.suptitle("Exp 2b — Estimativas numéricas por persona e questão", y=1.02)
    plt.tight_layout()
    save("exp2b_numeric.png")

# Plot 5 — Exp4: heatmap temperatura × questão

def plot_exp4_heatmap():
    print("Plot 5: Exp4 heatmap...")
    summary = load_json(os.path.join(DATA, "exp4", "summary.json"))

    temps = sorted(set(r["temperature"] for r in summary["grid"]))
    questions = list(QUESTION_LABELS.keys())

    matrix = np.zeros((len(temps), len(questions)))
    for r in summary["grid"]:
        ti = temps.index(r["temperature"])
        qi = questions.index(r["question_key"])
        matrix[ti, qi] = r["cosine_divergence_mean"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(matrix, ax=ax,
                xticklabels=[QUESTION_LABELS[q] for q in questions],
                yticklabels=[str(t) for t in temps],
                annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Divergência cosseno"})

    ax.set_xlabel("Questão")
    ax.set_ylabel("Temperatura")
    ax.set_title("Exp 4 — Heatmap: divergência cosseno por temperatura × questão")
    plt.tight_layout()
    save("exp4_heatmap.png")

# Plot 6 — Comparação de métricas: baseline vs exp2a (resumo executivo)

def plot_baseline_vs_exp2a():
    print("Plot 6: Baseline vs Exp2a summary...")
    enriched = load_json(os.path.join(DATA, "exp2a", "enriched", "summary.json"))

    q_sent = defaultdict(list)
    for r in enriched["results"]:
        q_sent[r["question_key"]].append(r["sentence_cosine_divergence"])

    questions = list(QUESTION_LABELS.keys())
    sent_means = [np.mean(q_sent[q]) for q in questions]
    sent_stds  = [np.std(q_sent[q], ddof=0) for q in questions]

    # baseline sentence cosine from exp1 temp=0
    exp1_enriched = load_json(os.path.join(DATA, "exp1", "enriched", "summary.json"))
    baseline_sent = [r["sentence_cosine_divergence"]
                     for r in exp1_enriched["results"] if r["temperature"] == 0.0]
    baseline_mean = np.mean(baseline_sent)

    x = np.arange(len(questions))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, sent_means, yerr=sent_stds, capsize=4,
                  color=COLORS[1], alpha=0.85, label="Questões polêmicas (Exp 2a)")
    ax.axhline(baseline_mean, color="red", linestyle="--", linewidth=1.5,
               label=f"Baseline (questão original): {baseline_mean:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels([QUESTION_LABELS[q] for q in questions])
    ax.set_ylabel("Divergência sentence-cosine (média)")
    ax.set_title("Questões polêmicas vs Baseline — Sentence-cosine divergence")
    ax.legend()
    plt.tight_layout()
    save("baseline_vs_exp2a_sentcosine.png")

def main():
    print(f"Gerando gráficos em {OUT}...")
    plot_exp1_temperature()
    plot_exp2a_questions()
    plot_exp2a_scatter()
    plot_exp2b_numeric()
    plot_exp4_heatmap()
    plot_baseline_vs_exp2a()
    print("Concluído.")

if __name__ == "__main__":
    main()
