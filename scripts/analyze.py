"""Genera tablas, pruebas estadísticas y gráficas a partir de results/data.

Uso:  python scripts/analyze.py
Salidas: results/figures/*.png, results/tables/*.csv, results/RESUMEN_RESULTADOS.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from neuroeco.config import ROOT, load_training  # noqa: E402

DATA = ROOT / "results" / "data"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
SC = ["a_abundancia", "b_escasez", "c_presion"]
SC_LBL = {"a_abundancia": "(a) abundancia", "b_escasez": "(b) escasez", "c_presion": "(c) presión+obst."}
METRIC_LBL = {
    "prey_lifetime": "Vida media de presas (fracción del episodio)",
    "prey_survival": "Supervivencia final de presas",
    "food_per_prey": "Alimento consumido por presa",
    "prey_final_energy": "Energía final de presas vivas",
    "captures": "Capturas totales por episodio",
    "captures_per_pred": "Capturas por depredador",
    "time_to_first_capture": "Pasos hasta la primera captura",
    "pred_final_energy": "Energía final de depredadores",
    "pred_survival": "Supervivencia de depredadores",
}
PREY_M = ["prey_lifetime", "prey_survival", "food_per_prey", "prey_final_energy"]
PRED_M = ["captures_per_pred", "time_to_first_capture", "pred_final_energy", "pred_survival"]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

md = []


def fmt(x):
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"


def summary(df, by, metrics):
    g = df.groupby(by)[metrics]
    out = g.mean().round(4).add_suffix("_media").join(g.std().round(4).add_suffix("_desv"))
    out["n"] = g.size()
    return out


def mw(a, b):
    try:
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except ValueError:
        return float("nan")


def md_table(df, ctrl_col, order, metrics):
    lines = ["| Escenario | Controlador | " + " | ".join(METRIC_LBL[m].split(" (")[0] for m in metrics) + " | n |",
             "|---|---|" + "---|" * len(metrics) + "---|"]
    for sc in SC:
        for c in order:
            d = df[(df.scenario == sc) & (df[ctrl_col] == c)]
            if d.empty:
                continue
            cells = [f"{fmt(d[m].mean())} ± {fmt(d[m].std())}" for m in metrics]
            lines.append(f"| {SC_LBL[sc]} | {c} | " + " | ".join(cells) + f" | {len(d)} |")
    return "\n".join(lines)


def comparisons(df, ctrl_col, pairs, metric, title):
    lines = [f"\n**Pruebas de Mann-Whitney (bilateral) — {METRIC_LBL[metric]}**\n",
             "| Escenario | Comparación | Media A | Media B | Diferencia | p |", "|---|---|---|---|---|---|"]
    rows = []
    for sc in SC:
        for a, b in pairs:
            A = df[(df.scenario == sc) & (df[ctrl_col] == a)][metric]
            B = df[(df.scenario == sc) & (df[ctrl_col] == b)][metric]
            if A.empty or B.empty:
                continue
            p = mw(A, B)
            rows.append(dict(experimento=title, escenario=sc, A=a, B=b, metrica=metric,
                             media_A=A.mean(), media_B=B.mean(), p=p))
            lines.append(f"| {SC_LBL[sc]} | {a} vs {b} | {fmt(A.mean())} | {fmt(B.mean())} | "
                         f"{fmt(A.mean() - B.mean())} | {p:.3g} |")
    md.append("\n".join(lines))
    return rows


def bars(df, ctrl_col, order, metrics, fname, title, labels=None):
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    axes = np.atleast_1d(axes)
    w = 0.8 / len(order)
    for ax, m in zip(axes, metrics):
        for i, c in enumerate(order):
            vals = [df[(df.scenario == s) & (df[ctrl_col] == c)][m] for s in SC]
            mu = [v.mean() for v in vals]
            sd = [v.std() for v in vals]
            ax.bar(np.arange(3) + (i - (len(order) - 1) / 2) * w, mu, w, yerr=sd, capsize=3,
                   color=COLORS[i % len(COLORS)], label=(labels or {}).get(c, c))
        ax.set_xticks(range(3)); ax.set_xticklabels([SC_LBL[s] for s in SC], fontsize=8)
        ax.set_title(METRIC_LBL[m], fontsize=9)
        ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title + "  (barras = media, líneas = desviación estándar)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=130)
    plt.close(fig)


def training_figures():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 3.6))
    for ci, cfg in enumerate(("base", "small")):
        for si, s in enumerate(load_training(cfg)["train_seeds"]):
            p = ROOT / "models" / f"{cfg}_seed{s}_history.json"
            if not p.exists():
                continue
            h = pd.DataFrame(json.load(open(p))["history"])
            ls = "-" if cfg == "base" else "--"
            col = COLORS[si]
            for j, sp in enumerate(("prey", "pred")):
                axes[j, ci].plot(h.gen, h[f"{sp}_fit_max"], ls, color=col, label=f"máx semilla {s}")
                axes[j, ci].plot(h.gen, h[f"{sp}_fit_mean"], ":", color=col, alpha=.8, label=f"media semilla {s}")
                ax2[j].plot(h.gen, h[f"{sp}_dist_from_init"], ls, color=col, label=f"{cfg} s{s}")
    for j, sp in enumerate(("Presas", "Depredadores")):
        for ci, cfg in enumerate(("base 16-16", "reducida 4-4")):
            axes[j, ci].set_title(f"{sp} — {cfg}: aptitud por generación", fontsize=9)
            axes[j, ci].set_xlabel("generación"); axes[j, ci].grid(alpha=.3)
        ax2[j].set_title(f"{sp}: distancia L2 entre pesos de la élite y la élite de la gen. 0", fontsize=9)
        ax2[j].set_xlabel("generación"); ax2[j].grid(alpha=.3)
    axes[0, 0].legend(fontsize=6); ax2[0].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "fig1_aptitud_entrenamiento.png", dpi=130); plt.close(fig)
    fig2.tight_layout(); fig2.savefig(FIG / "fig2_cambio_de_pesos.png", dpi=130); plt.close(fig2)

    # pérdida supervisada
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for si, s in enumerate(load_training("base")["train_seeds"]):
        p = ROOT / "models" / f"base_seed{s}.npz"
        if p.exists():
            L = np.load(p)["sup_losses"]
            ax.plot(L, color=COLORS[si], label=f"semilla {s}")
    ax.set_title("Pre-entrenamiento supervisado del depredador (base)", fontsize=9)
    ax.set_xlabel("época"); ax.set_ylabel("MSE"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig3_perdida_supervisada.png", dpi=130); plt.close(fig)


def curve_figure(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for j, (sp, m) in enumerate((("prey", "prey_lifetime"), ("pred", "captures_per_pred"))):
        d = df[(df.species == sp) & (df.config == "base")]
        for k, sc in enumerate(SC):
            g = d[d.scenario == sc].groupby("gen")[m]
            mu, sd = g.mean(), g.std()
            axes[j].plot(mu.index, mu.values, "o-", color=COLORS[k], label=SC_LBL[sc])
            axes[j].fill_between(mu.index, mu - sd, mu + sd, color=COLORS[k], alpha=.15)
        axes[j].set_title(("Presas: " if sp == "prey" else "Depredadores: ") + METRIC_LBL[m] +
                          "\n(élite en cada punto de control vs. oponente por reglas)", fontsize=9)
        axes[j].set_xlabel("generación"); axes[j].grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig4_curva_aprendizaje.png", dpi=130); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    all_tests = []
    training_figures()

    md.append("# Resumen de resultados (generado automáticamente por scripts/analyze.py)\n")
    md.append("Valores: media ± desviación estándar sobre episodios. Semillas de prueba 5000-5029 "
              "(30 por escenario). Para las redes, n = 3 semillas de entrenamiento × 30 semillas de prueba = 90. "
              "En E1-E3 el oponente es siempre el agente por reglas.\n")

    e1 = pd.read_csv(DATA / "e1_antes_despues.csv")
    for sp, metrics, order in (("prey", PREY_M, ["red_aleatoria_init", "red_gen0", "red_final"]),
                               ("pred", PRED_M, ["red_aleatoria_init", "red_supervisada", "red_gen0", "red_final"])):
        d = e1[e1.species == sp]
        summary(d, ["scenario", "controller"], metrics).to_csv(TAB / f"e1_{sp}.csv")
        md.append(f"\n## E1 — Antes vs. después del entrenamiento ({'presas' if sp == 'prey' else 'depredadores'})\n")
        md.append(md_table(d, "controller", order, metrics))
        key = metrics[0]
        pairs = [("red_final", "red_gen0"), ("red_final", "red_aleatoria_init")]
        if sp == "pred":
            pairs += [("red_final", "red_supervisada"), ("red_supervisada", "red_aleatoria_init")]
        all_tests += comparisons(d, "controller", pairs, key, f"E1_{sp}")
        bars(d, "controller", order, metrics[:3], f"fig5_e1_{sp}.png",
             f"E1 {'Presas' if sp == 'prey' else 'Depredadores'}: antes vs. después")

    cv = pd.read_csv(DATA / "e1_curva_aprendizaje.csv")
    curve_figure(cv)
    summary(cv, ["species", "config", "scenario", "gen"], ["prey_lifetime", "captures_per_pred"]).to_csv(TAB / "e1_curva.csv")

    e2 = pd.read_csv(DATA / "e2_arquitecturas.csv")
    for sp, metrics in (("prey", PREY_M), ("pred", PRED_M)):
        d = e2[e2.species == sp]
        summary(d, ["scenario", "config"], metrics).to_csv(TAB / f"e2_{sp}.csv")
        md.append(f"\n## E2 — Arquitectura base 16-16 vs. reducida 4-4 ({'presas' if sp == 'prey' else 'depredadores'})\n")
        md.append(md_table(d, "config", ["base", "small"], metrics))
        all_tests += comparisons(d, "config", [("base", "small")], metrics[0], f"E2_{sp}")
        # variabilidad entre semillas de entrenamiento
        v = d.groupby(["config", "train_seed", "scenario"])[metrics[0]].mean().unstack("scenario").round(3)
        md.append(f"\nMedia de {METRIC_LBL[metrics[0]].lower()} por semilla de entrenamiento:\n\n" + v.to_markdown())
        bars(d, "config", ["base", "small"], metrics[:3], f"fig6_e2_{sp}.png",
             f"E2 {'Presas' if sp == 'prey' else 'Depredadores'}: base 16-16 vs. reducida 4-4",
             labels={"base": "base 16-16", "small": "reducida 4-4"})

    e3 = pd.read_csv(DATA / "e3_referencias.csv")
    e3["ctrl"] = e3.controller.replace({"red_final": "red neuronal"})
    for sp, metrics in (("prey", PREY_M), ("pred", PRED_M)):
        d = e3[e3.species == sp]
        summary(d, ["scenario", "ctrl"], metrics).to_csv(TAB / f"e3_{sp}.csv")
        md.append(f"\n## E3 — Red neuronal vs. referencias ({'presas' if sp == 'prey' else 'depredadores'})\n")
        md.append(md_table(d, "ctrl", ["red neuronal", "reglas", "aleatorio"], metrics))
        for m in metrics[:2]:
            all_tests += comparisons(d, "ctrl", [("red neuronal", "reglas"), ("red neuronal", "aleatorio")], m, f"E3_{sp}")
        bars(d, "ctrl", ["red neuronal", "reglas", "aleatorio"], metrics[:3], f"fig7_e3_{sp}.png",
             f"E3 {'Presas' if sp == 'prey' else 'Depredadores'}: red vs. reglas vs. aleatorio")

    e4 = pd.read_csv(DATA / "e4_ecosistema.csv")
    em = ["prey_lifetime", "prey_survival", "captures", "food_per_prey", "pred_final_energy", "time_to_first_capture"]
    summary(e4, ["scenario", "matchup"], em).to_csv(TAB / "e4.csv")
    md.append("\n## E4 — Ecosistema completo (descriptivo)\n")
    md.append(md_table(e4, "matchup", ["red_vs_red", "reglas_vs_reglas"], em))
    bars(e4, "matchup", ["red_vs_red", "reglas_vs_reglas"], ["prey_survival", "captures", "food_per_prey"],
         "fig8_e4_ecosistema.png", "E4 Ecosistema completo: red vs red y reglas vs reglas")

    pd.DataFrame(all_tests).to_csv(TAB / "pruebas_estadisticas.csv", index=False)
    (ROOT / "results" / "RESUMEN_RESULTADOS.md").write_text("\n".join(md), encoding="utf-8")
    print("Listo: figuras en results/figures, tablas en results/tables, resumen en results/RESUMEN_RESULTADOS.md")


if __name__ == "__main__":
    main()
