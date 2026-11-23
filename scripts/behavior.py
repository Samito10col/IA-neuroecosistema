"""Mide QUÉ hacen los agentes (no solo cuánto rinden): búsqueda, persecución y evasión.

Para cada paso y cada agente vivo se compara su acción (vector de movimiento) con lo que percibe:
  * búsqueda (presa):   coseno entre la acción y la dirección a la planta más cercana,
                        cuando ve una planta y NO ve depredadores.
  * evasión (presa):    coseno entre la acción y la dirección OPUESTA al depredador, cuando lo ve.
  * persecución (depr): coseno entre la acción y la dirección a la presa, cuando la ve y no digiere.
  * rincones/paredes:   fracción del tiempo a menos de 5 unidades de una pared (posible conducta indebida).
  * velocidad media:    norma de la acción (0 a 1).
Coseno = 1: va exactamente hacia allá; 0: indiferente; -1: hace lo contrario.

Uso:  python scripts/behavior.py      -> results/tables/comportamiento.csv y fig9
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from neuroeco.config import ROOT, load_scenarios, load_training  # noqa: E402
from neuroeco.env import run_episode  # noqa: E402
from neuroeco.controllers import (NeuralController, RandomController, RulePreyController,  # noqa: E402
                                  RulePredController)
from neuroeco.evaluate import load_model, load_experiments  # noqa: E402

SC = ["a_abundancia", "b_escasez", "c_presion"]


def cos(a, d):
    na = np.linalg.norm(a, axis=-1); nd = np.linalg.norm(d, axis=-1)
    return (a * d).sum(-1) / np.maximum(na * nd, 1e-9)


def measure(prey_ctrl, pred_ctrl, seeds, species):
    scen = load_scenarios()
    sc_list, names = [], []
    for n in SC:
        for s in seeds:
            sc_list.append(scen[n]); names.append(n)
    B = len(sc_list)
    acc = {k: np.zeros(B) for k in ("seek_s", "seek_n", "flee_s", "flee_n", "chase_s", "chase_n",
                                     "wall_s", "wall_n", "speed_s")}

    def rec(env, oq, op, aq, ap, qc, pc):
        W = env.W
        if species == "prey":
            al = env.prey_alive
            a = aq / np.maximum(np.linalg.norm(aq, axis=-1, keepdims=True), 1.0)
            seek = al & (oq[..., 2] < 1) & (oq[..., 5] >= 1)
            flee = al & (oq[..., 5] < 1)
            c1, c2 = cos(a, oq[..., 0:2]), cos(a, -oq[..., 3:5])
            acc["seek_s"] += (c1 * seek).sum(1); acc["seek_n"] += seek.sum(1)
            acc["flee_s"] += (c2 * flee).sum(1); acc["flee_n"] += flee.sum(1)
            pos = env.prey_pos
        else:
            al = env.pred_alive & env.pred_active
            a = ap / np.maximum(np.linalg.norm(ap, axis=-1, keepdims=True), 1.0)
            ch = al & (op[..., 2] < 1) & (op[..., 4] < 0.5)
            acc["chase_s"] += (cos(a, op[..., 0:2]) * ch).sum(1); acc["chase_n"] += ch.sum(1)
            pos = env.pred_pos
        near = np.minimum(pos, W - pos).min(-1) < 5
        acc["wall_s"] += (near & al).sum(1); acc["wall_n"] += al.sum(1)
        acc["speed_s"] += (np.minimum(np.linalg.norm(a, axis=-1), 1) * al).sum(1)

    run_episode(sc_list, seeds * 3, prey_ctrl, pred_ctrl, recorder=rec)
    out = pd.DataFrame({"scenario": names})
    n = np.maximum
    if species == "prey":
        out["busqueda_cos"] = acc["seek_s"] / n(acc["seek_n"], 1)
        out["evasion_cos"] = acc["flee_s"] / n(acc["flee_n"], 1)
    else:
        out["persecucion_cos"] = acc["chase_s"] / n(acc["chase_n"], 1)
    out["frac_junto_pared"] = acc["wall_s"] / n(acc["wall_n"], 1)
    out["velocidad_media"] = acc["speed_s"] / n(acc["wall_n"], 1)
    return out


def main():
    seeds = load_experiments()["test_seeds"][:10]
    rows = []
    B = len(seeds) * 3
    for ts in load_training("base")["train_seeds"]:
        p = ROOT / "models" / f"base_seed{ts}.npz"
        if not p.exists():
            continue
        M = load_model(p); hid = [int(h) for h in M["hidden"]]
        for lbl, key in (("red_final", "final"), ("red_gen0", "gen0")):
            d = measure(NeuralController(np.repeat(M[f"prey_{key}"][None], B, 0), hid), RulePredController(), seeds, "prey")
            d["species"], d["controller"], d["train_seed"] = "prey", lbl, ts; rows.append(d)
        for lbl, key in (("red_final", "final"), ("red_supervisada", "pretrained"), ("red_aleatoria_init", "random")):
            d = measure(RulePreyController(), NeuralController(np.repeat(M[f"pred_{key}"][None], B, 0), hid), seeds, "pred")
            d["species"], d["controller"], d["train_seed"] = "pred", lbl, ts; rows.append(d)
    for lbl, c in (("reglas", RulePreyController), ("aleatorio", RandomController)):
        d = measure(c(), RulePredController(), seeds, "prey"); d["species"], d["controller"], d["train_seed"] = "prey", lbl, -1; rows.append(d)
    for lbl, c in (("reglas", RulePredController), ("aleatorio", RandomController)):
        d = measure(RulePreyController(), c(), seeds, "pred"); d["species"], d["controller"], d["train_seed"] = "pred", lbl, -1; rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results" / "data" / "comportamiento_episodios.csv", index=False)
    summ = df.groupby(["species", "controller", "scenario"]).mean(numeric_only=True).drop(columns="train_seed").round(3)
    summ.to_csv(ROOT / "results" / "tables" / "comportamiento.csv")
    print(summ.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    specs = [("prey", "busqueda_cos", "Presas: búsqueda de alimento (coseno)", ["red_gen0", "red_final", "reglas", "aleatorio"]),
             ("prey", "evasion_cos", "Presas: evasión del depredador (coseno)", ["red_gen0", "red_final", "reglas", "aleatorio"]),
             ("pred", "persecucion_cos", "Depredadores: persecución (coseno)", ["red_aleatoria_init", "red_supervisada", "red_final", "reglas", "aleatorio"])]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for ax, (sp, m, t, order) in zip(axes, specs):
        d = df[df.species == sp]
        w = 0.8 / len(order)
        for i, c in enumerate(order):
            mu = [d[(d.controller == c) & (d.scenario == s)][m].mean() for s in SC]
            ax.bar(np.arange(3) + (i - (len(order) - 1) / 2) * w, mu, w, color=colors[i], label=c)
        ax.set_xticks(range(3)); ax.set_xticklabels(["(a)", "(b)", "(c)"]); ax.set_title(t, fontsize=9)
        ax.axhline(0, color="k", lw=.6); ax.set_ylim(-0.3, 1.05); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "figures" / "fig9_comportamiento.png", dpi=130)


if __name__ == "__main__":
    main()
