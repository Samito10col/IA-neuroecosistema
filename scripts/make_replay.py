"""Graba un episodio y genera un visor HTML autocontenido (abrir con doble clic).

Ejemplos:
    python scripts/make_replay.py                                   # red vs red, escenario a, semilla 5000
    python scripts/make_replay.py --scenario c_presion --seed 5003
    python scripts/make_replay.py --prey rule --pred nn             # presa por reglas vs depredador neuronal
    python scripts/make_replay.py --scenario a_abundancia --n_pred 4 --name a_con_4_depredadores
    python scripts/make_replay.py --model models/small_seed11.npz
    python scripts/make_replay.py --demo                            # genera los 4 visores de la sustentación

Parámetros del escenario que se pueden alterar en vivo: --n_pred --n_food --regrow --n_obst
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from neuroeco.config import ROOT, WORLD, PREY, PRED, load_scenarios  # noqa: E402
from neuroeco.env import run_episode  # noqa: E402
from neuroeco.controllers import (NeuralController, RandomController, RulePreyController,  # noqa: E402
                                  RulePredController)
from neuroeco.evaluate import load_model  # noqa: E402

PREY_IN = ["dir. planta x", "dir. planta y", "dist. planta", "dir. depred. x", "dir. depred. y",
           "dist. depred.", "energía", "posición x", "posición y", "dir. obstáculo x",
           "dir. obstáculo y", "dist. obstáculo", "exploración cos", "exploración sin"]
PRED_IN = ["dir. presa x", "dir. presa y", "dist. presa", "energía", "digiriendo", "dir. otro depr. x",
           "dir. otro depr. y", "posición x", "posición y", "dir. obstáculo x", "dir. obstáculo y",
           "dist. obstáculo", "exploración cos", "exploración sin"]


def ctrl(kind, species, M):
    if kind == "nn":
        hid = [int(h) for h in M["hidden"]]
        return NeuralController(M[f"{species}_final"][None], hid)
    if kind == "random":
        return RandomController()
    return RulePreyController() if species == "prey" else RulePredController()


def record(scenario, seed, prey_kind, pred_kind, model_path, title):
    M = load_model(model_path) if "nn" in (prey_kind, pred_kind) else None
    q, p = ctrl(prey_kind, "prey", M), ctrl(pred_kind, "pred", M)
    frames = []
    static = {}

    def r2(a):
        return np.round(np.asarray(a, float), 2).tolist()

    def rec(env, oq, op, aq, ap, qc, pc):
        if not static:
            static.update(food=r2(env.food_pos[0][env.food_active[0]]),
                          obst=[[*r2(c), round(float(r), 2)] for c, r in zip(env.obs_pos[0], env.obs_r[0]) if r > 0],
                          n_food=int(env.food_active[0].sum()), n_pred=int(env.pred_active[0].sum()))
        npred = static["n_pred"]
        f = dict(
            t=env.t,
            q=r2(env.prey_pos[0]), qa=env.prey_alive[0].astype(int).tolist(), qe=r2(env.prey_E[0]),
            p=r2(env.pred_pos[0][:npred]), pa=env.pred_alive[0][:npred].astype(int).tolist(),
            pe=r2(env.pred_E[0][:npred]), pc=env.pred_cool[0][:npred].astype(int).tolist(),
            fa=env.food_avail[0][env.food_active[0]].astype(int).tolist(),
            qin=r2(oq[0]), qout=r2(aq[0]),
            pin=r2(op[0][:npred]), pout=r2(ap[0][:npred]),
            caps=int(env.pred_caps[0].sum()), eaten=int(env.prey_food[0].sum()),
        )
        if getattr(qc, "last_hidden", None) is not None:
            f["qh"] = [r2(h[0]) for h in qc.last_hidden]
        if getattr(pc, "last_hidden", None) is not None:
            f["ph"] = [r2(h[0][:npred]) for h in pc.last_hidden]
        frames.append(f)

    m = run_episode([scenario], [seed], q, p, recorder=rec)
    summary = {k: round(float(v[0]), 3) for k, v in m.items()}
    return dict(title=title, scenario=scenario, seed=seed, prey_ctrl=prey_kind, pred_ctrl=pred_kind,
                model=str(Path(model_path).name) if M is not None else "-",
                world=WORLD, prey_params=PREY, pred_params=PRED,
                prey_inputs=PREY_IN, pred_inputs=PRED_IN, static=static, frames=frames, summary=summary)


def write_html(data, out):
    tpl = (ROOT / "viewer" / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    out.write_text(html, encoding="utf-8")
    print("visor generado:", out, f"({out.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="a_abundancia")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--prey", default="nn", choices=["nn", "rule", "random"])
    ap.add_argument("--pred", default="nn", choices=["nn", "rule", "random"])
    ap.add_argument("--model", default=str(ROOT / "models" / "base_seed11.npz"))
    ap.add_argument("--n_pred", type=int); ap.add_argument("--n_food", type=int)
    ap.add_argument("--regrow", type=float); ap.add_argument("--n_obst", type=int)
    ap.add_argument("--name", default=None)
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    scen = load_scenarios()
    if a.demo:
        jobs = [("a_abundancia", "nn", "nn", "demo_a_red_vs_red"),
                ("b_escasez", "nn", "nn", "demo_b_red_vs_red"),
                ("c_presion", "nn", "nn", "demo_c_red_vs_red"),
                ("c_presion", "rule", "rule", "demo_c_reglas_vs_reglas")]
        for sc, qk, pk, name in jobs:
            d = record(dict(scen[sc], nombre=sc), a.seed, qk, pk, a.model, f"{name} · semilla {a.seed}")
            write_html(d, ROOT / "viewer" / f"{name}.html")
        return
    sc = dict(scen[a.scenario], nombre=a.scenario)
    for k in ("n_pred", "n_food", "regrow", "n_obst"):
        if getattr(a, k) is not None:
            sc[k] = getattr(a, k)
    sc["n_pred"] = min(sc["n_pred"], WORLD["max_pred"])
    sc["n_food"] = min(sc["n_food"], WORLD["max_food"])
    sc["n_obst"] = min(sc["n_obst"], WORLD["max_obst"])
    name = a.name or f"replay_{a.scenario}_{a.prey}_vs_{a.pred}_s{a.seed}"
    d = record(sc, a.seed, a.prey, a.pred, a.model, f"{name}")
    write_html(d, ROOT / "viewer" / f"{name}.html")


if __name__ == "__main__":
    main()
