"""Genera viewer/juego.html: simulación en vivo en el navegador con los pesos reales entrenados.

El juego es una réplica en JavaScript de neuroeco/env.py. Incrusta:
  * las redes de la semilla de entrenamiento elegida (base 16-16 y reducida 4-4) en cada punto de
    control (sin entrenar, solo supervisado, generaciones 0, 10, ..., 60);
  * los escenarios de config/scenarios.json y las constantes de neuroeco/config.py.

Uso:  python scripts/make_game.py            (semilla de entrenamiento 11)
      python scripts/make_game.py --train_seed 22
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from neuroeco.config import ROOT, WORLD, PREY, PRED, load_scenarios  # noqa: E402


def r(a):
    return [round(float(x), 6) for x in np.asarray(a).ravel()]


def models(train_seed):
    out = {}
    for cfg in ("base", "small"):
        d = np.load(ROOT / "models" / f"{cfg}_seed{train_seed}.npz")
        q = {"random": r(d["prey_random"])}
        p = {"random": r(d["pred_random"]), "sup": r(d["pred_pretrained"])}
        for g, gq, gp in zip(d["ck_gens"], d["ck_prey"], d["ck_pred"]):
            q[f"g{int(g)}"] = r(gq)
            p[f"g{int(g)}"] = r(gp)
        out[cfg] = dict(hidden=[int(h) for h in d["hidden"]],
                        prey=dict(order=list(q), levels=q), pred=dict(order=list(p), levels=p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_seed", type=int, default=11)
    a = ap.parse_args()
    tpl = (ROOT / "viewer" / "game_template.html").read_text(encoding="utf-8")
    html = (tpl.replace("/*__MODELS__*/null", json.dumps(models(a.train_seed), separators=(",", ":")))
               .replace("/*__SCEN__*/null", json.dumps(load_scenarios(), ensure_ascii=False))
               .replace("/*__PHYS__*/null", json.dumps(dict(world=WORLD, prey=PREY, pred=PRED))))
    out = ROOT / "viewer" / "juego.html"
    out.write_text(html, encoding="utf-8")
    print("juego generado:", out, f"({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
