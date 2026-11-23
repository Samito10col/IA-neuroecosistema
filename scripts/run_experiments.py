"""Corre todos los experimentos de evaluación y guarda los datos crudos en results/data.

Experimentos:
  E1 antes vs después del entrenamiento (red aleatoria, mejor de la gen. 0, élite final)
     + curva de aprendizaje con los puntos de control cada 10 generaciones.
  E2 arquitectura base (16-16) vs reducida (4-4).
  E3 red neuronal vs referencia por reglas vs referencia aleatoria.
  E4 ecosistema completo: red vs red y reglas vs reglas (descriptivo).
En E1-E3 el oponente es SIEMPRE el agente por reglas, para que lo único que
cambie sea el controlador de la especie evaluada.

Uso:  python scripts/run_experiments.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from neuroeco.config import ROOT, load_training  # noqa: E402
from neuroeco.evaluate import evaluate, load_model, load_experiments  # noqa: E402

OUT = ROOT / "results" / "data"
RULE = ("rule", None, None)
RAND = ("random", None, None)


def nn(g, hid):
    return ("nn", g, hid)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ex = load_experiments()
    seeds, cseeds = ex["test_seeds"], ex["curve_seeds"]
    rows_e1, rows_curve, rows_e2, rows_e3, rows_e4 = [], [], [], [], []

    for cfg_name in ("base", "small"):
        for ts in load_training(cfg_name)["train_seeds"]:
            path = ROOT / "models" / f"{cfg_name}_seed{ts}.npz"
            if not path.exists():
                print("  (falta modelo)", path.name)
                continue
            M = load_model(path)
            hid = [int(h) for h in M["hidden"]]
            print(f"Evaluando {path.name}")
            for sp in ("prey", "pred"):
                def run(spec, tags, sd=seeds):
                    if sp == "prey":
                        return evaluate(spec, RULE, sd, tags=tags)
                    return evaluate(RULE, spec, sd, tags=tags)
                base_tags = dict(species=sp, config=cfg_name, train_seed=ts)
                # E2 / E3 / E1: élite final
                df_final = run(nn(M[f"{sp}_final"], hid), dict(base_tags, controller="red_final"))
                rows_e2.append(df_final)
                if cfg_name == "base":
                    rows_e3.append(df_final)
                    rows_e1.append(df_final)
                    rows_e1.append(run(nn(M[f"{sp}_gen0"], hid), dict(base_tags, controller="red_gen0")))
                    rows_e1.append(run(nn(M[f"{sp}_random"], hid), dict(base_tags, controller="red_aleatoria_init")))
                    if sp == "pred":
                        rows_e1.append(run(nn(M["pred_pretrained"], hid), dict(base_tags, controller="red_supervisada")))
                # curva de aprendizaje
                for g, genome in zip(M["ck_gens"], M[f"ck_{sp}"]):
                    rows_curve.append(run(nn(genome, hid), dict(base_tags, controller="checkpoint", gen=int(g)), cseeds))
            if cfg_name == "base":
                rows_e4.append(evaluate(nn(M["prey_final"], hid), nn(M["pred_final"], hid), seeds,
                                        tags=dict(config=cfg_name, train_seed=ts, matchup="red_vs_red")))

    # referencias (no dependen del entrenamiento)
    for sp in ("prey", "pred"):
        for name, spec in (("reglas", RULE), ("aleatorio", RAND)):
            tags = dict(species=sp, config="referencia", train_seed=-1, controller=name)
            df = evaluate(spec, RULE, seeds, tags=tags) if sp == "prey" else evaluate(RULE, spec, seeds, tags=tags)
            rows_e3.append(df)
    rows_e4.append(evaluate(RULE, RULE, seeds, tags=dict(config="referencia", train_seed=-1, matchup="reglas_vs_reglas")))

    for name, rows in (("e1_antes_despues", rows_e1), ("e1_curva_aprendizaje", rows_curve),
                       ("e2_arquitecturas", rows_e2), ("e3_referencias", rows_e3),
                       ("e4_ecosistema", rows_e4)):
        if rows:
            pd.concat(rows, ignore_index=True).to_csv(OUT / f"{name}.csv", index=False)
            print("guardado", name)


if __name__ == "__main__":
    main()
