"""Entrena presas y depredadores por neuroevolución.

Uso:
    python scripts/train.py --config base --seed 11
    python scripts/train.py --config small --seed 22 --generations 20   (prueba rápida)
    python scripts/train.py --all          (todas las configuraciones y semillas de config/training.json)
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuroeco.config import ROOT, load_training  # noqa: E402
from neuroeco.evolution import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    jobs = []
    if a.all:
        for name in ("base", "small"):
            for s in load_training(name)["train_seeds"]:
                jobs.append((name, s))
    else:
        cfg = load_training(a.config)
        jobs = [(a.config, a.seed if a.seed is not None else cfg["train_seeds"][0])]
    for name, seed in jobs:
        cfg = load_training(name)
        if a.generations is not None:
            cfg["generations"] = a.generations
        out = ROOT / "models" / f"{name}_seed{seed}.npz"
        print(f"== Entrenando '{name}' {cfg['hidden']} semilla {seed} -> {out.name}")
        train(cfg, seed, out)


if __name__ == "__main__":
    main()
