"""Evaluación de controladores en los escenarios con semillas registradas."""
import json
import numpy as np
import pandas as pd

from .config import ROOT, load_scenarios
from .env import run_episode
from .controllers import (NeuralController, RandomController, RulePreyController,
                          RulePredController)


def load_experiments():
    with open(ROOT / "config" / "experiments.json", encoding="utf-8") as f:
        return json.load(f)


def load_model(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def make_ctrl(species, kind, genome=None, hidden=None):
    """kind: 'nn' | 'rule' | 'random'"""
    if kind == "nn":
        return ("nn", genome, list(hidden))
    return (kind, None, None)


def _build(spec, B):
    kind, genome, hidden = spec
    if kind == "nn":
        return NeuralController(np.repeat(genome[None], B, 0), hidden)
    if kind == "random":
        return RandomController()
    return None


def evaluate(prey_spec, pred_spec, seeds, scenarios=None, tags=None):
    """Corre todos los escenarios x semillas en un solo lote. Devuelve DataFrame (una fila por episodio)."""
    scen = load_scenarios()
    names = scenarios or list(scen)
    worlds_sc, worlds_seed, worlds_name = [], [], []
    for n in names:
        for s in seeds:
            worlds_sc.append(scen[n]); worlds_seed.append(s); worlds_name.append(n)
    B = len(worlds_sc)
    q = _build(prey_spec, B) or RulePreyController()
    p = _build(pred_spec, B) or RulePredController()
    m = run_episode(worlds_sc, worlds_seed, q, p)
    df = pd.DataFrame({k: v for k, v in m.items()})
    df.insert(0, "scenario", worlds_name)
    df.insert(1, "seed", worlds_seed)
    for k, v in (tags or {}).items():
        df.insert(0, k, v)
    return df
