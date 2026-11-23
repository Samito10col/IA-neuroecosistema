"""Entrenamiento por neuroevolución (algoritmo genético) con coevolución.

Se mantienen dos poblaciones de genomas (presas y depredadores). En cada
generación:
  1. Cada genoma de presa se evalúa en 6 episodios: los 3 escenarios x 2 tipos
     de oponente (depredador por reglas y depredador élite de la generación
     anterior). Lo mismo para los depredadores. Todos los genomas de una
     generación usan las MISMAS semillas (números aleatorios comunes), así la
     diferencia de aptitud se debe a la red y no a la suerte del mapa.
  2. Se calcula la aptitud promedio y se forma la siguiente generación:
     élite sin cambios + hijos por torneo, cruce uniforme y mutación gaussiana.
Los pesos cambian SOLO por este procedimiento (no hay gradientes).
"""
import json
import time
import numpy as np

from .config import load_scenarios, N_IN, N_OUT
from .nn import MLPSpec
from .env import run_episode
from .supervised import collect_rule_examples, train_supervised
from .controllers import (NeuralController, RulePreyController, RulePredController,
                          MixedController)

SCEN_ORDER = ["a_abundancia", "b_escasez", "c_presion"]


def fitness_prey(m):
    """Vida media normalizada + pequeño premio por comer."""
    return m["prey_lifetime"] + 0.02 * m["food_per_prey"]


def fitness_pred(m):
    """Capturas por depredador + medio punto por la energía final + término de acecho.

    El acecho (cercanía media a la presa más cercana, en [0, 1]) es un moldeado de
    la recompensa: sin él, casi ninguna red aleatoria captura y la selección no
    tiene gradiente (ver bitácora, decisión D4)."""
    return m["captures_per_pred"] + 0.5 * m["pred_final_energy"] + 1.0 * m["pred_pursuit"]


def _next_generation(rng, pop, fit, cfg):
    P, G = pop.shape
    order = np.argsort(-fit)
    new = [pop[i].copy() for i in order[:cfg["elite"]]]
    while len(new) < P:
        def pick():
            c = rng.choice(P, cfg["tournament"], replace=False)
            return pop[c[np.argmax(fit[c])]]
        p1 = pick()
        child = p1.copy()
        if rng.random() < cfg["crossover_rate"]:
            p2 = pick()
            mask = rng.random(G) < 0.5
            child[mask] = p2[mask]
        mut = rng.random(G) < cfg["mutation_rate"]
        child[mut] += rng.normal(0, cfg["mutation_sigma"], mut.sum())
        new.append(child)
    return np.array(new)


def _evaluate(species, pop, opp_elite, hidden_self, hidden_opp, scen, seeds_gen):
    """Evalúa una población completa en paralelo. Devuelve aptitud (P,) y métricas."""
    P = pop.shape[0]
    worlds_g, worlds_sc, worlds_seed, worlds_nn = [], [], [], []
    for i in range(P):
        for k, name in enumerate(SCEN_ORDER):
            for use_nn in (False, True):
                worlds_g.append(i)
                worlds_sc.append(scen[name])
                worlds_seed.append(seeds_gen[k])
                worlds_nn.append(use_nn)
    worlds_g = np.array(worlds_g)
    B = len(worlds_g)
    me = NeuralController(pop[worlds_g], hidden_self)
    opp_nn = NeuralController(np.repeat(opp_elite[None], B, 0), hidden_opp)
    if species == "prey":
        opp = MixedController(RulePredController(), opp_nn, worlds_nn)
        m = run_episode(worlds_sc, worlds_seed, me, opp)
        f = fitness_prey(m)
    else:
        opp = MixedController(RulePreyController(), opp_nn, worlds_nn)
        m = run_episode(worlds_sc, worlds_seed, opp, me)
        f = fitness_pred(m)
    fit = np.zeros(P)
    np.add.at(fit, worlds_g, f)
    fit /= (B // P)
    return fit, m


def train(cfg, seed, out_path, log=print):
    rng = np.random.default_rng(seed)
    scen = load_scenarios()
    hid = cfg["hidden"]
    spec = MLPSpec(N_IN, hid, N_OUT)
    P = cfg["pop_size"]
    prey_pop = spec.init(rng, P)
    pred_pop = spec.init(rng, P)
    random_init = dict(prey=prey_pop[0].copy(), pred=pred_pop[0].copy())
    pretrained, sup_losses = None, []
    if cfg.get("pred_pretrain", False):
        # Depredadores: pre-entrenamiento supervisado imitando al depredador por reglas
        sup_seeds = [seed * 100000 + 90000 + i for i in range(cfg.get("pretrain_episodes", 30))]
        X, Y = collect_rule_examples(sup_seeds, "pred")
        log(f"   {len(X)} ejemplos (obs, acción) del depredador por reglas")
        pretrained, sup_losses = train_supervised(hid, X, Y, rng, epochs=cfg.get("pretrain_epochs", 30), log=log)
        pred_pop = np.repeat(pretrained[None], P, 0)
        pred_pop[1:] += rng.normal(0, cfg["mutation_sigma"], pred_pop[1:].shape)
    prey_elite, pred_elite = prey_pop[0].copy(), pred_pop[0].copy()
    checkpoints = {}
    history = []
    gen0_elite = None
    t0 = time.time()
    for gen in range(cfg["generations"] + 1):
        seeds_gen = [seed * 100000 + gen * 10 + k for k in range(3)]
        fq, _ = _evaluate("prey", prey_pop, pred_elite, hid, hid, scen, seeds_gen)
        fp, _ = _evaluate("pred", pred_pop, prey_elite, hid, hid, scen, seeds_gen)
        prey_elite = prey_pop[np.argmax(fq)].copy()
        pred_elite = pred_pop[np.argmax(fp)].copy()
        if gen == 0:
            gen0_elite = dict(prey=prey_elite.copy(), pred=pred_elite.copy())
        if gen % cfg["checkpoint_every"] == 0 or gen == cfg["generations"]:
            checkpoints[gen] = dict(prey=prey_elite.copy(), pred=pred_elite.copy())
        rec = dict(
            gen=gen,
            prey_fit_mean=float(fq.mean()), prey_fit_max=float(fq.max()), prey_fit_std=float(fq.std()),
            pred_fit_mean=float(fp.mean()), pred_fit_max=float(fp.max()), pred_fit_std=float(fp.std()),
            prey_elite_norm=float(np.linalg.norm(prey_elite)),
            pred_elite_norm=float(np.linalg.norm(pred_elite)),
            prey_dist_from_init=float(np.linalg.norm(prey_elite - gen0_elite["prey"])),
            pred_dist_from_init=float(np.linalg.norm(pred_elite - gen0_elite["pred"])),
            prey_pop_diversity=float(prey_pop.std(0).mean()),
            pred_pop_diversity=float(pred_pop.std(0).mean()),
            seeds=seeds_gen,
            elapsed_s=round(time.time() - t0, 1),
        )
        history.append(rec)
        if gen % 5 == 0:
            log(f"[seed {seed}] gen {gen:3d} | presa max {rec['prey_fit_max']:.3f} media {rec['prey_fit_mean']:.3f}"
                f" | depred max {rec['pred_fit_max']:.3f} media {rec['pred_fit_mean']:.3f} | {rec['elapsed_s']}s")
        if gen < cfg["generations"]:
            prey_pop = _next_generation(rng, prey_pop, fq, cfg)
            pred_pop = _next_generation(rng, pred_pop, fp, cfg)

    ck_gens = sorted(checkpoints)
    np.savez_compressed(
        out_path,
        hidden=np.array(hid),
        prey_final=prey_elite, pred_final=pred_elite,
        prey_gen0=gen0_elite["prey"], pred_gen0=gen0_elite["pred"],
        prey_random=random_init["prey"], pred_random=random_init["pred"],
        pred_pretrained=pretrained if pretrained is not None else random_init["pred"],
        sup_losses=np.array(sup_losses),
        ck_gens=np.array(ck_gens),
        ck_prey=np.array([checkpoints[g]["prey"] for g in ck_gens]),
        ck_pred=np.array([checkpoints[g]["pred"] for g in ck_gens]),
    )
    with open(str(out_path).replace(".npz", "_history.json"), "w", encoding="utf-8") as f:
        json.dump(dict(config=cfg, seed=seed, history=history), f, indent=1)
    return history
