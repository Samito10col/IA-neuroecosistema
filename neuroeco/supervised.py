"""Aprendizaje supervisado por imitación (clonación de comportamiento).

Se usa para PRE-ENTRENAR la red de los depredadores: se recolectan ejemplos
(observación, acción) del depredador por reglas y se ajusta la red con
retropropagación (error cuadrático medio, optimizador Adam). Después la
neuroevolución parte de esa red y puede mejorarla o empeorarla.

Motivo (bitácora D5): en el piloto, los depredadores evolucionados desde cero
casi no capturaban (0,65 capturas por episodio frente a 10,9 de las reglas en
el escenario a, tras 20 generaciones). La recompensa de captura es escasa.
"""
import numpy as np

from .config import load_scenarios, N_IN, N_OUT
from .nn import MLPSpec
from .env import Ecosystem
from .controllers import RulePreyController, RulePredController


def collect_rule_examples(seeds, species="pred"):
    """Juega episodios reglas-vs-reglas y guarda (obs, acción) de los agentes vivos."""
    scen = load_scenarios()
    names = list(scen)
    sc = [scen[names[i % 3]] for i in range(len(seeds))]
    env = Ecosystem(sc, seeds)
    q, p = RulePreyController(), RulePredController()
    q.reset(env.B, env.prey_pos.shape[1], env.rng)
    p.reset(env.B, env.pred_pos.shape[1], env.rng)
    X, Y = [], []
    while not env.done():
        oq, op = env.observe_prey(), env.observe_pred()
        aq, ap = q.act(oq), p.act(op)
        if species == "pred":
            m = env.pred_alive
            X.append(op[m]); Y.append(ap[m])
        else:
            m = env.prey_alive
            X.append(oq[m]); Y.append(aq[m])
        env.step(aq, ap)
    return np.concatenate(X), np.concatenate(Y)


def train_supervised(hidden, X, Y, rng, epochs=30, batch=256, lr=3e-3, log=print):
    """Retropropagación sobre una MLP tanh. Devuelve el genoma plano y la curva de pérdida."""
    spec = MLPSpec(N_IN, hidden, N_OUT)
    g = spec.init(rng, 1)[0]
    m = np.zeros_like(g); v = np.zeros_like(g)
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    losses = []
    n = len(X)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss, grad = _loss_grad(spec, g, X[idx], Y[idx])
            tot += loss * len(idx)
            t += 1
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad ** 2
            g -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        losses.append(tot / n)
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"   supervisado época {ep:2d}  MSE {losses[-1]:.4f}")
    return g, losses


def _loss_grad(spec, g, x, y):
    layers = spec.unpack(g[None])
    hs = [x]
    h = x
    for W, b in layers:
        h = np.tanh(h @ W[0] + b[0, 0])
        hs.append(h)
    err = hs[-1] - y
    loss = float((err ** 2).mean())
    grads = []
    delta = 2 * err / err.size * (1 - hs[-1] ** 2)
    for li in range(len(layers) - 1, -1, -1):
        W = layers[li][0][0]
        gW = hs[li].T @ delta
        gb = delta.sum(0)
        grads.append((gW, gb))
        if li > 0:
            delta = (delta @ W.T) * (1 - hs[li] ** 2)
    flat = []
    for gW, gb in reversed(grads):
        flat.append(gW.ravel()); flat.append(gb.ravel())
    return loss, np.concatenate(flat)
