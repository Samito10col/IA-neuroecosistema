"""Verificaciones rápidas de que el sistema hace lo que dice (correr antes de la sustentación).

Uso:  python scripts/check.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from neuroeco.config import load_scenarios, N_IN, N_OUT, ROOT  # noqa: E402
from neuroeco.env import Ecosystem, run_episode  # noqa: E402
from neuroeco.nn import MLPSpec  # noqa: E402
from neuroeco.controllers import NeuralController, RulePredController, RulePreyController  # noqa: E402
from neuroeco.supervised import _loss_grad  # noqa: E402

sc = load_scenarios()
ok = 0


def check(cond, msg):
    global ok
    print(("  OK   " if cond else "  FALLA ") + msg)
    ok += bool(cond)
    if not cond:
        raise SystemExit(1)


print("1. Reproducibilidad: misma semilla => mismo resultado")
m1 = run_episode([sc["c_presion"]] * 3, [1, 2, 3], RulePreyController(), RulePredController())
m2 = run_episode([sc["c_presion"]] * 3, [1, 2, 3], RulePreyController(), RulePredController())
check(all(np.array_equal(m1[k], m2[k]) for k in m1), "métricas idénticas en dos ejecuciones")

print("2. La red neuronal controla el movimiento")
spec = MLPSpec(N_IN, [16, 16], N_OUT)
zero = np.zeros((1, spec.n_params))
env = Ecosystem([sc["a_abundancia"]], [9])
p0 = env.prey_pos.copy()
nnz = NeuralController(zero, [16, 16])
for _ in range(5):
    env.step(nnz.act(env.observe_prey()), np.zeros((1, 5, 2)))
check(np.allclose(env.prey_pos, p0), "con todos los pesos en cero las presas no se mueven")
g = zero.copy(); g[0, -2] = 3.0  # sesgo de la salida vx muy positivo
env = Ecosystem([sc["a_abundancia"]], [9])
x0 = env.prey_pos[..., 0].copy()
nng = NeuralController(g, [16, 16])
for _ in range(5):
    env.step(nng.act(env.observe_prey()), np.zeros((1, 5, 2)))
check((env.prey_pos[..., 0] >= x0 - 1e-9).all() and (env.prey_pos[..., 0] > x0).mean() > 0.8,
      "cambiando un solo sesgo de salida las presas se desplazan hacia +x")

print("3. Energía y captura")
env = Ecosystem([dict(sc["a_abundancia"], n_food=0)], [9])   # sin plantas: nadie gana energía
e0 = env.prey_E.copy()
env.step(np.ones((1, 20, 2)), np.zeros((1, 5, 2)))
check((env.prey_E < e0).all(), "la energía disminuye al moverse")
env = Ecosystem([sc["a_abundancia"]], [9])
env.pred_pos[0, 0] = env.prey_pos[0, 0] + 0.5
env.step(np.zeros((1, 20, 2)), np.zeros((1, 5, 2)))
check(not env.prey_alive[0, 0] and env.pred_caps[0, 0] == 1, "un depredador junto a una presa la captura")

print("4. Gradiente del aprendizaje supervisado (comparación numérica)")
rng = np.random.default_rng(0)
s2 = MLPSpec(N_IN, [5, 4], N_OUT)
gg = s2.init(rng, 1)[0] + rng.normal(0, .1, s2.n_params)
x = rng.normal(size=(7, N_IN)); y = rng.uniform(-1, 1, (7, 2))
_, an = _loss_grad(s2, gg, x, y)
num = np.zeros_like(gg)
for i in range(len(gg)):
    e = np.zeros_like(gg); e[i] = 1e-6
    num[i] = (_loss_grad(s2, gg + e, x, y)[0] - _loss_grad(s2, gg - e, x, y)[0]) / 2e-6
check(np.abs(num - an).max() < 1e-6, f"error máximo {np.abs(num - an).max():.2e}")

print("5. Los modelos entrenados existen y difieren de su inicialización")
for p in sorted((ROOT / "models").glob("*.npz")):
    d = np.load(p)
    dq = np.linalg.norm(d["prey_final"] - d["prey_gen0"])
    dp = np.linalg.norm(d["pred_final"] - d["pred_pretrained"])
    check(dq > 0 and dp > 0, f"{p.name}: |Δ presa| = {dq:.2f}, |Δ depredador desde supervisado| = {dp:.2f}")

print(f"\nTodas las verificaciones pasaron ({ok}).")
