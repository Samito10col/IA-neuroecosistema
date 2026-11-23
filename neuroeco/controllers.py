"""Controladores de agentes: red neuronal, reglas fijas y aleatorio.

Todos reciben las observaciones (B, N, 12) y devuelven acciones (B, N, 2)
en [-1, 1]. Las acciones son el vector de movimiento deseado; el entorno
recorta la norma a 1 y la escala por la velocidad máxima de la especie.
"""
import numpy as np
from .nn import MLPSpec
from .config import N_IN, N_OUT


class NeuralController:
    kind = "nn"

    def __init__(self, genomes, hidden):
        self.spec = MLPSpec(N_IN, hidden, N_OUT)
        self.genomes = np.asarray(genomes, dtype=float)
        self.layers = self.spec.unpack(self.genomes)
        self.last_hidden = None

    def reset(self, B, N, rng):
        pass

    def act(self, obs, record=False):
        out, hid = self.spec.forward(self.layers, obs, return_hidden=True)
        if record:
            self.last_hidden = hid
        return out


class RandomController:
    """Caminata aleatoria persistente: sigue la señal de exploración (entradas 12-13),
    un rumbo que deriva al azar. Se desplaza por el mapa pero ignora lo que percibe."""
    kind = "random"

    def reset(self, B, N, rng):
        pass

    def act(self, obs, record=False):
        return 0.7 * obs[..., 12:14]


def _wall_push(obs):
    """Empuje suave hacia el centro cuando el agente está cerca de una pared.
    obs[..., 7:9] es la posición normalizada en [-1, 1]."""
    p = obs[..., 7:9]
    push = np.where(np.abs(p) > 0.85, -np.sign(p), 0.0)
    return push


class RulePreyController:
    """Presa heurística: huye si ve un depredador cercano; si no, va al alimento; si no, deambula."""
    kind = "rule"

    def reset(self, B, N, rng):
        self.wander = RandomController()
        self.wander.reset(B, N, rng)

    def act(self, obs, record=False):
        food_rel, food_d = obs[..., 0:2], obs[..., 2]
        pred_rel, pred_d = obs[..., 3:5], obs[..., 5]
        w = self.wander.act(obs)
        a = w.copy()
        sees_food = food_d < 1.0
        a = np.where(sees_food[..., None], _unit(food_rel) * 0.8, a)
        danger = pred_d < 0.6
        a = np.where(danger[..., None], -_unit(pred_rel), a)
        a = a + _wall_push(obs)
        return np.clip(a, -1, 1)


class RulePredController:
    """Depredador heurístico: persigue a la presa más cercana a máxima velocidad;
    en digestión se mueve despacio; sin presas a la vista, deambula."""
    kind = "rule"

    def reset(self, B, N, rng):
        self.wander = RandomController()
        self.wander.reset(B, N, rng)

    def act(self, obs, record=False):
        prey_rel, prey_d = obs[..., 0:2], obs[..., 2]
        cooling = obs[..., 4] > 0.5
        a = self.wander.act(obs)
        sees = prey_d < 1.0
        a = np.where(sees[..., None], _unit(prey_rel), a)
        a = np.where(cooling[..., None], a * 0.3, a)
        a = a + _wall_push(obs)
        return np.clip(a, -1, 1)


class MixedController:
    """Usa un controlador u otro según el mundo (máscara por mundo).
    Sirve para entrenar contra oponentes mixtos (reglas + élite neuronal)."""

    def __init__(self, ctrl_a, ctrl_b, use_b):
        self.a, self.b, self.use_b = ctrl_a, ctrl_b, np.asarray(use_b, bool)
        self.kind = f"mixed({ctrl_a.kind},{ctrl_b.kind})"

    def reset(self, B, N, rng):
        self.a.reset(B, N, rng)
        self.b.reset(B, N, rng)

    def act(self, obs, record=False):
        return np.where(self.use_b[:, None, None], self.b.act(obs), self.a.act(obs))


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)
