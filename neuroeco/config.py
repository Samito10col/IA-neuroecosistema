"""Parámetros físicos del mundo y definición de escenarios.

Todos los valores numéricos del proyecto viven aquí o en config/*.json,
para que cada experimento quede registrado y sea reproducible.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- mundo
WORLD = dict(
    size=100.0,          # arena cuadrada de 100 x 100, con paredes
    steps=400,           # duración máxima de un episodio
    n_prey=20,           # presas por episodio (fijo en todos los escenarios)
    max_pred=5,          # tamaño máximo de arreglo para depredadores
    max_food=60,
    max_obst=6,
)

PREY = dict(
    speed=1.0,           # desplazamiento máximo por paso
    sense=25.0,          # radio de percepción
    eat_radius=2.0,
    food_gain=0.30,      # energía que aporta una planta
    base_cost=0.0020,    # gasto por paso aunque no se mueva
    move_cost=0.0030,    # gasto adicional * velocidad^2
    energy0=0.6,
)

PRED = dict(
    speed=1.15,          # un poco más rápido que la presa...
    sense=30.0,
    capture_radius=2.0,
    prey_gain=0.50,
    base_cost=0.0020,
    move_cost=0.0045,    # ...pero le cuesta más moverse
    cooldown=20,         # pasos de "digestión" tras una captura
    energy0=0.7,
)

N_IN = 14   # entradas de cada red (ver env.observe_*)
N_OUT = 2   # salida: vector de movimiento (vx, vy) en [-1, 1]


def load_scenarios():
    with open(ROOT / "config" / "scenarios.json", encoding="utf-8") as f:
        return json.load(f)


def load_training(name):
    with open(ROOT / "config" / "training.json", encoding="utf-8") as f:
        return json.load(f)[name]
