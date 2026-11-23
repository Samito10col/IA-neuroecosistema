# NeuroEcosistema — Depredadores y presas con redes neuronales

Proyecto Final Integrador · Electiva I de Profundización – Inteligencia Artificial
Universidad Santo Tomás, Seccional Tunja · Facultad de Ingeniería de Sistemas

Simulación 2D en la que **presas** y **depredadores** deciden cómo moverse a partir de la salida de
una **red neuronal multicapa** (dos capas ocultas) propia de cada especie. Las redes se entrenan con
**neuroevolución** (algoritmo genético con coevolución) y, en el caso del depredador, con un
**pre-entrenamiento supervisado** por imitación. Todo se compara contra agentes por reglas y aleatorios
en tres escenarios, con semillas registradas.

> Pregunta orientadora: ¿en qué condiciones los agentes controlados por redes neuronales desarrollan
> comportamientos de búsqueda, persecución y evasión que mejoran su desempeño frente a estrategias sin
> aprendizaje?

## 1. Instalación

Requiere Python 3.10 o superior. No usa GPU ni librerías de aprendizaje profundo: la red, la
retropropagación y el algoritmo genético están escritos con NumPy para que cada paso sea visible.

```bash
git clone <URL-del-repositorio>
cd neuroecosistema
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python scripts/check.py        # verificaciones rápidas (deben salir todas OK)
```

## 2. Ver la simulación (demo)

### 2.1 Simulación en vivo (recomendada para la sustentación)

Abra `viewer/juego.html` con doble clic. Permite cambiar en vivo el número de presas, depredadores,
plantas, obstáculos y el rebrote; elegir el "cerebro" de cada especie (red, reglas o aleatorio); elegir
el **nivel de entrenamiento** (sin entrenar, solo supervisado, generaciones 0 a 60) y la arquitectura
(16-16 o 4-4). Cada cambio reinicia el mundo con la misma semilla, para que la única diferencia sea el
parámetro cambiado. Al hacer clic en un animal se ven sus 14 entradas, sus dos capas ocultas y su salida.

Es una réplica en JavaScript de `neuroeco/env.py` que usa los pesos reales de `models/`. Se validó contra
Python: la red da las mismas salidas (diferencia máx. 2·10⁻⁶) y los promedios en 30 mapas coinciden
(p. ej. supervivencia de presas neuronales en (a): 0,76 en JS vs 0,75 en Python). Su generador aleatorio
es distinto, así que un mapa con la misma semilla no es idéntico al de Python; las cifras oficiales salen
de `scripts/run_experiments.py`. Para regenerarla: `python scripts/make_game.py`.

### 2.2 Episodios grabados desde Python

Los visores son archivos HTML autocontenidos: se abren con doble clic, sin servidor.

```bash
python scripts/make_replay.py --demo     # genera viewer/demo_*.html (ya vienen generados)
```

Abra `viewer/demo_a_red_vs_red.html`. Puede reproducir, pausar, avanzar paso a paso y **hacer clic en
cualquier agente** para ver sus 14 entradas, las activaciones de sus dos capas ocultas y su salida
(vx, vy), que es exactamente el vector de movimiento que se aplica en el entorno.

Cambiar un parámetro en vivo (útil en la sustentación):

```bash
python scripts/make_replay.py --scenario a_abundancia --n_pred 4 --name prueba_4_depredadores
python scripts/make_replay.py --scenario b_escasez --prey rule --pred nn --seed 5007
python scripts/make_replay.py --model models/small_seed11.npz --scenario c_presion
```

## 3. Reproducir todo desde cero

```bash
python scripts/train.py --all          # 6 entrenamientos (base y small x 3 semillas), ~8 min c/u
python scripts/run_experiments.py      # evalúa con las semillas de prueba -> results/data/*.csv
python scripts/analyze.py              # tablas, pruebas estadísticas y figuras -> results/
python scripts/make_replay.py --demo   # visores
```

Un solo entrenamiento: `python scripts/train.py --config base --seed 11`
Prueba rápida: `python scripts/train.py --config small --seed 11 --generations 5`

## 4. Estructura

```
config/
  scenarios.json      escenarios (a) abundancia, (b) escasez, (c) presión + obstáculos
  training.json       hiperparámetros de entrenamiento y semillas (base 16-16, small 4-4)
  experiments.json    semillas de prueba (5000-5029) y de la curva de aprendizaje (7000-7009)
neuroeco/
  config.py           constantes físicas (velocidades, radios, costos de energía)
  env.py              entorno vectorizado (B mundos en paralelo), percepción, reglas y métricas
  nn.py               MLP en NumPy (tanh), genoma plano <-> capas
  controllers.py      controlador neuronal, por reglas y aleatorio
  evolution.py        algoritmo genético con coevolución y funciones de aptitud
  supervised.py       clonación de comportamiento con retropropagación + Adam
  evaluate.py         evaluación por escenario y semilla
scripts/              train, run_experiments, analyze, make_replay, check
models/               modelos entrenados (.npz) + historial por generación (_history.json)
results/data          datos crudos por episodio (CSV)
results/tables        resúmenes y pruebas de Mann-Whitney
results/figures       gráficas
results/RESUMEN_RESULTADOS.md   tablas de todos los experimentos
viewer/               visor HTML y demos
docs/                 informe técnico, bitácora de decisiones, pilotos, guía de sustentación
```

## 5. Resumen del diseño

| Elemento | Presa | Depredador |
|---|---|---|
| Red | 14 → 16 → 16 → 2, 562 parámetros | 14 → 16 → 16 → 2, 562 parámetros |
| Entradas | dirección y distancia a planta, a depredador y a obstáculo; energía; posición; señal de exploración | dirección y distancia a presa; energía; digestión; dirección a otro depredador; posición; obstáculo; exploración |
| Salida | vector de movimiento (vx, vy) | vector de movimiento (vx, vy) |
| Entrenamiento | neuroevolución desde pesos aleatorios | supervisado (imitación de reglas) + neuroevolución |
| Aptitud | vida media + 0,02 × plantas comidas | capturas/depredador + 0,5 × energía final + acecho |
| Física | velocidad 1,0 · percepción 25 | velocidad 1,15 · percepción 30 · digestión 20 pasos |

Detalle completo y justificación: `docs/INFORME_TECNICO.md` y `docs/BITACORA_DECISIONES.md`.

## 6. Semillas y trazabilidad

* Disposición inicial de cada mundo: depende solo de su semilla.
* Entrenamiento: semillas 11, 22, 33; los episodios de la generación *g* usan `semilla·100000 + g·10 + k`.
* Pre-entrenamiento supervisado: `semilla·100000 + 90000 + i`.
* Evaluación: 5000–5029 (nunca usadas en entrenamiento). Curva de aprendizaje: 7000–7009.
* Cada fila de `results/data/*.csv` registra escenario, semilla, controlador, configuración y semilla de entrenamiento.
