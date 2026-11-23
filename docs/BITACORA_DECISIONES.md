# Bitácora de decisiones técnicas — NeuroEcosistema

Cada decisión registra el problema, las alternativas consideradas, la evidencia disponible y el criterio
de selección. Las referencias "E1"–"E4" remiten a `results/RESUMEN_RESULTADOS.md`; "Piloto n" a
`docs/PILOTOS.md`. Cuando una decisión se tomó antes de tener datos, se dice explícitamente.

---

## D1. Qué percibe cada agente (representación del entorno)

**Problema.** La red necesita información suficiente para buscar, perseguir y huir, pero cada entrada
adicional agranda el espacio de búsqueda del algoritmo genético.

**Alternativas.**
1. Estado global del mapa (posiciones de todos los agentes y plantas): cientos de entradas y dependientes del número de agentes.
2. Sensores tipo "rayos" (por ejemplo 8 direcciones × 3 tipos de objeto = 24 entradas).
3. **Vecino más cercano de cada tipo dentro de un radio de percepción** (elegida).

**Decisión.** 14 entradas por especie. Presa: dirección y distancia a la planta disponible más cercana,
al depredador vivo más cercano y al obstáculo más cercano; energía propia; posición normalizada (para
"sentir" las paredes); dos entradas de exploración (D3). Depredador: dirección y distancia a la presa
más cercana; energía; si está digiriendo; dirección al otro depredador más cercano; posición; obstáculo;
exploración.

**Variables excluidas y por qué.** Velocidad de los demás agentes (exigiría memoria o predicción y no era
necesaria para el comportamiento mínimo); segundo vecino más cercano (duplica entradas; queda como mejora);
número de plantas o presas restantes (información global que un animal no tiene); tiempo del episodio (la
red podría aprender conductas atadas al reloj en lugar de al entorno); oclusión por obstáculos (no se
modela: los agentes "ven" a través de ellos — limitación declarada).

**Evidencia.** Con estas entradas la presa neuronal orienta su movimiento hacia la planta con coseno
medio 0,47–0,75 y lejos del depredador con coseno 0,77–0,83 (tabla de comportamiento), lo que muestra que
las entradas sí se usan. Antes de entrenar esos valores eran -0,08 a 0,0 y 0,38–0,40.

**Criterio.** Entradas de tamaño fijo, independientes del número de agentes, interpretables en el visor
y suficientes para los tres comportamientos que pide la pregunta orientadora.

---

## D2. Dirección unitaria + distancia por separado (en lugar del vector relativo escalado)

**Problema.** En la primera versión la entrada era el vector relativo dividido por el radio de percepción.
Cuando el objetivo está cerca ese vector tiende a cero, justo cuando más importa decidir hacia dónde ir.

**Alternativas.** (a) Vector escalado por el radio; (b) vector escalado + distancia; (c) **vector unitario
+ distancia** (elegida); (d) ángulo en radianes (discontinuo en ±π, mala entrada para una red).

**Evidencia.** Piloto 2 → Piloto 3 (mismo presupuesto, misma semilla de entrenamiento): la aptitud máxima de
las presas en la generación 10 pasó de 0,693 a 0,851. Es una comparación indicativa (un solo par de corridas,
y el oponente élite no es idéntico), no una prueba controlada.

**Criterio.** Separar "hacia dónde" de "qué tan lejos" permite que la red use la dirección con pesos pequeños
y regule la urgencia con la distancia.

---

## D3. Señal de exploración en lugar de memoria

**Problema.** La red es una función sin memoria: si no ve nada, todas sus entradas son casi constantes y
produce siempre la misma salida. En el Piloto 1 los agentes sin objetivos a la vista terminaban pegados a
una pared.

**Alternativas.** (a) Red recurrente (memoria): más parámetros y más difícil de evolucionar en el tiempo
disponible; (b) añadir ruido a la acción por fuera de la red: entonces parte del movimiento no lo decide la
red, lo cual contradice la condición de aceptación; (c) **dar a la red dos entradas cos θ, sin θ de un rumbo
que deriva al azar** (θ ← θ + N(0; 0,3) en cada paso), a modo de generador de patrones (elegida).

**Evidencia.** La red decide cuánto usar esa señal. Con la red entrenada, la presa se mueve a velocidad media
0,55–0,66 y la fracción de tiempo junto a la pared baja de 0,47–0,52 (gen. 0) a 0,17–0,31. La
referencia aleatoria usa exactamente esta señal y nada más, por lo que la comparación red vs. aleatorio
mide lo que la red añade sobre la exploración pura.

**Criterio.** Mantener todas las decisiones de movimiento dentro de la red y darle una fuente de variación
que pueda ignorar o aprovechar.

---

## D4. Arquitectura de la red

**Problema.** El enunciado exige al menos dos capas ocultas. Hay que elegir tamaño y activaciones.

**Alternativas.** 14-16-16-2 (562 parámetros); 14-4-4-2 (92 parámetros); 14-32-32-2 (1 602 parámetros);
activación ReLU en ocultas.

**Decisión.** 14-16-16-2 con tanh en todas las capas, igual para ambas especies (pesos independientes).
* tanh en la salida acota naturalmente vx, vy a [-1, 1].
* tanh en ocultas es simétrica, sin neuronas "muertas" y con activaciones legibles en el visor (-1 a 1).
* 32-32 se descartó a priori: casi triplica el genoma y con una población de 30 la búsqueda se vuelve lenta.

**Evidencia (E2, decisión tomada antes; datos posteriores).** La red reducida 4-4 rinde casi igual. Presas:
vida media 0,862 vs 0,829 en (a) (p = 0,005) y 0,649 vs 0,613 en (b) (p < 0,001), sin diferencia en (c)
(p = 0,85). Depredadores: sin diferencia en ningún escenario (p ≥ 0,15).

**Lectura.** La capacidad extra sirve algo a la presa (que debe equilibrar comer y huir) y nada al depredador,
cuya tarea es casi "ir hacia la presa". Si hubiera que cambiar algo, el depredador podría usar 4-4.

---

## D5. Cómo se modifican los pesos: neuroevolución con coevolución

**Problema.** La recompensa (sobrevivir, capturar) solo se conoce al final del episodio y depende de
interacciones no diferenciables (capturas, choques con obstáculos).

**Alternativas.** (a) Aprendizaje por refuerzo con gradiente (PPO, DQN): más eficiente en muestras pero mucho
más complejo de implementar y depurar en el plazo; (b) supervisado puro imitando reglas: la red solo copiaría
a su maestro; (c) **algoritmo genético** (elegida para ambas especies; para el depredador combinado con D6).

**Procedimiento.** Población de 30 genomas por especie, 60 generaciones. Élite de 3 sin cambios; resto por
torneo de 3, cruce uniforme (probabilidad 0,5) y mutación gaussiana (15 % de los genes, σ = 0,1). Cada genoma se
evalúa en 6 episodios: 3 escenarios × 2 oponentes (reglas y élite de la generación anterior). Todos los
genomas de una generación comparten semillas (números aleatorios comunes).

**Por qué oponentes mixtos.** Solo contra reglas, la red se sobreajusta a un oponente fijo y predecible. Solo
coevolución, las dos poblaciones pueden entrar en ciclos ("Reina Roja") y la aptitud deja de ser comparable
entre generaciones. La mezcla da una referencia estable y a la vez un oponente que mejora.

**Evidencia.** Las presas mejoran de forma sostenida en la curva de aprendizaje medida contra reglas (fig. 4):
vida media de 0,65 → 0,87 en (a) y de 0,41 → 0,66 en (c) entre la generación 0 y la 60. Los pesos de la élite se
alejan de la élite inicial hasta una distancia L2 de 7,6–10,8 (fig. 2). La aptitud de entrenamiento del
depredador, en cambio, es casi plana (fig. 1): su oponente élite mejora al mismo ritmo.

---

## D6. Pre-entrenamiento supervisado del depredador (configuraciones de entrenamiento diferenciadas)

**Problema.** En los pilotos 1–3 los depredadores evolucionados desde cero casi no capturaban: 0,65 capturas
por episodio frente a 10,9 de las reglas en (a). Una red aleatoria casi nunca toca una presa, así que la
selección no tiene diferencias que aprovechar (recompensa escasa).

**Alternativas.** (a) Más generaciones (costo de tiempo incierto); (b) solo moldear la recompensa con el
término de acecho (probado en el Piloto 2: insuficiente); (c) **clonar primero al depredador por reglas con
aprendizaje supervisado y luego afinar con neuroevolución** (elegida).

**Procedimiento.** 30 episodios reglas-vs-reglas (semillas propias), 33 000–34 000 pares (observación, acción),
retropropagación con Adam, 30 épocas. MSE 0,111–0,123 → 0,021–0,024 según la semilla (fig. 3). La población inicial del depredador es esa
red más mutaciones.

**Evidencia (E1).** Red aleatoria: 0,0 capturas. Red supervisada: 5,90 / 4,17 / 3,64 capturas por depredador
en (a)/(b)/(c), equivalente a las reglas. Tras la neuroevolución: 6,83 / 5,04 / 3,66. La mejora sobre la red
supervisada es significativa en (a) y (b) (p < 10⁻⁴) y nula en (c) (p = 0,95).

**Riesgo asumido.** La red empieza imitando una estrategia humana, así que su comportamiento final no es
"descubierto desde cero". Por eso se reporta por separado lo que aporta cada etapa.

---

## D7. Qué se optimiza (funciones de aptitud) y qué podrían favorecer indebidamente

**Presa:** vida media normalizada + 0,02 × plantas comidas por presa.
**Depredador:** capturas por depredador + 0,5 × energía final + 1,0 × acecho (cercanía media a la presa más
cercana, en [0, 1]).

**Conductas indebidas previstas y verificación.**
* *Presa que se esconde en rincones o paredes.* Allí hay menos direcciones de ataque. Medido: la presa
  entrenada pasa 17–31 % del tiempo a menos de 5 unidades de una pared; la presa por reglas, ~0 %. Es una
  estrategia parcialmente "de rincón", pero no exclusiva: la misma red come 12,3 plantas por presa en (a)
  frente a 9,0 de las reglas.
* *Depredador que acecha sin capturar* (el término de acecho premia estar cerca). Medido: el coseno de
  persecución es 0,94–0,96 y las capturas suben, así que no ocurrió.
* *Depredador que se agota.* Nada en la aptitud penaliza fuertemente moverse rápido. **Ocurrió en (c):** el
  depredador evolucionado se mueve a velocidad 0,88 frente a 0,72 de la red supervisada, termina con menos
  energía (0,15 vs 0,29) y sobrevive menos (0,41 vs 0,68), sin capturar más. Es la conducta indebida más clara
  del proyecto y motiva la mejora principal (ver informe, sección 8).

**Criterio.** El peso de cada término se fijó para que la meta principal domine (vida en la presa, capturas en
el depredador). No se ajustaron con búsqueda sistemática; es una limitación declarada.

---

## D8. Diseño experimental

**Qué cambia y qué se mantiene.** En E1–E3 la única variable es el controlador de la especie evaluada; el
oponente es siempre el agente por reglas, igual en todas las condiciones, y se usan las mismas 30 semillas por
escenario. Las semillas de prueba (5000–5029) nunca se usan para entrenar.

**Repeticiones.** 3 semillas de entrenamiento × 30 semillas de prueba = 90 episodios por escenario para
cada red; 30 para cada referencia. Se reporta media ± desviación estándar y la variación entre semillas de
entrenamiento (E2).

**Prueba estadística.** Mann-Whitney bilateral, porque las métricas están acotadas (fracciones, conteos) y no
se puede suponer normalidad. Advertencia: los 90 episodios de una red no son totalmente independientes (vienen
de 3 modelos), así que los valores p deben leerse como orientativos y siempre junto al tamaño de la diferencia.

**Alternativa descartada.** Evaluar todo como "red contra red": mezcla el efecto de las dos especies y no
permite atribuir la diferencia. Se reporta solo como descripción del ecosistema completo (E4).

---

## D9. Reglas del mundo: población fija, sin reproducción

**Alternativas.** (a) Poblaciones que se reproducen y mueren (dinámica Lotka-Volterra); (b) **población fija
por episodio de 400 pasos** (elegida).

**Criterio.** Con reproducción, las métricas por individuo dependen del tamaño cambiante de la población y la
comparación entre controladores se vuelve confusa. Con población fija cada episodio es una prueba comparable.
La digestión de 20 pasos tras una captura evita que un depredador vacíe el mapa en segundos.

**Limitación.** No se estudia el equilibrio poblacional a largo plazo.
