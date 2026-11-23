# Guía para la sustentación (uso interno del equipo)

## Guion de la demo (5 minutos)

1. `python scripts/check.py` → todo OK. Explica la prueba 2: pesos en cero = no se mueven; un sesgo cambia el
   movimiento. **Eso demuestra que la decisión sale de la red.**
2. Abrir `viewer/demo_c_red_vs_red.html`, reproducir, pausar, hacer clic en una presa cerca de un depredador:
   mostrar la entrada 5 (distancia al depredador) baja, las capas ocultas y la salida apuntando en sentido contrario.
3. Abrir `viewer/demo_c_reglas_vs_reglas.html` (mismo mapa, misma semilla) y comparar presas vivas al final.
4. Mostrar `results/figures/fig4_curva_aprendizaje.png` y la tabla E3 del informe.

## Si piden modificar un parámetro en vivo

```bash
python scripts/make_replay.py --scenario a_abundancia --n_pred 5 --name cambio_en_vivo
python scripts/make_replay.py --scenario b_escasez --n_food 5 --name escasez_extrema
python scripts/make_replay.py --scenario c_presion --prey rule --pred nn --name reglas_vs_red
```
Para cambiar física (velocidad, radios, costos): editar `neuroeco/config.py` y volver a correr el comando.
Para reentrenar rápido: `python scripts/train.py --config small --seed 11 --generations 10` (~1,5 min).

## Preguntas probables y respuesta corta

**¿Cuántas entradas/salidas y por qué?** 14 entradas (ver informe 2.2), 2 salidas (vx, vy) en [-1, 1] por tanh.
**¿Por qué tanh?** Acota la salida a una velocidad válida, es simétrica y no tiene neuronas muertas como ReLU.
**¿Por qué dos capas de 16?** Mínimo exigido son dos capas; 16 da margen sin un genoma enorme. E2 mostró que 4-4 rinde casi igual: la capacidad no era el cuello de botella.
**¿Cómo cambian los pesos?** Algoritmo genético: los mejores pasan intactos (élite 3), el resto nace de torneo + cruce uniforme + mutación gaussiana σ = 0,1 en 15 % de los genes. No hay gradiente en esa etapa.
**¿Y el depredador?** Primero aprendizaje supervisado (retropropagación + Adam, error cuadrático contra la acción de la regla), después el mismo algoritmo genético. Desde cero no aprendía (0,65 vs 10,9 capturas en el piloto).
**¿Qué se optimiza y qué puede salir mal?** Presa: tiempo vivo + algo de comida → riesgo de esconderse en paredes (pasa 17–31 % del tiempo junto a una pared). Depredador: capturas + energía + acecho → en (c) aprendió a correr de más y se agotó (supervivencia 0,68 → 0,41).
**¿Qué variable cambia en cada experimento?** Solo el controlador de la especie evaluada; oponente siempre por reglas; mismas 30 semillas.
**¿Qué resultado no esperaban?** La red grande casi no ayuda; el depredador evolucionado empeora en (c); en red vs red la ventaja de la presa desaparece (Reina Roja).
**¿Qué cambiarían primero?** El peso de la energía en la aptitud del depredador, por la evidencia de (c).
**¿Cómo leer la fig. 1 (aptitud por generación)?** La de presas sube; la del depredador es plana porque su oponente (la presa élite) mejora a la vez; por eso medimos aparte contra reglas fijas (fig. 4).
**¿Qué es el coseno de comportamiento?** Coseno entre el movimiento y la dirección al objetivo: 1 va directo, 0 indiferente, -1 al revés.
**¿Por qué Mann-Whitney?** Métricas acotadas, sin supuesto de normalidad. Valores p orientativos (90 episodios de 3 modelos).
**¿Qué es la señal de exploración?** Dos entradas (cos, sin) de un rumbo que cambia al azar; la red decide si la usa. Sin ella, sin nada a la vista, la red siempre daba la misma salida.
