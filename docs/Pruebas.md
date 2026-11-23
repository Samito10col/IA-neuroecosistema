# Registro de pruebas piloto (antes de la configuración final)

Estas corridas cortas se hicieron con la semilla de entrenamiento 1 y se evaluaron
con las semillas 5000–5019 (20 por escenario) contra el oponente por reglas.
Son la evidencia que motivó las decisiones D3, D4 y D5 de la bitácora. Los modelos
piloto no se conservaron porque el código cambió después de cada prueba; los números
se copiaron de la salida de consola.

## Piloto 1 — 12 entradas con vectores escalados por el radio, sin moldeado de recompensa

* Aptitud del depredador = capturas por depredador + 0,5 × energía final.
* Aptitud máxima del depredador: 2,16 en la generación 0 → 0,34 (gen. 5) → 0,53 (gen. 10) → 0,37 (gen. 15).
* El valor alto de la generación 0 es engañoso: su oponente era una presa con pesos aleatorios
  que casi no se mueve. En cuanto las presas mejoraron, los depredadores casi no capturaban.
* Diagnóstico: recompensa escasa (casi ningún depredador aleatorio captura) y, sin presas a la
  vista, una red sin memoria produce siempre la misma salida y termina pegada a una pared.

## Piloto 2 — se añade el término de acecho y la señal de exploración (entradas 12–13)

* Aptitud máxima de presas en la generación 10: 0,693.
* Aptitud máxima del depredador en la generación 10: 0,81. Sigue siendo bajo.

## Piloto 3 — direcciones unitarias + distancia aparte (configuración de entradas final), 20 generaciones

| Escenario | Vida media presas (red) | Vida media presas (reglas) | Capturas depredador red | Capturas depredador reglas |
|---|---|---|---|---|
| (a) | 0,756 | 0,692 | 0,65 | 10,9 |
| (b) | 0,583 | 0,536 | 0,70 | 8,0 |
| (c) | 0,514 | 0,416 | 2,45 | 18,2 |

* La aptitud máxima de presas en la generación 10 subió de 0,693 (piloto 2) a 0,851.
  Con el vector escalado, la dirección hacia un objetivo cercano tiende a cero justo cuando
  más importa; la dirección unitaria la mantiene. (Comparación indicativa: mismo
  presupuesto y semilla, pero el oponente élite no es idéntico.)
* Las presas neuronales ya superaban a las presas por reglas. Los depredadores neuronales no:
  capturaban entre 7 y 26 veces menos que los de reglas.

## Piloto 4 — pre-entrenamiento supervisado del depredador + neuroevolución (10 generaciones)

* 33 898 pares (observación, acción) del depredador por reglas; MSE 0,1086 → 0,0209 en 30 épocas.
* Depredador supervisado vs. presa por reglas, capturas: 11,85 / 7,80 / 17,95 (reglas: 12,25 / 8,15 / 18,0).
* Tras 10 generaciones de neuroevolución: 9,70 / 5,55 / 15,55. Es decir, **la evolución empeoró
  ligeramente al depredador frente a la presa por reglas**, aunque su aptitud de entrenamiento no
  cayó: la aptitud mezcla episodios contra la presa élite neuronal (que cambia) y premia el acecho
  y la energía, no solo las capturas. Esto se analiza en el informe (sección de discusión).
