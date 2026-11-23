# Resumen de resultados (generado automáticamente por scripts/analyze.py)

Valores: media ± desviación estándar sobre episodios. Semillas de prueba 5000-5029 (30 por escenario). Para las redes, n = 3 semillas de entrenamiento × 30 semillas de prueba = 90. En E1-E3 el oponente es siempre el agente por reglas.


## E1 — Antes vs. después del entrenamiento (presas)

| Escenario | Controlador | Vida media de presas | Supervivencia final de presas | Alimento consumido por presa | Energía final de presas vivas | n |
|---|---|---|---|---|---|---|
| (a) abundancia | red_aleatoria_init | 0.487 ± 0.081 | 0.116 ± 0.107 | 1.302 ± 0.863 | 0.225 ± 0.231 | 90 |
| (a) abundancia | red_gen0 | 0.625 ± 0.091 | 0.343 ± 0.124 | 2.282 ± 0.597 | 0.430 ± 0.112 | 90 |
| (a) abundancia | red_final | 0.862 ± 0.067 | 0.750 ± 0.140 | 12.3 ± 1.957 | 0.877 ± 0.056 | 90 |
| (b) escasez | red_aleatoria_init | 0.438 ± 0.066 | 0.027 ± 0.044 | 0.267 ± 0.186 | 0.076 ± 0.140 | 90 |
| (b) escasez | red_gen0 | 0.493 ± 0.076 | 0.065 ± 0.065 | 0.500 ± 0.177 | 0.164 ± 0.155 | 90 |
| (b) escasez | red_final | 0.649 ± 0.059 | 0.239 ± 0.095 | 1.419 ± 0.273 | 0.443 ± 0.148 | 90 |
| (c) presión+obst. | red_aleatoria_init | 0.287 ± 0.091 | 0.027 ± 0.042 | 0.774 ± 0.466 | 0.089 ± 0.185 | 90 |
| (c) presión+obst. | red_gen0 | 0.379 ± 0.108 | 0.119 ± 0.106 | 1.468 ± 0.485 | 0.262 ± 0.207 | 90 |
| (c) presión+obst. | red_final | 0.611 ± 0.108 | 0.369 ± 0.161 | 5.951 ± 2.014 | 0.743 ± 0.139 | 90 |

**Pruebas de Mann-Whitney (bilateral) — Vida media de presas (fracción del episodio)**

| Escenario | Comparación | Media A | Media B | Diferencia | p |
|---|---|---|---|---|---|
| (a) abundancia | red_final vs red_gen0 | 0.862 | 0.625 | 0.237 | 7.06e-28 |
| (a) abundancia | red_final vs red_aleatoria_init | 0.862 | 0.487 | 0.375 | 5.05e-31 |
| (b) escasez | red_final vs red_gen0 | 0.649 | 0.493 | 0.156 | 5.99e-24 |
| (b) escasez | red_final vs red_aleatoria_init | 0.649 | 0.438 | 0.211 | 1.02e-29 |
| (c) presión+obst. | red_final vs red_gen0 | 0.611 | 0.379 | 0.232 | 2.02e-23 |
| (c) presión+obst. | red_final vs red_aleatoria_init | 0.611 | 0.287 | 0.324 | 4.99e-29 |

## E1 — Antes vs. después del entrenamiento (depredadores)

| Escenario | Controlador | Capturas por depredador | Pasos hasta la primera captura | Energía final de depredadores | Supervivencia de depredadores | n |
|---|---|---|---|---|---|---|
| (a) abundancia | red_aleatoria_init | 0.006 ± 0.053 | 397.7 ± 22.0 | 0.000 ± 0.000 | 0.000 ± 0.000 | 90 |
| (a) abundancia | red_supervisada | 5.900 ± 1.565 | 54.1 ± 17.9 | 0.676 ± 0.199 | 0.922 ± 0.182 | 90 |
| (a) abundancia | red_gen0 | 5.778 ± 1.610 | 53.8 ± 17.9 | 0.623 ± 0.215 | 0.906 ± 0.224 | 90 |
| (a) abundancia | red_final | 6.828 ± 1.676 | 53.6 ± 16.2 | 0.526 ± 0.195 | 0.861 ± 0.249 | 90 |
| (b) escasez | red_aleatoria_init | 0.000 ± 0.000 | 400.0 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 90 |
| (b) escasez | red_supervisada | 4.172 ± 0.969 | 55.7 ± 18.2 | 0.421 ± 0.160 | 0.806 ± 0.256 | 90 |
| (b) escasez | red_gen0 | 4.222 ± 1.197 | 54.8 ± 18.3 | 0.393 ± 0.188 | 0.789 ± 0.290 | 90 |
| (b) escasez | red_final | 5.039 ± 1.195 | 52.3 ± 16.1 | 0.361 ± 0.193 | 0.767 ± 0.320 | 90 |
| (c) presión+obst. | red_aleatoria_init | 0.000 ± 0.000 | 400.0 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 90 |
| (c) presión+obst. | red_supervisada | 3.636 ± 0.489 | 27.3 ± 16.5 | 0.293 ± 0.108 | 0.682 ± 0.187 | 90 |
| (c) presión+obst. | red_gen0 | 3.518 ± 0.441 | 27.1 ± 16.2 | 0.285 ± 0.107 | 0.669 ± 0.208 | 90 |
| (c) presión+obst. | red_final | 3.662 ± 0.413 | 25.8 ± 14.7 | 0.153 ± 0.094 | 0.409 ± 0.196 | 90 |

**Pruebas de Mann-Whitney (bilateral) — Capturas por depredador**

| Escenario | Comparación | Media A | Media B | Diferencia | p |
|---|---|---|---|---|---|
| (a) abundancia | red_final vs red_gen0 | 6.828 | 5.778 | 1.050 | 4.79e-06 |
| (a) abundancia | red_final vs red_aleatoria_init | 6.828 | 0.006 | 6.822 | 3.81e-35 |
| (a) abundancia | red_final vs red_supervisada | 6.828 | 5.900 | 0.928 | 1.52e-05 |
| (a) abundancia | red_supervisada vs red_aleatoria_init | 5.900 | 0.006 | 5.894 | 3.93e-35 |
| (b) escasez | red_final vs red_gen0 | 5.039 | 4.222 | 0.817 | 5.15e-07 |
| (b) escasez | red_final vs red_aleatoria_init | 5.039 | 0.000 | 5.039 | 2.36e-35 |
| (b) escasez | red_final vs red_supervisada | 5.039 | 4.172 | 0.867 | 5.49e-09 |
| (b) escasez | red_supervisada vs red_aleatoria_init | 4.172 | 0.000 | 4.172 | 2.15e-35 |
| (c) presión+obst. | red_final vs red_gen0 | 3.662 | 3.518 | 0.144 | 0.00862 |
| (c) presión+obst. | red_final vs red_aleatoria_init | 3.662 | 0.000 | 3.662 | 1.42e-35 |
| (c) presión+obst. | red_final vs red_supervisada | 3.662 | 3.636 | 0.027 | 0.951 |
| (c) presión+obst. | red_supervisada vs red_aleatoria_init | 3.636 | 0.000 | 3.636 | 1.28e-35 |

## E2 — Arquitectura base 16-16 vs. reducida 4-4 (presas)

| Escenario | Controlador | Vida media de presas | Supervivencia final de presas | Alimento consumido por presa | Energía final de presas vivas | n |
|---|---|---|---|---|---|---|
| (a) abundancia | base | 0.862 ± 0.067 | 0.750 ± 0.140 | 12.3 ± 1.957 | 0.877 ± 0.056 | 90 |
| (a) abundancia | small | 0.829 ± 0.081 | 0.667 ± 0.160 | 9.969 ± 2.581 | 0.817 ± 0.105 | 90 |
| (b) escasez | base | 0.649 ± 0.059 | 0.239 ± 0.095 | 1.419 ± 0.273 | 0.443 ± 0.148 | 90 |
| (b) escasez | small | 0.613 ± 0.070 | 0.214 ± 0.098 | 1.304 ± 0.301 | 0.489 ± 0.192 | 90 |
| (c) presión+obst. | base | 0.611 ± 0.108 | 0.369 ± 0.161 | 5.951 ± 2.014 | 0.743 ± 0.139 | 90 |
| (c) presión+obst. | small | 0.602 ± 0.114 | 0.356 ± 0.171 | 6.177 ± 2.307 | 0.736 ± 0.217 | 90 |

**Pruebas de Mann-Whitney (bilateral) — Vida media de presas (fracción del episodio)**

| Escenario | Comparación | Media A | Media B | Diferencia | p |
|---|---|---|---|---|---|
| (a) abundancia | base vs small | 0.862 | 0.829 | 0.032 | 0.00549 |
| (b) escasez | base vs small | 0.649 | 0.613 | 0.035 | 0.000702 |
| (c) presión+obst. | base vs small | 0.611 | 0.602 | 0.008 | 0.854 |

Media de vida media de presas (fracción del episodio) por semilla de entrenamiento:

|               |   a_abundancia |   b_escasez |   c_presion |
|:--------------|---------------:|------------:|------------:|
| ('base', 11)  |          0.86  |       0.643 |       0.625 |
| ('base', 22)  |          0.843 |       0.63  |       0.589 |
| ('base', 33)  |          0.882 |       0.673 |       0.617 |
| ('small', 11) |          0.866 |       0.6   |       0.663 |
| ('small', 22) |          0.813 |       0.584 |       0.566 |
| ('small', 33) |          0.809 |       0.656 |       0.577 |

## E2 — Arquitectura base 16-16 vs. reducida 4-4 (depredadores)

| Escenario | Controlador | Capturas por depredador | Pasos hasta la primera captura | Energía final de depredadores | Supervivencia de depredadores | n |
|---|---|---|---|---|---|---|
| (a) abundancia | base | 6.828 ± 1.676 | 53.6 ± 16.2 | 0.526 ± 0.195 | 0.861 ± 0.249 | 90 |
| (a) abundancia | small | 6.828 ± 1.766 | 51.1 ± 17.5 | 0.591 ± 0.186 | 0.889 ± 0.209 | 90 |
| (b) escasez | base | 5.039 ± 1.195 | 52.3 ± 16.1 | 0.361 ± 0.193 | 0.767 ± 0.320 | 90 |
| (b) escasez | small | 4.850 ± 1.138 | 51.1 ± 16.9 | 0.358 ± 0.171 | 0.750 ± 0.283 | 90 |
| (c) presión+obst. | base | 3.662 ± 0.413 | 25.8 ± 14.7 | 0.153 ± 0.094 | 0.409 ± 0.196 | 90 |
| (c) presión+obst. | small | 3.622 ± 0.427 | 24.8 ± 13.9 | 0.149 ± 0.085 | 0.438 ± 0.207 | 90 |

**Pruebas de Mann-Whitney (bilateral) — Capturas por depredador**

| Escenario | Comparación | Media A | Media B | Diferencia | p |
|---|---|---|---|---|---|
| (a) abundancia | base vs small | 6.828 | 6.828 | 0.000 | 0.783 |
| (b) escasez | base vs small | 5.039 | 4.850 | 0.189 | 0.154 |
| (c) presión+obst. | base vs small | 3.662 | 3.622 | 0.040 | 0.467 |

Media de capturas por depredador por semilla de entrenamiento:

|               |   a_abundancia |   b_escasez |   c_presion |
|:--------------|---------------:|------------:|------------:|
| ('base', 11)  |          6.967 |       4.867 |       3.653 |
| ('base', 22)  |          6.883 |       5.067 |       3.733 |
| ('base', 33)  |          6.633 |       5.183 |       3.6   |
| ('small', 11) |          7.067 |       5.05  |       3.44  |
| ('small', 22) |          6.733 |       4.7   |       3.673 |
| ('small', 33) |          6.683 |       4.8   |       3.753 |

## E3 — Red neuronal vs. referencias (presas)

| Escenario | Controlador | Vida media de presas | Supervivencia final de presas | Alimento consumido por presa | Energía final de presas vivas | n |
|---|---|---|---|---|---|---|
| (a) abundancia | red neuronal | 0.862 ± 0.067 | 0.750 ± 0.140 | 12.3 ± 1.957 | 0.877 ± 0.056 | 90 |
| (a) abundancia | reglas | 0.671 ± 0.075 | 0.303 ± 0.139 | 8.983 ± 1.511 | 0.670 ± 0.150 | 30 |
| (a) abundancia | aleatorio | 0.486 ± 0.059 | 0.117 ± 0.066 | 2.222 ± 0.406 | 0.372 ± 0.246 | 30 |
| (b) escasez | red neuronal | 0.649 ± 0.059 | 0.239 ± 0.095 | 1.419 ± 0.273 | 0.443 ± 0.148 | 90 |
| (b) escasez | reglas | 0.544 ± 0.047 | 0.085 ± 0.088 | 1.360 ± 0.408 | 0.422 ± 0.338 | 30 |
| (b) escasez | aleatorio | 0.379 ± 0.035 | 0.003 ± 0.013 | 0.395 ± 0.129 | 0.007 ± 0.028 | 30 |
| (c) presión+obst. | red neuronal | 0.611 ± 0.108 | 0.369 ± 0.161 | 5.951 ± 2.014 | 0.743 ± 0.139 | 90 |
| (c) presión+obst. | reglas | 0.408 ± 0.074 | 0.042 ± 0.072 | 5.800 ± 1.290 | 0.258 ± 0.376 | 30 |
| (c) presión+obst. | aleatorio | 0.277 ± 0.056 | 0.017 ± 0.027 | 1.455 ± 0.322 | 0.131 ± 0.268 | 30 |

**Pruebas de Mann-Whitney (bilateral) — Vida media de presas (fracción del episodio)**

| Escenario | Comparación | Media A | Media B | Diferencia | p |
|---|---|---|---|---|---|
| (a) abundancia | red neuronal vs reglas | 0.862 | 0.671 | 0.191 | 4.75e-15 |
| (a) abundancia | red neuronal vs aleatorio | 0.862 | 0.486 | 0.375 | 2.87e-16 |
| (b) escasez | red neuronal vs reglas | 0.649 | 0.544 | 0.104 | 1.11e-11 |
| (b) escasez | red neuronal vs aleatorio | 0.649 | 0.379 | 0.270 | 2.87e-16 |
| (c) presión+obst. | red neuronal vs reglas | 0.611 | 0.408 | 0.203 | 5.38e-13 |
| (c) presión+obst. | red neuronal vs aleatorio | 0.611 | 0.277 | 0.334 | 3.68e-16 |

**Pruebas de Mann-Whitney (bilateral) — Supervivencia final de presas**

| Escenario | Comparación | Media A | Media B | Diferencia | p |
|---|---|---|---|---|---|
| (a) abundancia | red neuronal vs reglas | 0.750 | 0.303 | 0.447 | 2.48e-15 |
| (a) abundancia | red neuronal vs aleatorio | 0.750 | 0.117 | 0.633 | 2.24e-16 |
| (b) escasez | red neuronal vs reglas | 0.239 | 0.085 | 0.154 | 5.18e-10 |
| (b) escasez | red neuronal vs aleatorio | 0.239 | 0.003 | 0.236 | 1.53e-16 |
| (c) presión+obst. | red neuronal vs reglas | 0.369 | 0.042 | 0.328 | 4.76e-15 |
| (c) presión+obst. | red neuronal vs aleatorio | 0.369 | 0.017 | 0.353 | 2.19e-16 |

## E3 — Red neuronal vs. referencias (depredadores)

| Escenario | Controlador | Capturas por depredador | Pasos hasta la primera captura | Energía final de depredadores | Supervivencia de depredadores | n |
|---|---|---|---|---|---|---|
| (a) abundancia | red neuronal | 6.828 ± 1.676 | 53.6 ± 16.2 | 0.526 ± 0.195 | 0.861 ± 0.249 | 90 |
| (a) abundancia | reglas | 5.950 ± 1.511 | 52.5 ± 17.9 | 0.626 ± 0.216 | 0.900 ± 0.242 | 30 |
| (a) abundancia | aleatorio | 0.000 ± 0.000 | 400.0 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 30 |
| (b) escasez | red neuronal | 5.039 ± 1.195 | 52.3 ± 16.1 | 0.361 ± 0.193 | 0.767 ± 0.320 | 90 |
| (b) escasez | reglas | 4.133 ± 1.252 | 52.8 ± 18.6 | 0.375 ± 0.212 | 0.767 ± 0.286 | 30 |
| (b) escasez | aleatorio | 0.017 ± 0.091 | 389.0 ± 60.4 | 0.000 ± 0.000 | 0.000 ± 0.000 | 30 |
| (c) presión+obst. | red neuronal | 3.662 ± 0.413 | 25.8 ± 14.7 | 0.153 ± 0.094 | 0.409 ± 0.196 | 90 |
| (c) presión+obst. | reglas | 3.700 ± 0.472 | 24.4 ± 15.1 | 0.273 ± 0.097 | 0.647 ± 0.180 | 30 |
| (c) presión+obst. | aleatorio | 0.000 ± 0.000 | 400.0 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 30 |

**Pruebas de Mann-Whitney (bilateral) — Capturas por depredador**

| Escenario | Comparación | Media A | Media B | Diferencia | p |
|---|---|---|---|---|---|
| (a) abundancia | red neuronal vs reglas | 6.828 | 5.950 | 0.878 | 0.00279 |
| (a) abundancia | red neuronal vs aleatorio | 6.828 | 0.000 | 6.828 | 1.33e-16 |
| (b) escasez | red neuronal vs reglas | 5.039 | 4.133 | 0.906 | 0.000563 |
| (b) escasez | red neuronal vs aleatorio | 5.039 | 0.017 | 5.022 | 1.2e-16 |
| (c) presión+obst. | red neuronal vs reglas | 3.662 | 3.700 | -0.038 | 0.339 |
| (c) presión+obst. | red neuronal vs aleatorio | 3.662 | 0.000 | 3.662 | 5.65e-17 |

**Pruebas de Mann-Whitney (bilateral) — Pasos hasta la primera captura**

| Escenario | Comparación | Media A | Media B | Diferencia | p |
|---|---|---|---|---|---|
| (a) abundancia | red neuronal vs reglas | 53.6 | 52.5 | 1.089 | 0.86 |
| (a) abundancia | red neuronal vs aleatorio | 53.6 | 400.0 | -346.4 | 1.65e-16 |
| (b) escasez | red neuronal vs reglas | 52.3 | 52.8 | -0.478 | 0.75 |
| (b) escasez | red neuronal vs aleatorio | 52.3 | 389.0 | -336.7 | 3.71e-16 |
| (c) presión+obst. | red neuronal vs reglas | 25.8 | 24.4 | 1.378 | 0.623 |
| (c) presión+obst. | red neuronal vs aleatorio | 25.8 | 400.0 | -374.2 | 1.63e-16 |

## E4 — Ecosistema completo (descriptivo)

| Escenario | Controlador | Vida media de presas | Supervivencia final de presas | Capturas totales por episodio | Alimento consumido por presa | Energía final de depredadores | Pasos hasta la primera captura | n |
|---|---|---|---|---|---|---|---|---|
| (a) abundancia | red_vs_red | 0.640 ± 0.076 | 0.299 ± 0.150 | 13.7 ± 2.951 | 9.679 ± 2.155 | 0.669 ± 0.189 | 36.7 ± 19.8 | 90 |
| (a) abundancia | reglas_vs_reglas | 0.671 ± 0.075 | 0.303 ± 0.139 | 11.9 ± 3.021 | 8.983 ± 1.511 | 0.626 ± 0.216 | 52.5 ± 17.9 | 30 |
| (b) escasez | red_vs_red | 0.506 ± 0.066 | 0.071 ± 0.077 | 11.3 ± 3.079 | 1.216 ± 0.279 | 0.378 ± 0.181 | 38.0 ± 18.0 | 90 |
| (b) escasez | reglas_vs_reglas | 0.544 ± 0.047 | 0.085 ± 0.088 | 8.267 ± 2.504 | 1.360 ± 0.408 | 0.375 ± 0.212 | 52.8 ± 18.6 | 30 |
| (c) presión+obst. | red_vs_red | 0.355 ± 0.090 | 0.047 ± 0.082 | 18.8 ± 1.691 | 4.145 ± 1.466 | 0.148 ± 0.095 | 17.8 ± 12.5 | 90 |
| (c) presión+obst. | reglas_vs_reglas | 0.408 ± 0.074 | 0.042 ± 0.072 | 18.5 ± 2.360 | 5.800 ± 1.290 | 0.273 ± 0.097 | 24.4 ± 15.1 | 30 |