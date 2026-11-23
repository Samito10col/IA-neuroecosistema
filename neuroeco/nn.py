"""Red neuronal multicapa (MLP) implementada con NumPy.

Cada genoma es un vector plano con todos los pesos y sesgos. Se trabaja en
lote: B genomas distintos (uno por mundo simulado) se evalúan a la vez.

Arquitectura: entrada -> oculta1 (tanh) -> oculta2 (tanh) -> salida (tanh)
"""
import numpy as np


class MLPSpec:
    def __init__(self, n_in, hidden, n_out):
        self.sizes = [n_in, *hidden, n_out]
        self.shapes = []
        for a, b in zip(self.sizes[:-1], self.sizes[1:]):
            self.shapes.append((a, b))   # W
            self.shapes.append((b,))     # bias
        self.n_params = int(sum(np.prod(s) for s in self.shapes))

    def init(self, rng, n):
        """n genomas con inicialización tipo Xavier (escala 1/sqrt(fan_in))."""
        g = np.empty((n, self.n_params))
        i = 0
        for s in self.shapes:
            k = int(np.prod(s))
            if len(s) == 2:
                g[:, i:i + k] = rng.normal(0, 1 / np.sqrt(s[0]), (n, k))
            else:
                g[:, i:i + k] = 0.0
            i += k
        return g

    def unpack(self, genomes):
        """genomes (B, P) -> lista [(W (B,a,b), b (B,1,b)), ...]"""
        B = genomes.shape[0]
        out, i = [], 0
        for s in self.shapes:
            k = int(np.prod(s))
            arr = genomes[:, i:i + k]
            if len(s) == 2:
                out.append(arr.reshape(B, *s))
            else:
                out.append(arr.reshape(B, 1, s[0]))
            i += k
        return [(out[j], out[j + 1]) for j in range(0, len(out), 2)]

    def forward(self, layers, x, return_hidden=False):
        """x (B, N, n_in) -> (B, N, n_out). Opcionalmente devuelve activaciones ocultas."""
        h = x
        acts = []
        for W, b in layers:
            h = np.tanh(np.matmul(h, W) + b)
            acts.append(h)
        if return_hidden:
            return h, acts[:-1]
        return h
