"""Entorno 2D vectorizado: B mundos independientes se simulan en paralelo.

Cada mundo contiene plantas, presas, depredadores y (opcionalmente) obstáculos
circulares. Reglas:
  * Movimiento: la acción es un vector (vx, vy); su norma se recorta a 1 y se
    multiplica por la velocidad máxima de la especie. Las paredes recortan la
    posición y los obstáculos expulsan al agente hacia su borde.
  * Energía: cada paso cuesta base_cost + move_cost * velocidad^2. Con energía
    <= 0 el agente muere.
  * Alimentación: una presa a menos de eat_radius de una planta disponible la
    consume (+food_gain). La planta rebrota con probabilidad `regrow` por paso.
  * Captura: un depredador sin digestión pendiente a menos de capture_radius
    de una presa la captura (+prey_gain) y entra en digestión `cooldown` pasos.
  * Fin del episodio: al llegar a WORLD['steps'] o cuando no quedan presas.
"""
import numpy as np
from .config import WORLD, PREY, PRED

INF = 1e9


def _pair(A, T):
    """Componentes dx, dy y distancia^2 entre cada agente de A y cada objetivo de T."""
    dx = T[:, None, :, 0] - A[:, :, None, 0]
    dy = T[:, None, :, 1] - A[:, :, None, 1]
    return dx, dy, dx * dx + dy * dy


def _nearest(A, T, valid_t, sense, exclude_self=False):
    """Objetivo válido más cercano dentro del radio de percepción.
    A (B,Na,2), T (B,Nt,2), valid_t (B,Nt) -> rel (B,Na,2) dirección unitaria,
    dist_norm (B,Na) en [0,1] (1 = nada a la vista), idx (B,Na), found (B,Na)."""
    dx, dy, d2 = _pair(A, T)
    d2 = np.where(valid_t[:, None, :], d2, INF)
    if exclude_self:
        d2 = d2 + np.eye(A.shape[1])[None] * INF
    idx = d2.argmin(-1)[..., None]
    dmin = np.sqrt(np.take_along_axis(d2, idx, -1)[..., 0])
    found = dmin <= sense
    rx = np.take_along_axis(dx, idx, -1)[..., 0]
    ry = np.take_along_axis(dy, idx, -1)[..., 0]
    # dirección UNITARIA (no se atenúa al acercarse) + distancia normalizada aparte
    r = np.where(found[..., None], np.stack([rx, ry], -1) / np.maximum(dmin, 1e-6)[..., None], 0.0)
    dn = np.where(found, dmin / sense, 1.0)
    return r, dn, idx[..., 0], found


class Ecosystem:
    def __init__(self, scenarios, seeds):
        """scenarios: lista de dicts (uno por mundo); seeds: lista de enteros."""
        self.B = len(seeds)
        self.W = WORLD["size"]
        self.T = WORLD["steps"]
        self.scen = scenarios
        self.seeds = list(seeds)
        # generador para el azar durante el episodio (rebrote, deambular)
        self.rng = np.random.default_rng([int(s) for s in seeds])
        self.reset()

    # ------------------------------------------------------------ reset
    def reset(self):
        B, W = self.B, self.W
        nq, mp, mf, mo = WORLD["n_prey"], WORLD["max_pred"], WORLD["max_food"], WORLD["max_obst"]
        self.prey_pos = np.zeros((B, nq, 2)); self.pred_pos = np.zeros((B, mp, 2))
        self.food_pos = np.zeros((B, mf, 2)); self.obs_pos = np.zeros((B, mo, 2))
        self.obs_r = np.zeros((B, mo))
        self.food_active = np.zeros((B, mf), bool)
        self.pred_active = np.zeros((B, mp), bool)
        self.regrow = np.zeros(B)
        for b in range(B):
            # disposición inicial determinada SOLO por la semilla de ese mundo
            r = np.random.default_rng(self.seeds[b])
            sc = self.scen[b]
            no = sc["n_obst"]
            self.obs_pos[b, :no] = r.uniform(15, W - 15, (no, 2))
            self.obs_r[b, :no] = r.uniform(5, 9, no)
            self.food_pos[b] = r.uniform(2, W - 2, (mf, 2))
            self.food_active[b, :sc["n_food"]] = True
            self.prey_pos[b] = r.uniform(5, W - 5, (nq, 2))
            # los depredadores empiezan en un borde para dar margen a las presas
            self.pred_pos[b] = np.stack([r.uniform(5, W - 5, mp), r.choice([3.0, W - 3.0], mp)], -1)
            self.pred_active[b, :sc["n_pred"]] = True
            self.regrow[b] = sc["regrow"]
        self._resolve_obstacles(self.food_pos)
        self._resolve_obstacles(self.prey_pos)
        self._resolve_obstacles(self.pred_pos)
        self.food_avail = self.food_active.copy()
        self.prey_E = np.full((B, nq), PREY["energy0"]); self.prey_alive = np.ones((B, nq), bool)
        self.pred_E = np.full((B, mp), PRED["energy0"]); self.pred_alive = self.pred_active.copy()
        self.pred_cool = np.zeros((B, mp))
        self.prey_life = np.zeros((B, nq)); self.prey_food = np.zeros((B, nq))
        self.pred_caps = np.zeros((B, mp))
        self.pred_pursuit = np.zeros((B, mp))
        self.prey_caught = np.zeros((B, nq), bool)   # muerte por captura vs por hambre
        self.first_capture = np.full(B, float(self.T))
        self.capture_times = [[] for _ in range(B)]
        # señal de exploración: rumbo que deriva al azar (como un generador central de patrones)
        self.prey_theta = self.rng.uniform(0, 2 * np.pi, (B, nq))
        self.pred_theta = self.rng.uniform(0, 2 * np.pi, (B, mp))
        self.t = 0
        self.prey_dist = np.zeros(B)   # distancia total recorrida (diagnóstico)

    def _resolve_obstacles(self, P):
        for k in range(self.obs_r.shape[1]):
            c = self.obs_pos[:, k][:, None, :]
            r = self.obs_r[:, k][:, None, None]
            v = P - c
            d = np.linalg.norm(v, axis=-1, keepdims=True)
            inside = (d < r) & (r > 0)
            u = v / np.maximum(d, 1e-9)
            P[:] = np.where(inside, c + u * (r + 1e-3), P)

    # ------------------------------------------------------ observación
    def _obst_obs(self, A, sense):
        valid = self.obs_r > 0
        rel = self.obs_pos[:, None] - A[:, :, None]
        dc = np.sqrt((rel ** 2).sum(-1))
        ds = np.where(valid[:, None], dc - self.obs_r[:, None], INF)
        idx = ds.argmin(-1)
        dmin = np.take_along_axis(ds, idx[..., None], -1)[..., 0]
        found = dmin <= sense
        r = np.take_along_axis(rel, idx[..., None, None].repeat(2, -1), 2)[:, :, 0]
        r = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-6)
        r = np.where(found[..., None], r, 0.0)
        dn = np.where(found, np.clip(dmin / sense, 0, 1), 1.0)
        return r, dn

    def observe_prey(self):
        """12 entradas por presa:
        0-1 dirección unitaria a la planta disponible más cercana, 2 su distancia/radio (1 = no ve),
        3-4 dirección al depredador vivo más cercano, 5 su distancia,
        6 energía propia, 7-8 posición normalizada [-1,1] (proximidad a paredes),
        9-10 dirección al obstáculo más cercano, 11 distancia a su borde,
        12-13 señal de exploración (cos, sin de un rumbo que deriva al azar)."""
        s = PREY["sense"]
        fr, fd, _, _ = _nearest(self.prey_pos, self.food_pos, self.food_avail, s)
        pr, pd, _, _ = _nearest(self.prey_pos, self.pred_pos, self.pred_alive, s)
        orl, od = self._obst_obs(self.prey_pos, s)
        pos = self.prey_pos / self.W * 2 - 1
        expl = np.stack([np.cos(self.prey_theta), np.sin(self.prey_theta)], -1)
        return np.concatenate([fr, fd[..., None], pr, pd[..., None], self.prey_E[..., None],
                               pos, orl, od[..., None], expl], -1)

    def observe_pred(self):
        """12 entradas por depredador:
        0-1 dirección unitaria a la presa viva más cercana, 2 su distancia/radio,
        3 energía propia, 4 digestión en curso (1/0),
        5-6 dirección al otro depredador más cercano (evitar competir por la misma presa),
        7-8 posición normalizada, 9-10 dirección al obstáculo más cercano, 11 distancia al borde,
        12-13 señal de exploración (cos, sin de un rumbo que deriva al azar)."""
        s = PRED["sense"]
        qr, qd, _, _ = _nearest(self.pred_pos, self.prey_pos, self.prey_alive, s)
        orr, _, _, _ = _nearest(self.pred_pos, self.pred_pos, self.pred_alive, s, exclude_self=True)
        obr, od = self._obst_obs(self.pred_pos, s)
        pos = self.pred_pos / self.W * 2 - 1
        cool = (self.pred_cool > 0).astype(float)
        # registro de "acecho": cercanía a la presa más cercana (1 = encima, 0 = fuera de vista)
        self.pred_pursuit += (1.0 - qd) * self.pred_alive
        expl = np.stack([np.cos(self.pred_theta), np.sin(self.pred_theta)], -1)
        return np.concatenate([qr, qd[..., None], self.pred_E[..., None], cool[..., None], orr,
                               pos, obr, od[..., None], expl], -1)

    # ------------------------------------------------------------- paso
    @staticmethod
    def _clip_norm(a):
        n = np.linalg.norm(a, axis=-1, keepdims=True)
        return a / np.maximum(n, 1.0), np.minimum(n[..., 0], 1.0)

    def step(self, a_prey, a_pred):
        W = self.W
        # 1) movimiento (los muertos no se mueven)
        a_prey, sp_q = self._clip_norm(a_prey)
        a_pred, sp_p = self._clip_norm(a_pred)
        sp_q = sp_q * self.prey_alive; sp_p = sp_p * self.pred_alive
        old = self.prey_pos.copy()
        self.prey_pos = np.clip(self.prey_pos + a_prey * PREY["speed"] * self.prey_alive[..., None], 0, W)
        self.pred_pos = np.clip(self.pred_pos + a_pred * PRED["speed"] * self.pred_alive[..., None], 0, W)
        self._resolve_obstacles(self.prey_pos)
        self._resolve_obstacles(self.pred_pos)
        self.prey_dist += (np.linalg.norm(self.prey_pos - old, axis=-1) * self.prey_alive).sum(1)

        # 2) gasto de energía
        self.prey_E -= (PREY["base_cost"] + PREY["move_cost"] * sp_q ** 2) * self.prey_alive
        self.pred_E -= (PRED["base_cost"] + PRED["move_cost"] * sp_p ** 2) * self.pred_alive
        self.pred_cool = np.maximum(self.pred_cool - 1, 0)

        # 3) alimentación: cada planta la come la presa viva más cercana dentro del radio
        _, _, d = _pair(self.prey_pos, self.food_pos)       # distancia^2 (B,Nq,Nf)
        d = np.where(self.prey_alive[:, :, None] & self.food_avail[:, None, :], d, INF)
        who = d.argmin(1)                                    # (B,Nf)
        dmin = np.take_along_axis(d, who[:, None, :], 1)[:, 0]
        eaten = dmin < PREY["eat_radius"] ** 2
        onehot = (np.arange(d.shape[1])[None, :, None] == who[:, None, :]) & eaten[:, None, :]
        n_eat = onehot.sum(-1)
        self.prey_E = np.minimum(self.prey_E + n_eat * PREY["food_gain"], 1.0)
        self.prey_food += n_eat
        self.food_avail &= ~eaten

        # 4) captura: cada presa la atrapa el depredador apto más cercano
        able = self.pred_alive & (self.pred_cool <= 0)
        _, _, d = _pair(self.pred_pos, self.prey_pos)       # distancia^2 (B,Np,Nq)
        d = np.where(able[:, :, None] & self.prey_alive[:, None, :], d, INF)
        who = d.argmin(1)                                    # (B,Nq)
        dmin = np.take_along_axis(d, who[:, None, :], 1)[:, 0]
        caught = dmin < PRED["capture_radius"] ** 2
        onehot = (np.arange(d.shape[1])[None, :, None] == who[:, None, :]) & caught[:, None, :]
        n_cap = onehot.sum(-1)
        self.pred_E = np.minimum(self.pred_E + n_cap * PRED["prey_gain"], 1.0)
        self.pred_caps += n_cap
        self.pred_cool = np.where(n_cap > 0, PRED["cooldown"], self.pred_cool)
        self.prey_alive &= ~caught
        self.prey_caught |= caught
        for b in np.nonzero(caught.any(1))[0]:
            self.capture_times[b].extend([self.t + 1] * int(caught[b].sum()))
        newly = caught.any(1) & (self.first_capture >= self.T)
        self.first_capture[newly] = self.t + 1

        # 5) muerte por hambre
        self.prey_alive &= self.prey_E > 0
        self.pred_alive &= self.pred_E > 0

        # 6) rebrote de plantas
        grow = self.rng.random(self.food_avail.shape) < self.regrow[:, None]
        self.food_avail |= grow & self.food_active

        # 7) deriva de la señal de exploración
        self.prey_theta += self.rng.normal(0, 0.3, self.prey_theta.shape)
        self.pred_theta += self.rng.normal(0, 0.3, self.pred_theta.shape)

        self.prey_life += self.prey_alive
        self.t += 1

    def done(self):
        return self.t >= self.T or not self.prey_alive.any()

    # ---------------------------------------------------------- métricas
    def metrics(self):
        T = self.T
        pa = self.pred_active
        npred = np.maximum(pa.sum(1), 1)
        alive = self.prey_alive
        m = dict(
            prey_survival=alive.mean(1),                       # fracción viva al final
            prey_lifetime=self.prey_life.mean(1) / T,          # vida media normalizada
            food_eaten=self.prey_food.sum(1),
            food_per_prey=self.prey_food.mean(1),
            captures=self.pred_caps.sum(1),
            captures_per_pred=(self.pred_caps * pa).sum(1) / npred,
            prey_starved=((~alive) & (~self.prey_caught)).sum(1),
            prey_final_energy=np.where(alive.any(1), (self.prey_E * alive).sum(1) / np.maximum(alive.sum(1), 1), 0.0),
            pred_final_energy=(np.maximum(self.pred_E, 0) * pa).sum(1) / npred,
            pred_pursuit=(self.pred_pursuit * pa).sum(1) / npred / T,
            pred_survival=(self.pred_alive & pa).sum(1) / npred,
            time_to_first_capture=self.first_capture.copy(),
            mean_time_to_capture=np.array([np.mean(c) if c else float(T) for c in self.capture_times]),
            prey_distance=self.prey_dist / WORLD["n_prey"],
        )
        return m


def run_episode(scenarios, seeds, prey_ctrl, pred_ctrl, recorder=None):
    """Ejecuta B episodios en paralelo y devuelve el dict de métricas (arrays de B)."""
    env = Ecosystem(scenarios, seeds)
    prey_ctrl.reset(env.B, WORLD["n_prey"], env.rng)
    pred_ctrl.reset(env.B, WORLD["max_pred"], env.rng)
    while not env.done():
        oq, op = env.observe_prey(), env.observe_pred()
        rec = recorder is not None
        aq = prey_ctrl.act(oq, record=rec)
        ap = pred_ctrl.act(op, record=rec)
        if rec:
            recorder(env, oq, op, aq, ap, prey_ctrl, pred_ctrl)
        env.step(aq, ap)
    return env.metrics()
