"""
gpmpc.py — GP-MPC: Gaussian Process Model Predictive Control

Reference:
    L. Hewing, K. P. Wabersich, M. Menner, and M. N. Zeilinger,
    "Cautious Model Predictive Control using Gaussian Process Regression,"
    IEEE Transactions on Control Systems Technology, vol. 28, no. 6,
    pp. 2736–2743, 2020.

GP models the local traversability field: high σ² → unexplored/risky region →
adaptive safety margin tightening, analogous to CFed-DSMPC but without
learning from federated experience distributions.

Key improvement: GP is fitted on in-corridor points with traversability
labels (1=safe, 0=blocked). The uncertainty σ peaks at corridor boundaries,
naturally guiding each vehicle type to stay within its passable corridor.
"""

import numpy as np
from config import Config


class SparseGP:
    """Sparse GP regression with RBF kernel, greedy inducing-point selection."""

    def __init__(self, length_scale: float = 3.0, noise: float = 0.08,
                 n_inducing: int = 80):
        self.ls    = length_scale
        self.noise = noise
        self.M     = n_inducing
        self.Z     = None
        self.alpha = None
        self.Kzz_inv = None
        self._fitted = False

    def _rbf(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        diff = A[:, None, :] - B[None, :, :]
        return np.exp(-np.sum(diff ** 2, axis=-1) / (2 * self.ls ** 2))

    def _greedy_inducing(self, X: np.ndarray) -> np.ndarray:
        idx = [np.random.randint(len(X))]
        for _ in range(min(self.M, len(X)) - 1):
            dists = np.min(np.linalg.norm(
                X[:, None, :] - X[np.array(idx)][None, :, :], axis=-1), axis=1)
            idx.append(int(np.argmax(dists)))
        return X[idx]

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        self.Z = self._greedy_inducing(X) if n >= self.M else X.copy()
        Kzz  = self._rbf(self.Z, self.Z) + 1e-5 * np.eye(len(self.Z))
        Kxz  = self._rbf(X, self.Z)
        Q    = Kxz.T @ Kxz / self.noise ** 2
        A    = Kzz + Q + 1e-5 * np.eye(len(self.Z))
        self.Kzz_inv = np.linalg.inv(Kzz)
        self.alpha   = np.linalg.solve(A, Kxz.T @ y / self.noise ** 2)
        self._fitted = True

    def predict(self, X_star: np.ndarray):
        if not self._fitted:
            return np.zeros(len(X_star)), np.ones(len(X_star))
        Ksz      = self._rbf(X_star, self.Z)
        mean     = Ksz @ self.alpha
        v        = Ksz @ self.Kzz_inv
        var      = np.maximum(1.0 - np.sum(v * Ksz, axis=-1), 1e-8)
        return mean, np.sqrt(var)


class GPMPCController:
    """
    GP-MPC: Gaussian Process Model Predictive Control.
    Hewing, Wabersich, Menner & Zeilinger, IEEE TCST 2020.

    Adapted for the three-corridor trapezoid scene:
    The GP models traversability per cluster. High GP uncertainty σ(x,y)
    at a position means the cluster rarely explored there — either because
    it is inside an obstacle or is a geometrically incompatible corridor.

    Width-aware corridor guidance: like CARRL, we add an affinity cost
    toward the cluster-preferred y* during corridor traversal. This ensures
    GP-MPC routes each vehicle type to its passable corridor even when the
    GP uncertainty field is not perfectly calibrated.

    The combined cost is:
        J = J_base + GP_WEIGHT * Σ_h [(margin(s_h) - d_h)²/margin(s_h)²]₊
          + CORRIDOR_W * (y_H - y*)²  [inside corridor zone]

    Reference: Hewing et al., IEEE TCST 2020.
    """
    BETA       = 0.55
    DELTA_0    = 0.12
    GP_WEIGHT  = 90.0
    CORRIDOR_W = 45.0   # width-aware corridor affinity

    def __init__(self, env, v_params: dict, gp):
        from utils import VehicleModel
        self.env     = env
        self.model   = VehicleModel(**v_params)
        self.gp      = gp
        self.horizon = Config.HORIZON
        self.cluster = v_params.get('cluster_id', 0)
        self.preferred_y = Config.CLUSTER_TARGET_Y.get(self.cluster, 0.0)
        self._entry_x = {0: -10.0, 1: -7.0, 2: -9.0}.get(self.cluster, -10.0)

    def _gp_margin(self, x, y):
        pt = np.array([[x, y]], dtype=np.float32)
        _, sigma = self.gp.predict(pt)
        return self.DELTA_0 + self.BETA * float(sigma[0])

    def _corridor_affinity(self, curr, state_x):
        dy = curr[1] - self.preferred_y
        if state_x < self._entry_x - 3.0:
            return self.CORRIDOR_W * dy**2
        elif state_x < 10.0:
            return self.CORRIDOR_W * 0.7 * dy**2
        return 0.0

    def _base_cost(self, state_x, curr, target):
        dy = target[1] - curr[1]
        cost = -curr[0]*30.0 if state_x >= self._entry_x-2.0 else -curr[0]*20.0
        if curr[0] > 13.0:     cost += abs(dy)*60.0
        elif curr[0] > 10.0:   cost += abs(dy)*20.0
        if curr[0] > 21.0:     cost += (curr[0]-21.0)*50.0
        cost += abs(Config.TARGET_SPEED - curr[2])*20.0
        yn = (curr[3]+np.pi)%(2*np.pi)-np.pi
        cost += yn**2*25.0
        y_dev = abs(curr[1]-self.preferred_y)
        if state_x < self._entry_x-2.0:     cost += y_dev*10.0
        elif state_x < self._entry_x+5.0:   cost += y_dev*5.0
        else:                                cost += y_dev*1.0
        return cost

    def _wall_check(self, curr):
        return curr[1]>12.5 or curr[1]<-12.5 or curr[0]>24.0 or curr[0]<-24.0

    def get_action(self, state, target):
        best_cost = float('inf'); best_action = np.array([0.0, Config.MAX_ACCEL*0.5])
        steers = np.linspace(-Config.MAX_STEER, Config.MAX_STEER, 41)
        accels = [Config.MAX_ACCEL, Config.MAX_ACCEL*0.75, Config.MAX_ACCEL*0.5,
                  Config.MAX_ACCEL*0.25, 0.0, -Config.MAX_ACCEL*0.5]
        for s in steers:
            for a in accels:
                c = self._evaluate(s, a, state, target)
                if c < best_cost: best_cost = c; best_action = np.array([s, a])
        if best_cost >= 1e8:
            return np.array([0.0, -Config.MAX_ACCEL])
        return best_action

    def _evaluate(self, steer, accel, state, target):
        curr = state.copy(); cx = state[0]
        min_d = float('inf'); total_viol = 0.0; in_zone = cx <= 10.0
        for _ in range(self.horizon):
            curr = self.model.step(curr, [steer, accel])
            d = self.env.get_min_dist(curr[0],curr[1],curr[3],self.model.L,self.model.W)
            if d <= 0.05 or self._wall_check(curr): return 1e9
            if d < min_d: min_d = d
            if in_zone:
                margin = self._gp_margin(curr[0], curr[1])
                if d < margin:
                    total_viol += ((margin-d)/max(margin,1e-3))**2
        cost  = self._base_cost(cx, curr, target)
        if in_zone:
            cost += self.GP_WEIGHT * total_viol / self.horizon
            cost += self._corridor_affinity(curr, cx)
        return cost


def build_cluster_gps(env, collected_states: dict) -> dict:
    """
    Fit one SparseGP per cluster.
    Training signal: traversability = sigmoid(d_min * 5) in [0,1].
    High traversability (d_min large) → safe region, low uncertainty expected.
    Low traversability (d_min small, near obstacle) → risky, high σ.
    """
    gps: dict[int, SparseGP] = {}
    for cid in range(Config.NUM_CLUSTERS):
        pts = collected_states.get(cid, [])
        if len(pts) < 10:
            gps[cid] = SparseGP(); continue
        vp   = Config.VEHICLE_TYPES[cid]
        L, W = vp['L'], vp['W']
        XY   = np.array(pts, dtype=np.float32)

        # Traversability label: 1 = safe, 0 = tight
        trav = np.array([
            1.0 / (1.0 + np.exp(-5.0 * (
                env.get_min_dist(p[0], p[1], 0.0, L, W) - 0.3)))
            for p in XY
        ], dtype=np.float32)

        gp = SparseGP(length_scale=3.0, noise=0.08, n_inducing=80)
        gp.fit(XY, trav)
        gps[cid] = gp
        print(f"  GP cluster-{cid}: {len(XY)} pts, "
              f"trav=[{trav.min():.2f},{trav.max():.2f}]")
    return gps
