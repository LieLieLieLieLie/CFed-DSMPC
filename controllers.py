"""
controllers.py — All MPC / planning controllers

1. LinearMPC       — Garcia et al., Automatica 1989
2. RobustMPC       — Mayne et al., Automatica 2005
3. FedAvgMPC       — McMahan et al., AISTATS 2017
4. CARRLController — Everett et al., IEEE RA-L 2021 (scene-adapted)
5. GPMPCController — Hewing et al., IEEE TCST 2020  (in gpmpc.py)
6. ShiftAwareMPC   — Proposed CFed-DSMPC
"""

import numpy as np
from config import Config
from utils import VehicleModel


class BaseMPC:
    def __init__(self, env, v_params: dict):
        self.env         = env
        self.model       = VehicleModel(**v_params)
        self.horizon     = Config.HORIZON
        self.cluster     = v_params.get('cluster_id', 0)
        self.preferred_y = Config.CLUSTER_TARGET_Y.get(self.cluster, 0.0)
        self._entry_x    = {0: -10.0, 1: -7.0, 2: -9.0}.get(self.cluster, -10.0)

    def get_action(self, state, target):
        best_cost   = float('inf')
        best_action = np.array([0.0, Config.MAX_ACCEL * 0.5])
        steers = np.linspace(-Config.MAX_STEER, Config.MAX_STEER, 41)
        accels = [Config.MAX_ACCEL, Config.MAX_ACCEL*0.75, Config.MAX_ACCEL*0.5,
                  Config.MAX_ACCEL*0.25, 0.0, -Config.MAX_ACCEL*0.5]
        for s in steers:
            for a in accels:
                c = self.evaluate_trajectory(s, a, state, target)
                if c < best_cost:
                    best_cost = c; best_action = np.array([s, a])
        if best_cost >= 1e8:
            return np.array([0.0, -Config.MAX_ACCEL])
        return best_action

    def evaluate_trajectory(self, steer, accel, state, target):
        raise NotImplementedError

    def _base_cost(self, state_x, curr, target):
        dy = target[1] - curr[1]
        cost = -curr[0]*30.0 if state_x >= self._entry_x-2.0 else -curr[0]*20.0
        if curr[0] > 13.0:     cost += abs(dy)*60.0
        elif curr[0] > 10.0:   cost += abs(dy)*20.0
        if curr[0] > 21.0:     cost += (curr[0]-21.0)*50.0
        cost += abs(Config.TARGET_SPEED - curr[2])*20.0
        yaw_n = (curr[3]+np.pi)%(2*np.pi)-np.pi
        cost += yaw_n**2*25.0
        y_dev = abs(curr[1]-self.preferred_y)
        if state_x < self._entry_x-2.0:     cost += y_dev*10.0
        elif state_x < self._entry_x+5.0:   cost += y_dev*5.0
        else:                                cost += y_dev*1.0
        return cost

    def _wall_check(self, curr):
        return curr[1]>12.5 or curr[1]<-12.5 or curr[0]>24.0 or curr[0]<-24.0


# ── 1. Linear MPC ─────────────────────────────────────────────────────────────
class LinearMPC(BaseMPC):
    """Garcia, Prett & Morari, Automatica 1989."""
    def evaluate_trajectory(self, steer, accel, state, target):
        curr = state.copy()
        for _ in range(self.horizon):
            curr = self.model.step(curr, [steer, accel])
            d = self.env.get_min_dist(curr[0],curr[1],curr[3],self.model.L,self.model.W)
            if d <= 0.05 or self._wall_check(curr): return 1e9
        return self._base_cost(state[0], curr, target)


# ── 2. Robust MPC ─────────────────────────────────────────────────────────────
class RobustMPC(BaseMPC):
    """Mayne, Seron & Rakovic, Automatica 2005."""
    def evaluate_trajectory(self, steer, accel, state, target):
        curr = state.copy(); min_d = float('inf')
        for _ in range(self.horizon):
            curr = self.model.step(curr, [steer, accel])
            d = self.env.get_min_dist(curr[0],curr[1],curr[3],self.model.L,self.model.W)
            if d < min_d: min_d = d
            if d <= 0.05 or self._wall_check(curr): return 1e9
        cost = self._base_cost(state[0], curr, target)
        if min_d < Config.ROBUST_FIXED_MARGIN:
            cost += (Config.ROBUST_FIXED_MARGIN - min_d)*150.0
        return cost


# ── 3. FedAvg MPC ─────────────────────────────────────────────────────────────
class FedAvgMPC(BaseMPC):
    """McMahan et al., AISTATS 2017. Shared density model, no clustering."""
    def __init__(self, env, v_params, shared_den_model):
        super().__init__(env, v_params)
        self.den_model = shared_den_model

    def evaluate_trajectory(self, steer, accel, state, target):
        curr = state.copy(); cum = 0.0; in_zone = state[0] <= 10.0
        for _ in range(self.horizon):
            curr = self.model.step(curr, [steer, accel])
            d = self.env.get_min_dist(curr[0],curr[1],curr[3],self.model.L,self.model.W)
            if d <= 0.05 or self._wall_check(curr): return 1e9
            if in_zone: cum += self.den_model.compute_shift(curr, np.array([steer, accel]))
        cost = self._base_cost(state[0], curr, target)
        if in_zone:
            cost += min(cum/self.horizon, Config.MAX_SHIFT_PENALTY)*Config.SHIFT_WEIGHT
        return cost


# ── 4. CARRL (scene-adapted) ──────────────────────────────────────────────────
class CARRLController(BaseMPC):
    """
    Heterogeneous Collision-Avoidance Risk-field MPC.
    Everett, Chen & How, IEEE RA-L 2021.

    Adapted for the three-corridor trapezoid scene:
    (a) Directional risk: lateral obstacles penalised with width-scaled risk,
        forward obstacles use lighter penalty to preserve forward momentum.
    (b) Width-aware corridor routing: vehicle width W determines which gap is
        geometrically passable. The corridor affinity potential pulls each
        vehicle toward its width-compatible corridor entry during approach.
        Small Cars (W=1.2m) → upper gap (y≈7.5), gap=2.0m
        Medium Cars (W=1.8m) → middle gap (y≈0.0), gap=2.8m
        Large Trucks (W=2.4m) → lower gap (y≈-7.5), gap=3.8m
    (c) Width-adaptive scan radius avoids overloading wider vehicles in dense corridors.
    (d) Escape boost: reverse briefly instead of freezing when fully blocked.
    """
    W_LAT      = 70.0
    W_FWD      = 6.0
    R_EPS      = 0.05
    CORRIDOR_W = 50.0   # corridor affinity weight (increased for stronger routing)
    APPROACH_W = 30.0   # extra pull during pre-corridor approach phase

    def _scan_radius(self):
        return max(4.0, 7.0 - self.model.W * 0.8)

    def _risk_at(self, curr):
        x, y, _, yaw = curr
        cos_h, sin_h = np.cos(yaw), np.sin(yaw)
        W = self.model.W; rsc = self._scan_radius(); risk = 0.0
        for obs in self.env.obstacles:
            dx = obs['x']-x; dy = obs['y']-y; dc = np.hypot(dx, dy)
            if dc > rsc: continue
            d_lon =  dx*cos_h + dy*sin_h
            d_lat = -dx*sin_h + dy*cos_h
            clearance  = max(dc - obs['r'] - W/2.0, self.R_EPS)
            lat_clear  = max(abs(d_lat) - obs['r'] - W/2.0, self.R_EPS)
            if d_lon > 0: risk += self.W_FWD  * (W/clearance)**2
            else:         risk += self.W_LAT  * (W/lat_clear)**2
        return risk

    def _corridor_affinity(self, curr, state_x):
        """
        Width-aware corridor routing:
        During approach (x < entry_x) apply strong pull toward preferred_y.
        Inside corridor (entry_x <= x <= 10) apply standard affinity.
        """
        dy = curr[1] - self.preferred_y
        if state_x < self._entry_x - 3.0:
            # Approach: stronger pull to route vehicle to correct corridor
            return self.APPROACH_W * dy**2
        elif state_x < 10.0:
            return self.CORRIDOR_W * dy**2
        return 0.0

    def evaluate_trajectory(self, steer, accel, state, target):
        curr = state.copy(); total_risk = 0.0
        for _ in range(self.horizon):
            curr = self.model.step(curr, [steer, accel])
            d = self.env.get_min_dist(curr[0],curr[1],curr[3],self.model.L,self.model.W)
            if d <= 0.05 or self._wall_check(curr): return 1e9
            total_risk += self._risk_at(curr)
        cost  = self._base_cost(state[0], curr, target)
        cost += total_risk / self.horizon
        cost += self._corridor_affinity(curr, state[0])
        return cost

    def get_action(self, state, target):
        best_cost = float('inf'); best_action = np.array([0.0, Config.MAX_ACCEL*0.5])
        steers = np.linspace(-Config.MAX_STEER, Config.MAX_STEER, 41)
        accels = [Config.MAX_ACCEL, Config.MAX_ACCEL*0.75, Config.MAX_ACCEL*0.5,
                  Config.MAX_ACCEL*0.25, 0.0, -Config.MAX_ACCEL*0.5]
        for s in steers:
            for a in accels:
                c = self.evaluate_trajectory(s, a, state, target)
                if c < best_cost: best_cost = c; best_action = np.array([s, a])
        if best_cost >= 1e8:
            return np.array([0.0, -Config.MAX_ACCEL*0.7])
        return best_action


# ── 5. CFed-DSMPC (proposed) ──────────────────────────────────────────────────
class ShiftAwareMPC(BaseMPC):
    """Clustered Federated Distribution-Shift-Aware MPC (proposed)."""
    CORRIDOR_W = 55.0
    SAFETY_WEIGHT = 320.0
    MIN_ADAPTIVE_MARGIN = 0.35
    SHIFT_MARGIN_GAIN = 0.28

    def __init__(self, env, v_params, den_model):
        super().__init__(env, v_params)
        self.den_model = den_model

    def _corridor_affinity(self, curr, state_x):
        dy = curr[1] - self.preferred_y
        if state_x < self._entry_x - 3.0:
            return self.CORRIDOR_W * 1.15 * dy**2
        if state_x < 10.0:
            return self.CORRIDOR_W * dy**2
        return 0.0

    def evaluate_trajectory(self, steer, accel, state, target):
        curr = state.copy(); cx = state[0]
        cum = 0.0; min_d = float('inf'); safety_viol = 0.0
        in_zone = cx <= 10.0; past_entry = cx > self._entry_x
        for _ in range(self.horizon):
            curr = self.model.step(curr, [steer, accel])
            d = self.env.get_min_dist(curr[0],curr[1],curr[3],self.model.L,self.model.W)
            if d <= 0.05 or self._wall_check(curr): return 1e9
            if d < min_d: min_d = d
            if in_zone: cum += self.den_model.compute_shift(curr, np.array([steer,accel]))
        cost = self._base_cost(cx, curr, target)
        if in_zone:
            avg = cum/self.horizon
            cost += min(avg, Config.MAX_SHIFT_PENALTY)*Config.SHIFT_WEIGHT
            cost += self._corridor_affinity(curr, cx)
            if past_entry and min_d < float('inf'):
                st = max(Config.SAFETY_MARGIN, self.MIN_ADAPTIVE_MARGIN)
                st += min(max(avg, 0.0), Config.MAX_SHIFT_PENALTY) * self.SHIFT_MARGIN_GAIN
                if min_d < st:
                    safety_viol = ((st - min_d) / max(st, 1e-3))**2
                    cost += safety_viol * self.SAFETY_WEIGHT
        return cost
