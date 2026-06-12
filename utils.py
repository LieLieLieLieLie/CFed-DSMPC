import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import os
from config import Config

# ─────────────────────────────────────────────
#  Vehicle kinematics (bicycle model)
# ─────────────────────────────────────────────
class VehicleModel:
    def __init__(self, L, W, mass, **kwargs):
        self.L, self.W = L, W

    def step(self, state, control):
        x, y, v, yaw = state
        steer   = float(np.clip(control[0], -Config.MAX_STEER, Config.MAX_STEER))
        accel   = float(np.clip(control[1], -Config.MAX_ACCEL,  Config.MAX_ACCEL))
        beta    = np.arctan(0.5 * np.tan(steer))
        x_new   = x   + v * np.cos(yaw + beta) * Config.DT
        y_new   = y   + v * np.sin(yaw + beta) * Config.DT
        v_new   = max(0.0, v + (accel - Config.DRAG * v) * Config.DT)
        yaw_new = yaw + (v / max(self.L, 1e-3)) * np.sin(beta) * Config.DT
        return np.array([x_new, y_new, v_new, yaw_new])


# ─────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────
class CrossingEnv:
    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.obstacles: list[dict] = []

        for x in np.linspace(Config.X_MIN, Config.X_MAX, 120):
            self.obstacles.append({'x': x, 'y':  13.2, 'r': 0.45, 'type': 'wall'})
            self.obstacles.append({'x': x, 'y': -13.2, 'r': 0.45, 'type': 'wall'})
        for y in np.linspace(Config.Y_MIN + 1, Config.Y_MAX - 1, 56):
            self.obstacles.append({'x': Config.X_MIN + 0.1, 'y': y, 'r': 0.3, 'type': 'wall'})
            self.obstacles.append({'x': Config.X_MAX - 0.1, 'y': y, 'r': 0.3, 'type': 'wall'})

        self._place_trapezoid_obstacles(
            rng, y_range=(3.5, 12.0), top_span=(-13.0, 13.0), bot_span=(-10.0, 10.0),
            r_mean=0.38, r_std=0.01, col_spacing=1.7, gap=2.0, gate_cols=3,
            wave_amp=0.8, wave_freq=0.35, target_y=7.5, label='obs_upper')
        self._place_gate_pillars(
            gate_center_y=7.5, gate_width=2.0, pillar_r=0.42,
            x_positions=[-9.8, -9.1, -8.4, 8.4, 9.1, 9.8],
            top_span=(-13.0, 13.0), bot_span=(-10.0, 10.0),
            y_lo=3.5, y_hi=12.0, label='obs_upper')
        self._place_trapezoid_obstacles(
            rng, y_range=(-4.0, 3.5), top_span=(-8.0, 8.0), bot_span=(-7.0, 7.0),
            r_mean=0.50, r_std=0.01, col_spacing=2.3, gap=2.8, gate_cols=3,
            wave_amp=0.8, wave_freq=0.35, target_y=0.0, label='obs_middle')
        self._place_gate_pillars(
            gate_center_y=0.0, gate_width=2.35, pillar_r=0.50,
            x_positions=[-6.8, -6.1, -5.4, -4.7, 4.7, 5.4, 6.1, 6.8],
            top_span=(-8.0, 8.0), bot_span=(-7.0, 7.0),
            y_lo=-4.0, y_hi=3.5, label='obs_middle')
        self._place_trapezoid_obstacles(
            rng, y_range=(-12.0, -4.0), top_span=(-13.0, 13.0), bot_span=(-4.0, 4.0),
            r_mean=0.60, r_std=0.01, col_spacing=2.9, gap=3.8, gate_cols=2,
            wave_amp=0.8, wave_freq=0.35, target_y=-7.5, label='obs_lower')
        self._place_gate_pillars(
            gate_center_y=-7.5, gate_width=4.5, pillar_r=0.60,
            x_positions=[-3.8, -3.0, 3.0, 3.8],
            top_span=(-13.0, 13.0), bot_span=(-4.0, 4.0),
            y_lo=-12.0, y_hi=-4.0, label='obs_lower')

        self.obs_matrix = np.array([[o['x'], o['y'], o['r']] for o in self.obstacles])

    @staticmethod
    def _trapezoid_y_bounds(x, top_span, bot_span, y_lo, y_hi):
        xtL, xtR = top_span
        xbL, xbR = bot_span
        v_min = y_lo
        v_max = y_hi
        if x < xbL:
            slope_L = (y_hi - y_lo) / (xtL - xbL)
            v_min   = y_lo + (x - xbL) * slope_L
        if x > xbR:
            slope_R = (y_hi - y_lo) / (xtR - xbR)
            v_min   = y_lo + (x - xbR) * slope_R
        if x < xtL or x > xtR:
            return None, None
        return v_min, v_max

    def _place_gate_pillars(self, gate_center_y, gate_width, pillar_r,
                            x_positions, top_span, bot_span, y_lo, y_hi, label):
        half   = gate_width / 2.0
        gap_lo = gate_center_y - half
        gap_hi = gate_center_y + half
        for xp in x_positions:
            v_min, v_max = self._trapezoid_y_bounds(xp, top_span, bot_span, y_lo, y_hi)
            if v_min is None:
                continue
            y = v_min + pillar_r
            while y < gap_lo - pillar_r * 0.5:
                self.obstacles.append({'x': xp, 'y': y, 'r': pillar_r, 'type': label})
                y += pillar_r * 2.0 + 0.05
            if v_min < gap_lo - pillar_r * 0.1:
                self.obstacles.append({'x': xp, 'y': gap_lo - pillar_r, 'r': pillar_r, 'type': label})
            if gap_hi + pillar_r < v_max:
                self.obstacles.append({'x': xp, 'y': gap_hi + pillar_r, 'r': pillar_r, 'type': label})
            y = gap_hi + pillar_r * 2.0 + 0.05
            while y < v_max - pillar_r * 0.5:
                self.obstacles.append({'x': xp, 'y': y, 'r': pillar_r, 'type': label})
                y += pillar_r * 2.0 + 0.05

    def _place_trapezoid_obstacles(self, rng, y_range, top_span, bot_span, r_mean, r_std,
                                   col_spacing, gap, gate_cols,
                                   wave_amp, wave_freq, target_y, label):
        y_lo, y_hi = y_range
        xtL, xtR   = top_span
        all_cols   = []
        x = xtL + col_spacing / 2.0
        while x < xtR:
            all_cols.append(x)
            x += col_spacing
        for col_idx, x in enumerate(all_cols):
            v_min, v_max = self._trapezoid_y_bounds(x, top_span, bot_span, y_lo, y_hi)
            if v_min is None or v_min >= v_max - 0.5:
                continue
            dist_from_left  = col_idx
            dist_from_right = len(all_cols) - 1 - col_idx
            edge_dist   = min(dist_from_left, dist_from_right)
            wave_factor = min(1.0, max(0.0, (edge_dist - gate_cols + 1) / 3.0))
            path_y  = target_y + np.sin((x - xtL) * wave_freq) * wave_amp * wave_factor
            path_y  = float(np.clip(path_y, v_min + gap / 2.0, v_max - gap / 2.0))
            gap_lo  = path_y - gap / 2.0
            gap_hi  = path_y + gap / 2.0
            is_gate = (dist_from_left < gate_cols) or (dist_from_right < gate_cols)
            jitter  = 0.04 if is_gate else 0.10
            y = v_min + r_mean * 0.5
            while y < v_max:
                r     = float(np.clip(rng.normal(r_mean, r_std), r_mean * 0.92, r_mean * 1.08))
                obs_y = float(np.clip(y + rng.uniform(-jitter, jitter), v_min, v_max))
                obs_x = float(np.clip(x + rng.uniform(-jitter, jitter), xtL, xtR))
                if not (obs_y - r < gap_hi and obs_y + r > gap_lo):
                    self.obstacles.append({'x': obs_x, 'y': obs_y, 'r': r, 'type': label})
                y += 2 * r + rng.uniform(0.25, 0.55)

    def get_min_dist(self, x, y, yaw, L, W) -> float:
        R = W / 2.0
        d = max(0.0, L / 2.0 - W / 2.0)
        centers = np.array([
            [x, y],
            [x + d * np.cos(yaw), y + d * np.sin(yaw)],
            [x - d * np.cos(yaw), y - d * np.sin(yaw)],
        ])
        obs_xy, obs_r = self.obs_matrix[:, :2], self.obs_matrix[:, 2]
        diff  = centers[:, np.newaxis, :] - obs_xy[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2) - obs_r[np.newaxis, :] - R
        return float(np.min(dists))


# ─────────────────────────────────────────────
#  Plot constants
# ─────────────────────────────────────────────
OBS_COLORS = {
    'wall':       '#455A64',
    'obs_upper':  '#78909C',
    'obs_middle': '#8D6E63',
    'obs_lower':  '#A1887F',
}
CLUSTER_COLORS = {0: '#3399FF', 1: '#FFAA53', 2: '#FF6666'}

# Methods that actually USE a learned density/GP model → show experience scatter
DENSITY_METHODS = {'FedAvg MPC', 'GP-MPC', 'CFed-DSMPC (Ours)'}

# Font sizes
FS_TITLE  = 38   # panel title
FS_AXIS   = 38   # axis label
FS_TICK   = 33   # tick label
FS_LEGEND = 36   # legend


def _display_method(method):
    return method.replace(' (Ours)', '')


def _add_trapezoid_patches(ax):
    upper_poly  = np.array([[-13, 12], [13, 12], [10, 3.5], [-10, 3.5]])
    middle_poly = np.array([[-8, 3.5], [8, 3.5], [7, -4.0], [-7, -4.0]])
    lower_poly  = np.array([[-13, -4.0], [13, -4.0], [4, -12.0], [-4, -12.0]])
    ax.add_patch(patches.Polygon(upper_poly,  closed=True, color='#BBDEFB', alpha=0.30, zorder=0))
    ax.add_patch(patches.Polygon(middle_poly, closed=True, color='#FFF9C4', alpha=0.24, zorder=0))
    ax.add_patch(patches.Polygon(lower_poly,  closed=True, color='#FCE4EC', alpha=0.20, zorder=0))


def _draw_vehicle_box(ax, cx, cy, yaw, L, W, color):
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    hw, hl = W / 2.0, L / 2.0
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    R  = np.array([[cos_y, sin_y], [-sin_y, cos_y]])
    wc = corners @ R + [cx, cy]
    ax.add_patch(plt.Polygon(wc, closed=True, color=color, alpha=0.5, ec='k', lw=1.5))


def _draw_env_panel(ax, env, method, results, starts, targets,
                    collected_states, is_first_col: bool, is_bottom_row: bool):
    ax.set_facecolor('#F0F4F8')
    ax.set_xlim(Config.X_MIN - 1, Config.X_MAX + 1)
    ax.set_ylim(Config.Y_MIN - 1, Config.Y_MAX + 1)
    ax.set_title(_display_method(method), fontweight='bold', fontsize=FS_TITLE, pad=18)

    _add_trapezoid_patches(ax)

    # Background experience scatter ONLY for density-model methods
    if method in DENSITY_METHODS:
        for cid, states in collected_states.items():
            if states:
                pts = np.array(states)
                ax.scatter(pts[:, 0], pts[:, 1], s=4,
                           color=CLUSTER_COLORS[cid], alpha=0.12, zorder=1)

    for obs in env.obstacles:
        c = OBS_COLORS.get(obs['type'], '#90A4AE')
        ax.add_patch(patches.Circle((obs['x'], obs['y']), obs['r'],
                                    color=c, alpha=0.75, zorder=2))

    ax.scatter(starts[0][0], starts[0][1], marker='P', c='#4CAF50', s=220,
               edgecolors='black', linewidths=1.2, zorder=5, label='Start')
    ax.scatter(targets[0][0], targets[0][1], marker='*', c='#FFD700', s=380,
               edgecolors='black', linewidths=1.2, zorder=5, label='Goal')

    if method in results and results[method]:
        drawn_labels = set()
        for i, res in results[method].items():
            cid  = Config.AGENT_CLUSTERS[i]
            v    = Config.VEHICLE_TYPES[cid]
            traj = np.array(res['traj'])
            label = v['name'] if v['name'] not in drawn_labels else '_nolegend_'
            ls    = '-' if i % 2 == 0 else '--'
            ax.plot(traj[:, 0], traj[:, 1], color=v['color'],
                    lw=3.2, ls=ls, alpha=0.9, zorder=4, label=label)
            drawn_labels.add(v['name'])
            if not res['success']:
                ax.scatter(traj[-1, 0], traj[-1, 1],
                           marker='X', color='red', s=200, zorder=6)
            _draw_vehicle_box(ax, traj[-1, 0], traj[-1, 1], traj[-1, 3],
                              v['L'], v['W'], v['color'])

    ax.grid(True, ls=':', alpha=0.4)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    if not is_first_col:
        ax.tick_params(axis='y', labelleft=False)
    if not is_bottom_row:
        ax.tick_params(axis='x', labelbottom=False)
    if is_first_col:
        ax.set_ylabel('Y Position (m)', fontsize=FS_AXIS, labelpad=-20)
    if is_bottom_row:
        ax.set_xlabel('X Position (m)', fontsize=FS_AXIS, labelpad=-2)


def _draw_expl_panel(ax, env, cid, collected_states, is_bottom: bool, is_top: bool):
    cluster_names = {0: 'Small Agent', 1: 'Medium Agent', 2: 'Large Agent'}
    ax.set_facecolor('#F8F9FA')
    ax.set_title(f"Experience: {cluster_names[cid]}",
                 fontsize=FS_TITLE, fontweight='bold', pad=18)
    _add_trapezoid_patches(ax)
    for obs in env.obstacles:
        ax.add_patch(patches.Circle((obs['x'], obs['y']), obs['r'],
                                    color='#CFD8DC', alpha=0.5, zorder=2))
    states = collected_states.get(cid, [])
    if states:
        pts = np.array(states)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color=CLUSTER_COLORS[cid],
                   alpha=0.65, zorder=3)
    ax.set_xlim(Config.X_MIN - 1, Config.X_MAX + 1)
    ax.set_ylim(Config.Y_MIN - 1, Config.Y_MAX + 1)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    if not is_bottom:
        ax.tick_params(axis='x', labelbottom=False)
    if is_bottom:
        ax.set_xlabel('X Position (m)', fontsize=FS_AXIS)
    ax.set_ylabel('Y Position (m)', fontsize=FS_AXIS, labelpad=6)
    ax.yaxis.set_label_coords(-0.11, 0.5)


# ─────────────────────────────────────────────
#  Main plotting function
#
#  Layout  (method_names length == 6, row-major):
#
#   col:   0            1            2          3 (experience)
#  row 0: method[0]   method[1]   method[2]   expl cluster-0
#  row 1: method[3]   method[4]   method[5]   expl cluster-1
#                                              expl cluster-2  (3rd sub-row)
# ─────────────────────────────────────────────
def plot_experiment_1(env, results, starts, targets,
                      collected_states, method_names: list, tag: str = 'seed2'):
    assert len(method_names) == 6

    # ── Figure & outer GridSpec ───────────────────────────────────────────
    fig = plt.figure(figsize=(50, 25))
    outer = gridspec.GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[1.5, 1.5, 1.5, 1.0],
        wspace=0.15,
    )
    left_gs = gridspec.GridSpecFromSubplotSpec(
        3, 3,
        subplot_spec=outer[0, 0:3],
        height_ratios=[1.74, 1.74, 0.06],
        hspace=0.15,
        wspace=0.15,
    )

    # Left 2×3 comparison panels
    traj_axes = []
    for row in range(2):
        for col in range(3):
            traj_axes.append(fig.add_subplot(left_gs[row, col]))
    legend_ax = fig.add_subplot(left_gs[2, :])
    legend_ax.axis('off')

    # Right column: 3 stacked experience panels (span both outer rows)
    right_gs = gridspec.GridSpecFromSubplotSpec(
        3, 1,
        subplot_spec=outer[0, 3],
        hspace=0.18,
    )
    expl_axes = [fig.add_subplot(right_gs[k]) for k in range(3)]

    # ── Draw left comparison panels ───────────────────────────────────────
    for idx, (ax, method) in enumerate(zip(traj_axes, method_names)):
        row = idx // 3
        col = idx  % 3
        _draw_env_panel(
            ax, env, method, results, starts, targets, collected_states,
            is_first_col=(col == 0),
            is_bottom_row=(row == 1),
        )

    # ── Draw right experience panels ──────────────────────────────────────
    for cid, ax in enumerate(expl_axes):
        _draw_expl_panel(ax, env, cid, collected_states,
                         is_bottom=(cid == 2), is_top=(cid == 0))

    legend_handles = [
        plt.Line2D([0], [0], marker='P', color='none', markerfacecolor='#4CAF50',
                   markeredgecolor='black', markersize=18, label='Start'),
        plt.Line2D([0], [0], marker='*', color='none', markerfacecolor='#FFD700',
                   markeredgecolor='black', markersize=22, label='Goal'),
    ]
    for cid in range(3):
        v = Config.VEHICLE_TYPES[cid]
        legend_handles.append(
            plt.Line2D([0], [0], color=v['color'], lw=4.0, label=v['name'])
        )
    legend_ax.legend(handles=legend_handles, loc='lower center', ncol=5,
                     fontsize=FS_LEGEND, frameon=False,
                     handlelength=1.9, columnspacing=1.2,
                     borderaxespad=0.0, bbox_to_anchor=(0.5, -4))

    save_path = os.path.join(Config.FIGURES_DIR, f'{tag}_traj.pdf')
    plt.savefig(save_path, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"  [Plot saved → {save_path}]")
