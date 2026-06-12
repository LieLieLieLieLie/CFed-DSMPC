"""
metrics.py — Experiment metrics, saving, and visualisation

Design notes
────────────
• safety_comply: computed over MOVING steps only (speed > MOVE_THRESH).
  Stuck agents record safety_comply = -1 and are excluded from rate.
• effective_safety: success AND safety_comply >= 0.9
• No figure suptitles (top-level title suppressed per paper style)
• Colour palette: FF6666 / FFAA53 / 50CC55 / 3399FF / 6666FF / 9933FF
  (7-colour variant adds 00DDDD between 50CC55 and 3399FF)
"""

import os, json, csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from config import Config

plt.rcParams['hatch.linewidth'] = 2.2

RESULTS_DIR = Config.RESULTS_DIR
FIGURES_DIR = Config.FIGURES_DIR
TABLES_DIR = Config.TABLES_DIR
MOVE_THRESH = 0.05

# ── Colour palette ─────────────────────────────────────────────────────────
METHOD_COLORS = {
    'Linear MPC':        '#3399FF',
    'Robust MPC':        '#50CC55',
    'CARRL':             '#FFAA53',
    'FedAvg MPC':        '#6666FF',
    'GP-MPC':            '#9933FF',
    'CFed-DSMPC (Ours)': '#FF6666',
}
METHOD_MARKERS = {
    'Linear MPC':'o', 'Robust MPC':'s', 'CARRL':'^',
    'FedAvg MPC':'D', 'GP-MPC':'v', 'CFed-DSMPC (Ours)':'*',
}
METHOD_LS = {
    'Linear MPC':'-', 'Robust MPC':'--', 'CARRL':':',
    'FedAvg MPC':'-.', 'GP-MPC':(0,(3,1,1,1)), 'CFed-DSMPC (Ours)':'-',
}

# Ablation colours: Full=red, 3 ablations = light variants
ABL_COLORS = ['#FF6666', '#FFAA53', '#50CC55', '#6666FF']

# Blue→Red heatmap
BLUE_RED = LinearSegmentedColormap.from_list(
    'blue_red', ['#007FFF', '#FFFFFF', '#FF4F4F'], N=256)

SAFETY_MARGIN_BY_METHOD = {
    'Linear MPC':        0.05,
    'Robust MPC':        Config.ROBUST_FIXED_MARGIN,
    'CARRL':             0.10,
    'FedAvg MPC':        Config.SAFETY_MARGIN,
    'GP-MPC':            0.15,
    'CFed-DSMPC (Ours)': Config.SAFETY_MARGIN,
    'w/o Shift':   Config.SAFETY_MARGIN,
    'w/o Safety':  Config.SAFETY_MARGIN,
    'w/o CFL':     Config.SAFETY_MARGIN,
}

FS = dict(title=24, label=23, tick=23, legend=23)

# 7 colours for per-cluster metrics
METRIC_COLORS_7 = ['#FF6666','#FFAA53','#50CC55','#00DDDD','#3399FF','#6666FF','#9933FF']


def _display_method(method):
    return method.replace(' (Ours)', '')


def _save(fig, path, bbox_inches='tight'):
    # Always save as PDF (vector); replace any .png suffix just in case
    pdf_path = path.replace('.png', '.pdf')
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    fig.savefig(pdf_path, bbox_inches=bbox_inches, format='pdf')
    plt.close(fig)
    print(f"  [Plot] {pdf_path}")


def _spine(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', ls=':', alpha=0.4, lw=0.8)


# ─────────────────────────────────────────────────────────────────────────────
#  Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(result, env, v_params, method):
    traj    = np.array(result['traj'])
    L, W    = v_params['L'], v_params['W']
    cid     = v_params.get('cluster_id', 0)
    target_y= Config.CLUSTER_TARGET_Y.get(cid, 0.0)
    margin  = SAFETY_MARGIN_BY_METHOD.get(method, 0.1)

    diffs       = np.diff(traj[:,:2], axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))
    min_dists   = np.array([
        env.get_min_dist(traj[t,0],traj[t,1],traj[t,3],L,W)
        for t in range(len(traj))])
    speeds      = traj[:,2]
    moving_mask = speeds > MOVE_THRESH
    n_moving    = int(moving_mask.sum())

    # Safety compliance: fraction of moving steps that keep margin per method's threshold
    sc = float(np.mean(min_dists[moving_mask] > margin)) if n_moving > 0 else -1.0

    # Unified safety compliance at the nominal margin (0.2m) — method-agnostic, for EffSafe
    UNIFIED_MARGIN = Config.SAFETY_MARGIN  # 0.2 m, same for all methods
    sc_unified = float(np.mean(min_dists[moving_mask] > UNIFIED_MARGIN)) if n_moving > 0 else -1.0

    raw_min  = float(np.min(min_dists))
    disp_min = max(raw_min, 0.0)

    # Min obstacle distance — only meaningful inside the obstacle zone (x > -10).
    # Agents that are stuck outside the zone never engaged with obstacles, so their
    # large clearance is uninformative.  We compute the min distance restricted to
    # trajectory points that have entered the obstacle zone.  If no such point exists
    # (agent was fully stuck outside), we return -1 so the aggregate can exclude it.
    ZONE_ENTRY_X = -10.0
    zone_mask = traj[:, 0] > ZONE_ENTRY_X
    if zone_mask.sum() > 0:
        zone_min_dist = max(float(np.min(min_dists[zone_mask])), 0.0)
    else:
        zone_min_dist = -1.0  # never entered the zone; exclude from aggregate

    mid_mask = (traj[:,0] > -3.) & (traj[:,0] < 3.)
    corr     = (float(abs(float(np.mean(traj[mid_mask,1])) - target_y) < 4.0)
                if mid_mask.sum() > 0 else 0.0)
    yaw_diff = np.abs(np.diff(traj[:,3]))
    smooth   = float(np.mean(yaw_diff)) if len(yaw_diff) > 0 else 0.0

    succ = int(result['success'])
    stk  = int(result.get('stuck', False))
    coll = int(not result['success'] and not stk and raw_min <= 0.05)
    tout = int(not result['success'] and not stk and raw_min > 0.05)

    return {
        'success':          succ, 'collision': coll, 'stuck': stk, 'timeout': tout,
        'steps':            len(traj)-1, 'n_moving_steps': n_moving,
        'path_length':      round(path_length,3),
        'avg_min_dist':     round(float(np.mean(min_dists)),4),
        'overall_min_dist': round(disp_min,4),        # full trajectory (legacy)
        'zone_min_dist':    round(zone_min_dist,4),   # inside obstacle zone only
        'raw_min_dist':     round(raw_min,4),
        'safety_comply':    round(sc,4),
        'safety_comply_unified': round(sc_unified,4),  # method-agnostic, for EffSafe
        # Continuous effective-safety score: success gates the metric, while
        # safety compliance contributes smoothly instead of an all-or-nothing cutoff.
        'effective_safety': round(float(succ) * max(0.0, min(sc_unified / 0.9, 1.0)), 4),
        'corridor_correct': round(corr,4),
        'avg_speed':        round(float(np.mean(speeds[moving_mask])) if n_moving>0 else 0.,4),
        'smoothness':       round(smooth,6),
        'cluster_id': cid, 'method': method,
    }


def aggregate_metrics(per_agent):
    keys = ['success','collision','stuck','timeout','steps','path_length',
            'avg_min_dist','overall_min_dist','safety_comply','effective_safety',
            'corridor_correct','avg_speed','smoothness','n_moving_steps']
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_agent if m.get(k,-1) >= 0]
        agg[f'{k}_mean'] = round(float(np.mean(vals)),4) if vals else 0.
        agg[f'{k}_std']  = round(float(np.std(vals)),4)  if vals else 0.
    # zone_min_dist: exclude agents that never entered the obstacle zone (value == -1)
    zvals = [m['zone_min_dist'] for m in per_agent if m.get('zone_min_dist', -1) >= 0]
    agg['zone_min_dist_mean'] = round(float(np.mean(zvals)),4) if zvals else 0.
    agg['zone_min_dist_std']  = round(float(np.std(zvals)),4)  if zvals else 0.
    for r in ['success','collision','stuck','timeout','effective_safety','corridor_correct']:
        agg[f'{r}_rate'] = round(float(np.mean([m[r] for m in per_agent])),4)
    sc = [m['safety_comply'] for m in per_agent if m['safety_comply'] >= 0]
    agg['safety_comply_rate'] = round(float(np.mean(sc)),4) if sc else 0.
    agg['safety_comply_std']  = round(float(np.std(sc)),4)  if sc else 0.
    return agg


# ─────────────────────────────────────────────────────────────────────────────
#  File naming helper
# ─────────────────────────────────────────────────────────────────────────────

def make_tag(sweep_param='seed', val=None):
    """Generate a clean filename tag: paramName_value."""
    if val is None:
        mapping = {'seed': Config.SEED, 'env_seed': Config.ENV_SEED,
                   'explore_samples': Config.EXPLORE_SAMPLES,
                   'rounds': Config.ROUNDS, 'shift_weight': Config.SHIFT_WEIGHT,
                   'proximal_mu': Config.PROXIMAL_MU}
        val = mapping.get(sweep_param, 0)
    return f"{sweep_param}{val}"


# ─────────────────────────────────────────────────────────────────────────────
#  Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_metrics, tag):
    """Save metrics JSON + summary CSV. Files named by tag (no timestamps)."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    jp = os.path.join(TABLES_DIR, f'{tag}_metrics.json')
    with open(jp, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    methods = list(all_metrics.keys())
    if methods:
        agg = {m: aggregate_metrics(list(all_metrics[m].values())) for m in methods}
        cp  = os.path.join(TABLES_DIR, f'{tag}_summary.csv')
        fn  = ['method'] + list(list(agg.values())[0].keys())
        with open(cp, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            for m in methods:
                w.writerow({'method': m, **agg[m]})
    print(f"  [Saved] {tag}")
    return tag


def _json_ready(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def save_simulation_records(sim_results, starts, targets, collected_states, tag,
                            experiment='main'):
    """Save full trajectory-level simulation data for later re-plotting."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    suffix = 'trajectories' if experiment == 'main' else f'{experiment}_trajectories'
    json_path = os.path.join(TABLES_DIR, f'{tag}_{suffix}.json')
    csv_path = os.path.join(TABLES_DIR, f'{tag}_{suffix}.csv')

    payload = {
        'tag': tag,
        'experiment': experiment,
        'starts': starts,
        'targets': targets,
        'collected_states': collected_states,
        'results': sim_results,
    }
    with open(json_path, 'w') as f:
        json.dump(_json_ready(payload), f, indent=2)

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'method', 'agent_id', 'cluster_id', 'step',
                    'x', 'y', 'speed', 'yaw', 'success', 'stuck', 'min_dist'])
        for method, agents in sim_results.items():
            for agent_id, res in agents.items():
                cid = Config.AGENT_CLUSTERS[int(agent_id)]
                for step, state in enumerate(res.get('traj', [])):
                    x, y, speed, yaw = np.asarray(state, dtype=float)
                    w.writerow([experiment, method, int(agent_id), cid, step,
                                x, y, speed, yaw,
                                int(bool(res.get('success', False))),
                                int(bool(res.get('stuck', False))),
                                res.get('min_dist', '')])
    print(f"  [Saved trajectories] {json_path}")
    return json_path, csv_path


def save_aggregate_table(aggregate_results, tag, experiment):
    """Save aggregate metric dictionaries such as ablation summaries."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    json_path = os.path.join(TABLES_DIR, f'{tag}_{experiment}.json')
    csv_path = os.path.join(TABLES_DIR, f'{tag}_{experiment}.csv')
    with open(json_path, 'w') as f:
        json.dump(_json_ready(aggregate_results), f, indent=2)
    rows = list(aggregate_results.items())
    if rows:
        keys = sorted({k for _, vals in rows for k in vals.keys()})
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name'] + keys)
            w.writeheader()
            for name, vals in rows:
                w.writerow({'name': name.replace('\n', ' '), **vals})
    print(f"  [Saved table] {json_path}")
    return json_path, csv_path


# ─────────────────────────────────────────────────────────────────────────────
#  Summary: 7 panels in ONE row, NO suptitle
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary_bars(all_metrics, tag):
    body_fs = 24
    methods = list(all_metrics.keys())
    n = len(methods)
    agg = {m: aggregate_metrics(list(all_metrics[m].values())) for m in methods}
    colors = [METHOD_COLORS[m] for m in methods]
    short = [_display_method(m) for m in methods]

    fig, axes = plt.subplots(1, 6, figsize=(42, 7.6))
    axes = list(axes.flat)
    plt.subplots_adjust(wspace=0.34, bottom=0.27)

    ax = axes[0]
    succ = np.array([agg[m]['success_rate'] for m in methods])
    stk = np.array([agg[m]['stuck_rate'] for m in methods])
    col = np.array([agg[m]['collision_rate'] for m in methods])
    tout = np.array([agg[m]['timeout_rate'] for m in methods])
    x = np.arange(n)
    outcome_cfgs = [
        ('Success', succ, np.zeros_like(succ), '#50CC55', '//'),
        ('Stuck', stk, succ, '#FFAA53', '//'),
        ('Collision', col, succ+stk, '#FF6666', '//'),
        ('Timeout', tout, succ+stk+col, '#BBBBBB', '//'),
    ]
    outcome_handles = []
    old_hatch_lw = plt.rcParams.get('hatch.linewidth', 1.0)
    plt.rcParams['hatch.linewidth'] = 1.4
    for label, vals, bottom, color, hatch in outcome_cfgs:
        bars = ax.bar(x, vals, bottom=bottom, color=color, label=label,
                      ec='#111', lw=1.3, hatch=hatch)
        outcome_handles.append(bars[0])
    plt.rcParams['hatch.linewidth'] = old_hatch_lw
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    swatch_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for xi, color in zip(x, colors):
        ax.add_patch(plt.Rectangle((xi - 0.28, -0.105), 0.56, 0.055,
                                   transform=swatch_transform, clip_on=False,
                                   color=color, ec='#333', lw=0.7))
    ax.set_ylim(0, 1.22)
    ax.set_ylabel('Proportion', fontsize=body_fs)
    ax.set_title('Outcome Distribution', fontsize=FS['title'], fontweight='bold')
    ax.tick_params(labelsize=body_fs)
    _spine(ax)

    ax = axes[1]
    for i, m in enumerate(methods):
        ax.errorbar(i, agg[m]['safety_comply_rate'], yerr=agg[m].get('safety_comply_std', 0),
                    fmt=METHOD_MARKERS[m], color=METHOD_COLORS[m], markersize=13,
                    capsize=5, capthick=1.8, elinewidth=1.8)
    ax.axhline(1.0, color='#888', ls='--', lw=1, alpha=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Rate (moving steps)', fontsize=body_fs)
    ax.set_title('Safety Compliance\n(stuck agents excluded)', fontsize=FS['title'], fontweight='bold')
    _spine(ax)
    ax.tick_params(labelsize=body_fs)

    ax = axes[2]
    np.random.seed(0)
    for i, m in enumerate(methods):
        dists = []
        for v in all_metrics[m].values():
            zd = v.get('zone_min_dist', -1)
            dists.append(zd if zd >= 0 else v['overall_min_dist'])
        valid = [d for d in dists if d >= 0]
        if not valid:
            continue
        jit = np.random.uniform(-0.18, 0.18, len(valid))
        ax.scatter(np.full(len(valid), i)+jit, valid,
                   color=METHOD_COLORS[m], s=70, alpha=0.75,
                   marker=METHOD_MARKERS[m], zorder=3)
        ax.plot([i-0.32, i+0.32], [np.mean(valid)]*2,
                color=METHOD_COLORS[m], lw=3.0, zorder=4)
    ax.axhline(0, color='#FF4444', ls='--', lw=1.2, alpha=0.7, label='Contact')
    ax.set_xticks(range(n))
    ax.set_xticklabels([])
    ax.set_ylabel('Distance (m)', fontsize=body_fs)
    ax.set_title('Min Obstacle Distance\n(in-zone agents only)', fontsize=FS['title'], fontweight='bold')
    ax.legend(fontsize=body_fs)
    _spine(ax)
    ax.tick_params(labelsize=body_fs)

    ax = axes[3]
    agent_names = ['Small\nAgent', 'Medium\nAgent', 'Large\nAgent']
    bw = 0.12
    xb = np.arange(3)
    offsets = np.linspace(-(n-1)*bw/2, (n-1)*bw/2, n)
    for i, m in enumerate(methods):
        vals = []
        for cid in range(3):
            ag = [v for v in all_metrics[m].values() if v.get('cluster_id') == cid]
            vals.append(np.mean([a['corridor_correct'] for a in ag]) if ag else 0)
        ax.bar(xb+offsets[i], vals, bw, color=METHOD_COLORS[m], ec='#333', lw=0.6)
    ax.set_xticks(xb)
    ax.set_xticklabels(agent_names, fontsize=body_fs)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Accuracy', fontsize=body_fs)
    ax.set_title('Corridor Accuracy\nper Agent Type', fontsize=FS['title'], fontweight='bold')
    _spine(ax)
    ax.tick_params(labelsize=body_fs)

    # Path Length panel intentionally removed.

    ax = axes[4]
    sdata = [[v['smoothness'] for v in all_metrics[m].values()] for m in methods]
    has_var = any(len(set(d)) > 1 for d in sdata)
    if has_var:
        parts = ax.violinplot(sdata, positions=range(n), showmeans=True,
                              showextrema=True, widths=0.6)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.65)
        for part in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
            parts[part].set_colors('#333')
            parts[part].set_linewidth(1.2)
    else:
        sm_m = [np.mean(d) for d in sdata]
        sm_s = [np.std(d) for d in sdata]
        ax.bar(range(n), sm_m, yerr=sm_s, color=colors, ec='#333', lw=0.7,
               capsize=4, width=0.6)
    ax.set_xticks(range(n))
    ax.set_xticklabels([])
    ax.set_ylabel('|Delta yaw| (rad/step)', fontsize=body_fs)
    ax.set_title('Path Smoothness\n(lower = smoother)', fontsize=FS['title'], fontweight='bold')
    _spine(ax)
    ax.tick_params(labelsize=body_fs)

    ax = axes[5]
    eff = [agg[m]['effective_safety_rate'] for m in methods]
    bars = ax.bar(range(n), eff, color=colors, ec='#333', lw=0.7, width=0.6)
    for bar, val in zip(bars, eff):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=body_fs, fontweight='bold')
    ax.set_xticks(range(n))
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.28)
    ax.set_ylabel('Rate', fontsize=body_fs)
    ax.set_title('Effective Safety\n(success-weighted safety)', fontsize=FS['title'], fontweight='bold')
    _spine(ax)
    ax.tick_params(labelsize=body_fs)

    method_handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m], ec='#333', lw=0.6)
                      for m in methods]
    fig.legend(outcome_handles + method_handles,
               [cfg[0] for cfg in outcome_cfgs] + [_display_method(m) for m in methods],
               loc='lower center', bbox_to_anchor=(0.5, 0.08),
               ncol=len(outcome_handles) + len(method_handles),
               frameon=False, fontsize=body_fs, handlelength=1.15,
               columnspacing=1.25)

    _save(fig, os.path.join(FIGURES_DIR, f'{tag}_summary.pdf'))


def plot_ood_heatmap(env, server, tag):
    import torch
    title_fs = 35
    body_fs = 33
    xs = np.linspace(Config.X_MIN+1, Config.X_MAX-1, 90)
    ys = np.linspace(Config.Y_MIN+1, Config.Y_MAX-1, 70)
    XX, YY = np.meshgrid(xs, ys)
    agent_names = {0: 'Small Agent (Cluster 0)', 1: 'Medium Agent (Cluster 1)',
                   2: 'Large Agent (Cluster 2)'}
    fig, axes = plt.subplots(1, 3, figsize=(28, 8.8))
    dummy_a = np.zeros(2, dtype=np.float32)
    for cid, ax in enumerate(axes):
        model = server.global_den[cid]
        model.eval()
        Z = np.zeros_like(XX)
        with torch.no_grad():
            for j, y_ in enumerate(ys):
                for i, x_ in enumerate(xs):
                    s = np.array([x_, y_, Config.TARGET_SPEED, 0.], dtype=np.float32)
                    Z[j, i] = float(model.compute_shift(s, dummy_a))
        vmax = float(np.percentile(Z[np.isfinite(Z)], 95))
        vmin = float(np.percentile(Z[np.isfinite(Z)], 5))
        pcm = ax.pcolormesh(XX, YY, Z, cmap=BLUE_RED, vmin=vmin, vmax=vmax, shading='gouraud')
        cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        if cid == 2:
            cb.set_label(r'Reconstruction Error $\Delta$', fontsize=body_fs)
        cb.ax.tick_params(labelsize=body_fs)
        for obs in env.obstacles:
            if obs['type'] != 'wall':
                ax.add_patch(plt.Circle((obs['x'], obs['y']), obs['r'],
                                        color='#37474F', alpha=0.65, zorder=3))
        ty = Config.CLUSTER_TARGET_Y[cid]
        ax.axhline(ty, color='white', ls='--', lw=1.8, alpha=0.9)
        ax.set_xlim(Config.X_MIN+1, Config.X_MAX-1)
        ax.set_ylim(Config.Y_MIN+1, Config.Y_MAX-1)
        ax.set_title(f'{agent_names[cid]}\nCorridor y={ty}', fontsize=title_fs, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=body_fs)
        if cid == 0:
            ax.set_ylabel('Y Position (m)', fontsize=body_fs, labelpad=-20)
        else:
            ax.set_ylabel('')
        ax.tick_params(labelsize=body_fs)
    plt.tight_layout()
    _save(fig, os.path.join(FIGURES_DIR, f'{tag}_ood_heatmap.pdf'))

# keep legacy name for compatibility
plot_safety_heatmap = plot_ood_heatmap


def plot_per_cluster(all_metrics, tag):
    body_fs = 32
    title_fs = 35
    methods = list(all_metrics.keys())
    method_labels = [_display_method(m) for m in methods]
    agent_names = {0: 'Small Agent', 1: 'Medium Agent', 2: 'Large Agent'}
    metrics = [
        ('success_rate', 'Success', 'high'),
        ('safety_comply_rate', 'Safety*', 'high'),
        ('effective_safety_rate', 'EffSafe', 'high'),
        ('corridor_correct_rate', 'Corridor', 'high'),
        ('zone_min_dist_mean', 'MinDist', 'high_norm'),
        ('smoothness_mean', 'Smooth', 'low_norm'),
    ]
    metric_colors = ['#50CC55', '#FFAA53', '#FF6666', '#3399FF', '#6666FF', '#9933FF']

    fig, axes = plt.subplots(1, 3, figsize=(30, 11.2), sharex=True)
    plt.subplots_adjust(wspace=0.08, bottom=0.20, left=0.17, right=0.98)

    y = np.arange(len(methods))
    bh = 0.11
    offsets = np.linspace(-(len(metrics)-1)*bh/2, (len(metrics)-1)*bh/2, len(metrics))

    for cid, ax in enumerate(axes):
        agg_by_method = {}
        for m in methods:
            agents = [v for v in all_metrics[m].values() if v.get('cluster_id') == cid]
            agg_m = aggregate_metrics(agents) if agents else {}
            agg_by_method[m] = {key: agg_m.get(key, 0.) for key, _, _ in metrics}

        ranges = {}
        for key, _, direction in metrics:
            vals = [agg_by_method[m][key] for m in methods]
            ranges[key] = (min(vals), max(vals), direction)

        for metric_idx, (key, label, direction) in enumerate(metrics):
            vals = []
            for m in methods:
                v = agg_by_method[m][key]
                lo, hi, _ = ranges[key]
                if direction == 'high_norm':
                    v = v / max(hi, 1e-6)
                elif direction == 'low_norm':
                    v = 1.0 if hi - lo < 1e-8 else 1.0 - (v - lo) / (hi - lo)
                vals.append(v)
            ax.barh(y + offsets[metric_idx], vals, bh,
                    color=metric_colors[metric_idx], ec='#333', lw=0.45,
                    label=label)

        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel('Normalised Score', fontsize=body_fs)
        ax.set_yticks(y)
        if cid == 0:
            ax.set_yticklabels(method_labels, fontsize=body_fs)
        else:
            ax.set_yticklabels([])
        ax.set_title(agent_names[cid], fontsize=title_fs, fontweight='bold')
        ax.invert_yaxis()
        _spine(ax)
        ax.tick_params(labelsize=body_fs)

    handles = [plt.Rectangle((0, 0), 1, 1, color=metric_colors[i], ec='#333', lw=0.5)
               for i in range(len(metrics))]
    labels = [label for _, label, _ in metrics]
    fig.legend(handles, labels, fontsize=body_fs, loc='lower center',
               bbox_to_anchor=(0.5, 0.035), frameon=False,
               handlelength=1.2, ncol=6, columnspacing=1.6)

    _save(fig, os.path.join(FIGURES_DIR, f'{tag}_per_cluster.pdf'))


def plot_ablation(ablation_results, tag):
    body_fs = 30
    variants = list(ablation_results.keys())
    metric_cfgs = [
        ('success_rate', 'SR'),
        ('safety_comply_rate', r'SC$^*$'),
        ('zone_min_dist_mean', 'ZoneMin'),
        ('corridor_correct_rate', 'CA'),
    ]
    fig, ax = plt.subplots(1, 1, figsize=(20.0, 10.0))
    plt.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.18)

    group_y = np.arange(len(metric_cfgs)) * 3.80
    bar_h = 0.90
    offsets = (np.arange(len(variants)) - (len(variants) - 1) / 2) * bar_h
    metric_best = {
        key: max(ablation_results[variant].get(key, 0) for variant in variants)
        for key, _ in metric_cfgs
    }
    for vidx, variant in enumerate(variants):
        vals = [ablation_results[variant].get(key, 0) for key, _ in metric_cfgs]
        bars = ax.barh(group_y + offsets[vidx], vals, height=bar_h * 0.78,
                       color=ABL_COLORS[vidx], ec='#333', lw=0.8,
                       label=_display_method(variant.replace('\n', ' ')))
        for bar, val, (key, _) in zip(bars, vals, metric_cfgs):
            ax.text(val + 0.018, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}' if val < 0.1 else f'{val:.2f}',
                    ha='left', va='center', fontsize=body_fs,
                    fontweight='bold' if np.isclose(val, metric_best[key]) else 'normal')

    ax.set_yticks(group_y)
    ax.set_yticklabels([label for _, label in metric_cfgs],
                       fontsize=body_fs, fontweight='normal',
                       rotation=90, va='center')
    ax.set_xlabel('')
    ax.set_xlim(0, 1.18)
    ax.set_xticks(np.linspace(0, 1.0, 6))
    ax.invert_yaxis()
    ax.set_ylim(group_y[-1] + offsets[-1] + bar_h * 0.58,
                group_y[0] + offsets[0] - bar_h * 0.58)
    _spine(ax)
    ax.grid(axis='x', ls=':', alpha=0.45, lw=0.9)
    ax.grid(axis='y', visible=False)
    ax.tick_params(axis='x', labelsize=body_fs)

    handles = [plt.Rectangle((0, 0), 1, 1, color=ABL_COLORS[i], ec='#333', lw=0.8)
               for i in range(len(variants))]
    labels = [_display_method(v.replace('\n', ' ')) for v in variants]
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.545, 0.045),
               ncol=4, frameon=False, fontsize=body_fs,
               handlelength=1.0, columnspacing=0.55, labelspacing=0.6)

    _save(fig, os.path.join(FIGURES_DIR, f'{tag}_ablation.pdf'), bbox_inches=None)


def plot_federated_analysis(rounds_results, mu_results, tag):
    body_fs = 31
    title_fs = body_fs + 2
    fig, axes = plt.subplots(1, 2, figsize=(21, 8.0))
    plt.subplots_adjust(wspace=0.28, bottom=0.36)
    methods_track = ['FedAvg MPC', 'CFed-DSMPC (Ours)']
    line_cfgs = [
        ('success_rate', 'Success Rate', '-', 'o'),
        ('safety_comply_rate', 'Safety Compliance', '--', 's'),
    ]
    legend_handles = []
    legend_labels = []

    ax = axes[0]
    r_vals = sorted(rounds_results.keys())
    for m in methods_track:
        for key, label, ls, mk in line_cfgs:
            vals = [rounds_results[r].get(m, {}).get(key, 0) for r in r_vals]
            line, = ax.plot(r_vals, vals, color=METHOD_COLORS[m], lw=2.8, ls=ls,
                            marker=mk, markersize=11)
            legend_handles.append(line)
            legend_labels.append(f"{_display_method(m)} ({label})")
    ax.set_xlabel('Communication Rounds', fontsize=body_fs, labelpad=12)
    ax.set_ylabel('Rate', fontsize=body_fs)
    ax.set_title('FL Convergence vs. Communication Rounds',
                 fontsize=title_fs, fontweight='bold', pad=18)
    ax.set_xticks(r_vals)
    ax.tick_params(labelsize=body_fs)
    ax.grid(ls=':', alpha=0.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    mu_vals = sorted(mu_results.keys())
    for m in methods_track:
        for key, label, ls, mk in line_cfgs:
            vals = [mu_results[mu].get(m, {}).get(key, 0) for mu in mu_vals]
            ax.plot(mu_vals, vals, color=METHOD_COLORS[m], lw=2.8, ls=ls,
                    marker=mk, markersize=11)
    ax.set_xlabel('Proximal Coefficient mu (stronger privacy)', fontsize=body_fs, labelpad=12)
    ax.set_ylabel('Rate', fontsize=body_fs)
    ax.set_title('Privacy-Utility Trade-off (FedProx mu)',
                 fontsize=title_fs, fontweight='bold', pad=18)
    ax.set_xscale('log')
    ax.tick_params(labelsize=body_fs)
    ax.grid(ls=':', alpha=0.45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.legend(legend_handles, legend_labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.015), ncol=2,
               frameon=False, fontsize=body_fs, handlelength=2.2,
               columnspacing=1.15, labelspacing=0.28)
    _save(fig, os.path.join(FIGURES_DIR, f'{tag}_federated.pdf'))


def plot_sweep_lines(sweep_results, sweep_param, tag):
    if not sweep_results: return
    param_vals = sorted(sweep_results.keys())
    methods    = list(list(sweep_results.values())[0].keys())
    metric_cfgs = [
        ('success_rate',          'Success Rate'),
        ('overall_min_dist_mean', 'Min Obstacle Dist (m)'),
        ('safety_comply_rate',    'Safety Compliance'),
        ('corridor_correct_rate', 'Corridor Accuracy'),
        ('path_length_mean',      'Path Length (m)'),
        ('smoothness_mean',       'Path Smoothness'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (key, ylabel) in zip(axes.flat, metric_cfgs):
        for m in methods:
            vals = [sweep_results[pv][m].get(key,0) for pv in param_vals]
            ax.plot(param_vals, vals, color=METHOD_COLORS.get(m,'#999'),
                    lw=2.2, marker=METHOD_MARKERS.get(m,'o'),
                    ls=METHOD_LS.get(m,'-'), markersize=8,
                    label=_display_method(m))
        ax.set_xlabel(sweep_param, fontsize=FS['label'])
        ax.set_ylabel(ylabel, fontsize=FS['label'])
        ax.set_title(ylabel, fontsize=FS['title'], fontweight='bold')
        ax.tick_params(labelsize=FS['tick'])
        ax.legend(fontsize=FS['legend']-1)
        ax.grid(ls=':', alpha=0.45)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    _save(fig, os.path.join(FIGURES_DIR, f'{tag}_sweep_{sweep_param}.pdf'))


# ─────────────────────────────────────────────────────────────────────────────
#  Console table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(all_metrics):
    methods = list(all_metrics.keys())
    agg = {m: aggregate_metrics(list(all_metrics[m].values())) for m in methods}
    cols = [('success_rate','Succ%'),('collision_rate','Coll%'),
            ('stuck_rate','Stuck%'),('timeout_rate','Tout%'),
            ('safety_comply_rate','Safe%*'),('effective_safety_rate','EffSafe%'),
            ('corridor_correct_rate','Corr%'),('overall_min_dist_mean','MinDist'),
            ('zone_min_dist_mean','ZoneMin'),
            ('path_length_mean','PathLen'),('smoothness_mean','Smooth')]
    hdr = f"{'Method':<22}" + "".join(f"{h:>10}" for _,h in cols)
    print("\n"+"="*len(hdr))
    print(hdr)
    print("  (* Safe% excludes stuck agents;  EffSafe% = success AND safe≥0.9 @0.2m unified;  ZoneMin = in-zone only)")
    print("-"*len(hdr))
    for m in methods:
        print(f"{_display_method(m):<22}"+"".join(f"{agg[m].get(k,0):>10.3f}" for k,_ in cols))
    print("="*len(hdr))
