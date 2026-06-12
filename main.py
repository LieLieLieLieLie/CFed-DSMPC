"""
main.py — CFed-DSMPC experiment runner

PyCharm direct-run: configure RUN_* flags at the top, then click Run.
Command-line:
    python main.py                         # main experiment only
    python main.py run_ablation            # ablation (loads saved models)
    python main.py run_federated           # FL analysis (loads saved models)
    python main.py run_all                 # ablation + federated
    python main.py --load                  # skip training
    python main.py --sweep seed            # generalisation sweep
    python main.py --sweep explore_samples
    python main.py --sweep shift_weight

File naming convention (no timestamps):
    results/seed2_metrics.json            main experiment
    results/seed2_summary.csv
    results/seed2_summary.png
    results/seed2_ood_heatmap.png
    results/seed2_ablation.png
    results/seed2_federated.png
    results/federated_rounds_metrics.json  (rounds sweep combined)
    results/federated_rounds_summary.csv
    results/federated_mu_metrics.json      (mu sweep combined)
    results/federated_mu_summary.csv
"""

import copy, sys, os, argparse, time
import numpy as np
import torch

from config import Config
from utils import CrossingEnv, plot_experiment_1
from federated import FedServer, FedAvgServer, FedClient
from controllers import (LinearMPC, RobustMPC, FedAvgMPC,
                         CARRLController, ShiftAwareMPC)
from gpmpc import GPMPCController, build_cluster_gps
from metrics import (compute_metrics, aggregate_metrics, save_results,
                     save_simulation_records, save_aggregate_table,
                     plot_summary_bars, plot_per_cluster,
                     plot_sweep_lines, plot_ablation,
                     plot_ood_heatmap, plot_federated_analysis,
                     print_summary_table, make_tag)


def _fmt(sec: float) -> str:
    """Format elapsed seconds as  Xm Ys  or  Xs."""
    m, s = divmod(int(sec), 60)
    return f"{m}m {s}s" if m else f"{s}s"


SAVE_DIR     = Config.MODELS_DIR
METHOD_NAMES = ['Linear MPC', 'Robust MPC', 'CARRL',
                'FedAvg MPC', 'GP-MPC', 'CFed-DSMPC (Ours)']
FL_TRACK     = ['FedAvg MPC', 'CFed-DSMPC (Ours)']  # methods tracked in FL analysis

# ── PyCharm direct-run configuration ──────────────────────────────────────────
RUN_ABLATION   = True   # set False to skip ablation
RUN_FEDERATED  = True   # set False to skip FL analysis
LOAD_MODELS    = False  # set True to skip training

# ── Sweep configurations ──────────────────────────────────────────────────────
SWEEP_CONFIGS = {
    'seed':            [0, 1, 2, 3, 4],
    'explore_samples': [200, 400, 800, 1600],
    'rounds':          [2, 5, 10, 20],
    'shift_weight':    [60, 120, 180, 240],
    'proximal_mu':     [0.01, 0.05, 0.1, 0.2],
}

# FL analysis sweep values (smaller for speed)
FL_ROUNDS_VALS = [2, 5, 10, 20]
FL_MU_VALS     = [0.01, 0.05, 0.1, 0.2]


# ─────────────────────────────────────────────────────────────────────────────
#  Simulation runner
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(controller, env, start, target, v_params, name,
                   agent_idx, silent=False):
    state = copy.deepcopy(start); traj = [start.copy()]
    success = stuck = False
    cid = v_params.get('cluster_id', 0)
    py  = Config.CLUSTER_TARGET_Y.get(cid, float(start[1]))

    if cid == 0:
        wps = [np.array([-14., py, 0., 0.], np.float32), target]
    elif cid == 1:
        wps = [np.array([-19., py, 0., 0.], np.float32),
               np.array([-12., py, 0., 0.], np.float32),
               np.array([ 16., target[1], 0., 0.], np.float32), target]
    else:
        wps = [np.array([-20., py, 0., 0.], np.float32),
               np.array([-11., py, 0., 0.], np.float32),
               np.array([ 16., py, 0., 0.], np.float32), target]
    wi = 0; ph = []

    for step in range(Config.SIM_STEPS):
        if not silent:
            sys.stdout.write(
                f"\r  [{name}] Agent {agent_idx+1}/{Config.NUM_AGENTS}"
                f" | Step {step+1:3d}/{Config.SIM_STEPS} ")
            sys.stdout.flush()
        cw = wps[wi]
        if wi < len(wps)-1:
            at = ((cw[0]>=10. and state[0]>=cw[0]-0.5) or
                  (cw[0]<10.  and state[0]>=cw[0]-0.5 and abs(state[1]-cw[1])<2.5))
            if at: wi = min(wi+1, len(wps)-1); cw = wps[wi]
        u     = controller.get_action(state, cw)
        state = controller.model.step(state, u)
        traj.append(state.copy())
        d = env.get_min_dist(state[0], state[1], state[3], v_params['L'], v_params['W'])
        if d <= 0.05: break
        if state[0] > 19. and abs(state[1]-target[1]) < 4.:
            success = True; break
        ph.append(state[:2].copy())
        if len(ph) > 40: ph.pop(0)
        if len(ph)==40:
            if np.sum(np.linalg.norm(np.diff(np.array(ph),axis=0),axis=1)) < 0.5:
                stuck = True; break

    if not silent:
        print(f" → {'✓' if success else ('⚠ Stuck' if stuck else '✗')}")
    ta  = np.array(traj)
    md  = float(np.min([env.get_min_dist(ta[t,0],ta[t,1],ta[t,3],
                                         v_params['L'],v_params['W'])
                        for t in range(len(ta))]))
    return {'traj': traj, 'success': success, 'stuck': stuck, 'min_dist': md}


# ─────────────────────────────────────────────────────────────────────────────
#  Controller factory
# ─────────────────────────────────────────────────────────────────────────────
def build_ctl(method, env, vp, server, fa_server, cluster_gps):
    cid = vp['cluster_id']
    if method=='Linear MPC':       return LinearMPC(env, vp)
    if method=='Robust MPC':       return RobustMPC(env, vp)
    if method=='CARRL':            return CARRLController(env, vp)
    if method=='FedAvg MPC':       return FedAvgMPC(env, vp, fa_server.global_den)
    if method=='GP-MPC':           return GPMPCController(env, vp, cluster_gps[cid])
    return ShiftAwareMPC(env, vp, server.global_den[cid])


# ─────────────────────────────────────────────────────────────────────────────
#  Training — Phase 1 (exploration) is shared; FL phases are re-trainable
# ─────────────────────────────────────────────────────────────────────────────
def _do_exploration(env, verbose=True):
    """Phase 1: Embodied exploration — returns (clients, server, fa_server)."""
    server = FedServer(); fa = FedAvgServer()
    clients = [FedClient(i, Config.AGENT_CLUSTERS[i]) for i in range(Config.NUM_AGENTS)]
    t0 = time.time()
    if verbose: print("  Phase 1: Exploration")
    for idx, c in enumerate(clients):
        cid = c.cluster_id; vp = Config.VEHICLE_TYPES[cid]
        cl = Config.EXPLORE_CLEARANCE[cid]; ty = Config.CLUSTER_TARGET_Y[cid]
        valid = attempts = 0
        if verbose:
            sys.stdout.write(f"  Agent {idx+1} ({vp['name']}) ... "); sys.stdout.flush()
        while valid < Config.EXPLORE_SAMPLES and attempts < Config.EXPLORE_ATTEMPTS:
            attempts += 1
            x = np.random.uniform(Config.X_MIN+2, Config.X_MAX-2)
            y = (float(np.clip(np.random.normal(ty,2.5),Config.Y_MIN+1,Config.Y_MAX-1))
                 if np.random.rand()<0.6 else
                 np.random.uniform(Config.Y_MIN+1,Config.Y_MAX-1))
            yaw = np.random.uniform(-np.pi/5, np.pi/5)
            v   = np.random.uniform(0.5, Config.TARGET_SPEED+0.5)
            if env.get_min_dist(x,y,yaw,vp['L'],vp['W']) > cl:
                s = np.array([x,y,v,yaw], np.float32)
                a = np.random.uniform([-Config.MAX_STEER,-Config.MAX_ACCEL],
                                      [Config.MAX_STEER, Config.MAX_ACCEL]).astype(np.float32)
                c.add_data(s, a); valid += 1
        if verbose: print(f"collected {valid}.")
    if verbose: print(f"  Phase 1 done — {_fmt(time.time()-t0)}")
    return clients, server, fa


def _do_fl_training(clients, server, fa_server, verbose=True):
    """Phase 2a+2b: FL training on existing clients."""
    t1 = time.time()
    if verbose: sys.stdout.write("  Phase 2a: Clustered FL ... "); sys.stdout.flush()
    for r in range(Config.ROUNDS):
        for c in clients: c.train(server.global_den[c.cluster_id].state_dict())
        for k in range(Config.NUM_CLUSTERS): server.aggregate(clients, k)
    if verbose: print(f"done ({_fmt(time.time()-t1)})")
    t2 = time.time()
    if verbose: sys.stdout.write("  Phase 2b: FedAvg ... "); sys.stdout.flush()
    for r in range(Config.ROUNDS):
        for c in clients: c.train(fa_server.global_den.state_dict())
        fa_server.aggregate(clients)
    if verbose: print(f"done ({_fmt(time.time()-t2)})")
    return server, fa_server


def train_all(env, verbose=True):
    if verbose: print("\n" + "="*50)
    clients, server, fa = _do_exploration(env, verbose)
    server, fa          = _do_fl_training(clients, server, fa, verbose)
    t3 = time.time()
    if verbose: sys.stdout.write("  Phase 2c: GP fitting ... "); sys.stdout.flush()
    gps = build_cluster_gps(env, server.collected_states)
    if verbose: print(f"done ({_fmt(time.time()-t3)})")
    return server, fa, gps, clients


# ─────────────────────────────────────────────────────────────────────────────
#  Save / Load
# ─────────────────────────────────────────────────────────────────────────────
def save_models(server, fa, gps, clients=None):
    os.makedirs(SAVE_DIR, exist_ok=True)
    for k, m in server.global_den.items():
        torch.save(m.state_dict(),
                   os.path.join(SAVE_DIR, f'clustered_fl_cluster{k}.pt'))
    torch.save(fa.global_den.state_dict(), os.path.join(SAVE_DIR, 'fedavg.pt'))
    torch.save(gps,  os.path.join(SAVE_DIR, 'cluster_gps.pt'))
    torch.save(server.collected_states, os.path.join(SAVE_DIR, 'collected_states.pt'))
    # Save exploration data so federated analysis can be reproduced via --load
    if clients is not None:
        clients_data = [
            {'client_id':  c.id,
             'cluster_id': c.cluster_id,
             'states':  [item[0] for item in c.data_buffer],
             'actions': [item[1] for item in c.data_buffer]}
            for c in clients
        ]
        torch.save(clients_data, os.path.join(SAVE_DIR, 'clients_data.pt'))
        n_samples = sum(len(c.data_buffer) for c in clients)
        print(f'  Exploration data saved ({n_samples} samples)')
    print(f"  Models saved \u2192 '{SAVE_DIR}/'")


def load_models():
    server = FedServer(); fa = FedAvgServer()
    for k in range(Config.NUM_CLUSTERS):
        server.global_den[k].load_state_dict(torch.load(
            os.path.join(SAVE_DIR, f'clustered_fl_cluster{k}.pt'),
            map_location=Config.DEVICE, weights_only=True))
    fa.global_den.load_state_dict(torch.load(
        os.path.join(SAVE_DIR, 'fedavg.pt'),
        map_location=Config.DEVICE, weights_only=True))
    gps = torch.load(os.path.join(SAVE_DIR, 'cluster_gps.pt'),
                     map_location='cpu', weights_only=False)
    cs  = torch.load(os.path.join(SAVE_DIR, 'collected_states.pt'),
                     map_location='cpu', weights_only=False)
    server.collected_states = {int(k): v for k, v in cs.items()}

    # Reconstruct FedClient list from saved exploration data (if available)
    clients = None
    clients_path = os.path.join(SAVE_DIR, 'clients_data.pt')
    if os.path.exists(clients_path):
        clients_data = torch.load(clients_path, map_location='cpu', weights_only=False)
        clients = []
        for cd in clients_data:
            c = FedClient(cd['client_id'], cd['cluster_id'])
            for s, a in zip(cd['states'], cd['actions']):
                c.add_data(np.array(s, dtype=np.float32),
                           np.array(a, dtype=np.float32))
            clients.append(c)
        total = sum(len(c.data_buffer) for c in clients)
        print(f'  Exploration data loaded ({total} samples, {len(clients)} agents)')
    else:
        print('  [Warning] clients_data.pt not found — federated analysis will re-explore')

    print('  Models loaded.')
    return server, fa, gps, clients


def _retrain_fl_only(clients, orig_states, rounds, mu, verbose=False):
    """
    Fast FL re-training for sweep experiments:
    Reuse Phase-1 exploration data (clients already have data_buffer),
    just reset FL model weights and re-run rounds.
    This is ~3x faster than full train_all() because Phase 1 is skipped.
    """
    import copy as _copy
    from models import DensityEstimator
    # Fresh server/fa with same architecture
    new_server = FedServer(); new_fa = FedAvgServer()
    orig_rounds = Config.ROUNDS; orig_mu = Config.PROXIMAL_MU
    Config.ROUNDS = rounds; Config.PROXIMAL_MU = mu
    new_server, new_fa = _do_fl_training(clients, new_server, new_fa, verbose=verbose)
    # Copy exploration states for GP fitting
    new_server.collected_states = orig_states
    Config.ROUNDS = orig_rounds; Config.PROXIMAL_MU = orig_mu
    return new_server, new_fa


# ─────────────────────────────────────────────────────────────────────────────
#  One experiment round
# ─────────────────────────────────────────────────────────────────────────────
def run_one_experiment(env, server, fa, gps, methods=None,
                       save_plot=True, silent=False, tag='seed2',
                       save_records=False, experiment='main'):
    if methods is None: methods = METHOD_NAMES
    speeds  = [1.8,2.2,1.8,2.2,1.8,2.2]
    starts  = [np.array([-23.,9.,speeds[i],0.],np.float32) for i in range(Config.NUM_AGENTS)]
    targets = [np.array([22.,9.,0.,0.],np.float32) for _ in range(Config.NUM_AGENTS)]
    sim = {m:{} for m in methods}; met = {m:{} for m in methods}

    for method in methods:
        t_m = time.time()
        if not silent: print(f"\n  ── {method} ──")
        for i in range(Config.NUM_AGENTS):
            cid = Config.AGENT_CLUSTERS[i]
            vp  = dict(Config.VEHICLE_TYPES[cid]); vp['cluster_id'] = cid
            ctl = build_ctl(method, env, vp, server, fa, gps)
            res = run_simulation(ctl, env, starts[i], targets[i], vp, method, i, silent)
            sim[method][i] = res
            met[method][i] = compute_metrics(res, env, vp, method)
        if not silent: print(f"  └─ {method} done ({_fmt(time.time()-t_m)})")
    if save_plot and methods == METHOD_NAMES:
        plot_experiment_1(env, sim, starts, targets,
                          server.collected_states, METHOD_NAMES, tag=tag)
    if save_records:
        save_simulation_records(sim, starts, targets, server.collected_states,
                                tag=tag, experiment=experiment)
    return sim, met


# ─────────────────────────────────────────────────────────────────────────────
#  Experiment 3: Ablation
# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(env, server, fa, gps, tag):
    print("\n" + "="*50 + "\n  Exp 3: Ablation Study\n" + "="*50)
    speeds  = [1.8,2.2,1.8,2.2,1.8,2.2]
    starts  = [np.array([-23.,9.,speeds[i],0.],np.float32) for i in range(Config.NUM_AGENTS)]
    targets = [np.array([22.,9.,0.,0.],np.float32) for _ in range(Config.NUM_AGENTS)]
    abl = {}

    def _run(name, ctl_fn):
        t_a = time.time()
        print(f"  Variant: {name.replace(chr(10),' ')}", end=' ', flush=True)
        ms = []
        variant_sim = {name: {}}
        for i in range(Config.NUM_AGENTS):
            cid = Config.AGENT_CLUSTERS[i]
            vp  = dict(Config.VEHICLE_TYPES[cid]); vp['cluster_id'] = cid
            ctl = ctl_fn(env, vp, cid)
            res = run_simulation(ctl, env, starts[i], targets[i], vp, name, i, True)
            variant_sim[name][i] = res
            ms.append(compute_metrics(res, env, vp, name))
        res = aggregate_metrics(ms)
        exp_name = 'ablation_' + name.lower().replace(' ', '_').replace('/', '_')
        save_simulation_records(variant_sim, starts, targets, server.collected_states,
                                tag=tag, experiment=exp_name)
        print(f"({_fmt(time.time()-t_a)})")
        return res

    # D) Full model
    _, m = run_one_experiment(env, server, fa, gps, save_plot=False, silent=True,
                              save_records=True, tag=tag,
                              experiment='ablation_full')
    abl['CFed-DSMPC\n(Full)'] = aggregate_metrics(list(m['CFed-DSMPC (Ours)'].values()))

    # A) w/o Shift Penalty
    orig = Config.SHIFT_WEIGHT; Config.SHIFT_WEIGHT = 0.0
    abl['w/o Shift\nPenalty'] = _run(
        'w/o Shift',
        lambda env,vp,cid: ShiftAwareMPC(env,vp,server.global_den[cid]))
    Config.SHIFT_WEIGHT = orig

    # B) w/o Adaptive Safety
    abl['w/o Adaptive\nSafety'] = _run(
        'w/o Safety',
        lambda env,vp,cid: FedAvgMPC(env,vp,server.global_den[cid]))

    # C) w/o Clustered FL
    abl['w/o Clustered\nFL'] = _run(
        'w/o CFL',
        lambda env,vp,cid: ShiftAwareMPC(env,vp,fa.global_den))

    plot_ablation(abl, f'{tag}_ablation')
    save_aggregate_table(abl, tag, 'ablation')

    # Print table
    keys = ['success_rate','safety_comply_rate','overall_min_dist_mean','corridor_correct_rate']
    print("\n  Ablation Results:")
    hdr = f"  {'Variant':<22}" + "".join(f"{k:>22}" for k in keys)
    print("  "+"-"*len(hdr)); print(hdr); print("  "+"-"*len(hdr))
    for v,ag in abl.items():
        print(f"  {v.replace(chr(10),' '):<22}" +
              "".join(f"{ag.get(k,0):>22.3f}" for k in keys))
    return abl


# ─────────────────────────────────────────────────────────────────────────────
#  Experiment 4: Federated analysis — fast version (reuses exploration data)
# ─────────────────────────────────────────────────────────────────────────────
def run_federated_analysis(env, clients, base_server, tag):
    """
    Reuses Phase-1 exploration clients (data already collected).
    Only re-runs FL training (Phase 2a/2b) for each parameter value.
    This is ~3× faster than full train_all() per sweep point.

    Saves combined results:
      results/<tag>_federated_rounds_metrics.json  (all rounds in one file)
      results/<tag>_federated_rounds_summary.csv
      results/<tag>_federated_mu_metrics.json
      results/<tag>_federated_mu_summary.csv
    """
    import json, csv
    print("\n" + "="*50 + "\n  Exp 4: Federated Analysis\n" + "="*50)
    orig_states = base_server.collected_states

    rounds_agg = {}; mu_agg = {}
    # Combined storage for all rounds/mu results
    rounds_all_metrics = {}; mu_all_metrics = {}

    # ── Part A: rounds sweep ──────────────────────────────────────────────
    print("\n  Part A: Rounds sweep", FL_ROUNDS_VALS)
    t_fed = time.time()
    orig_mu = Config.PROXIMAL_MU; Config.PROXIMAL_MU = 0.05
    for nr in FL_ROUNDS_VALS:
        t_r = time.time()
        sys.stdout.write(f"  rounds={nr} ... "); sys.stdout.flush()
        s, fa2 = _retrain_fl_only(clients, orig_states, rounds=nr, mu=0.05)
        gps2   = build_cluster_gps(env, orig_states)
        _, all_m = run_one_experiment(env, s, fa2, gps2,
                                      methods=FL_TRACK, save_plot=False,
                                      silent=True, save_records=True,
                                      tag=tag, experiment=f'federated_rounds_{nr}')
        rounds_agg[nr] = {m: aggregate_metrics(list(all_m[m].values()))
                          for m in FL_TRACK}
        rounds_all_metrics[f'rounds{nr}'] = {m: {str(k):v for k,v in all_m[m].items()}
                                              for m in FL_TRACK}
        print(f"done ({_fmt(time.time()-t_r)})")
    Config.PROXIMAL_MU = orig_mu

    # Save combined rounds results
    os.makedirs(Config.TABLES_DIR, exist_ok=True)
    r_tag = f"{tag}_federated_rounds{'-'.join(str(r) for r in FL_ROUNDS_VALS)}"
    jp = os.path.join(Config.TABLES_DIR, f'{r_tag}_metrics.json')
    with open(jp,'w') as f: json.dump(rounds_all_metrics, f, indent=2)
    cp = os.path.join(Config.TABLES_DIR, f'{r_tag}_summary.csv')
    with open(cp,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['rounds','method','success_rate','safety_comply_rate',
                    'overall_min_dist_mean','corridor_correct_rate'])
        for nr in FL_ROUNDS_VALS:
            for m in FL_TRACK:
                ag = rounds_agg[nr][m]
                w.writerow([nr, m, ag.get('success_rate',0),
                             ag.get('safety_comply_rate',0),
                             ag.get('overall_min_dist_mean',0),
                             ag.get('corridor_correct_rate',0)])
    print(f"  [Saved] {r_tag}")

    # ── Part B: mu sweep ──────────────────────────────────────────────────
    print("\n  Part B: μ sweep", FL_MU_VALS)
    orig_rounds = Config.ROUNDS; Config.ROUNDS = 10
    for mu in FL_MU_VALS:
        t_r = time.time()
        sys.stdout.write(f"  mu={mu} ... "); sys.stdout.flush()
        s, fa2 = _retrain_fl_only(clients, orig_states, rounds=10, mu=mu)
        gps2   = build_cluster_gps(env, orig_states)
        _, all_m = run_one_experiment(env, s, fa2, gps2,
                                      methods=FL_TRACK, save_plot=False,
                                      silent=True, save_records=True,
                                      tag=tag, experiment=f'federated_mu_{mu}')
        mu_agg[mu] = {m: aggregate_metrics(list(all_m[m].values()))
                      for m in FL_TRACK}
        mu_all_metrics[f'mu{mu}'] = {m: {str(k):v for k,v in all_m[m].items()}
                                     for m in FL_TRACK}
        print(f"done ({_fmt(time.time()-t_r)})")
    Config.ROUNDS = orig_rounds

    # Save combined mu results
    mu_tag = f"{tag}_federated_mu{'-'.join(str(m) for m in FL_MU_VALS)}"
    jp = os.path.join(Config.TABLES_DIR, f'{mu_tag}_metrics.json')
    with open(jp,'w') as f: json.dump(mu_all_metrics, f, indent=2)
    cp = os.path.join(Config.TABLES_DIR, f'{mu_tag}_summary.csv')
    with open(cp,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['mu','method','success_rate','safety_comply_rate',
                    'overall_min_dist_mean','corridor_correct_rate'])
        for mu in FL_MU_VALS:
            for m in FL_TRACK:
                ag = mu_agg[mu][m]
                w.writerow([mu, m, ag.get('success_rate',0),
                             ag.get('safety_comply_rate',0),
                             ag.get('overall_min_dist_mean',0),
                             ag.get('corridor_correct_rate',0)])
    print(f"  [Saved] {mu_tag}")

    plot_federated_analysis(rounds_agg, mu_agg, tag)
    save_aggregate_table(
        {f'rounds_{nr}_{m}': vals
         for nr, by_method in rounds_agg.items()
         for m, vals in by_method.items()},
        tag, 'federated_rounds_aggregate')
    save_aggregate_table(
        {f'mu_{mu}_{m}': vals
         for mu, by_method in mu_agg.items()
         for m, vals in by_method.items()},
        tag, 'federated_mu_aggregate')
    print(f"\n  Federated analysis complete — total {_fmt(time.time()-t_fed)}")
    return rounds_agg, mu_agg


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter sweep
# ─────────────────────────────────────────────────────────────────────────────
def _patch(param, val):
    mp = {'seed':'SEED','env_seed':'ENV_SEED','explore_samples':'EXPLORE_SAMPLES',
          'rounds':'ROUNDS','shift_weight':'SHIFT_WEIGHT','proximal_mu':'PROXIMAL_MU'}
    if param in mp:
        setattr(Config, mp[param], type(getattr(Config,mp[param]))(val))


def run_sweep(sweep_param):
    vals = SWEEP_CONFIGS.get(sweep_param, [])
    if not vals: print(f"Unknown sweep: {sweep_param}"); return
    print(f"\n{'='*50}\n  Sweep: {sweep_param} over {vals}\n{'='*50}")
    sr = {}
    for val in vals:
        print(f"\n  ── {sweep_param}={val} ──")
        _patch(sweep_param, val); Config.set_global_seed()
        env2 = CrossingEnv(seed=Config.ENV_SEED)
        s, fa2, gps2, _ = train_all(env2, verbose=False)
        sweep_tag = make_tag(sweep_param, val)
        _, all_m = run_one_experiment(env2, s, fa2, gps2, save_plot=False,
                                      silent=True, save_records=True,
                                      tag=sweep_tag,
                                      experiment=f'sweep_{sweep_param}')
        sr[val] = {m: aggregate_metrics(list(all_m[m].values())) for m in METHOD_NAMES}
        save_results({m: {str(k):v for k,v in all_m[m].items()} for m in METHOD_NAMES},
                     tag=sweep_tag)
    plot_sweep_lines(sr, sweep_param, f"sweep_{sweep_param}")
    print(f"\n  Sweep done → {Config.RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', nargs='?', default='',
                    choices=['','run_ablation','run_federated','run_all'])
    ap.add_argument('--load',  action='store_true')
    ap.add_argument('--sweep', type=str, default='')
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.sweep:
        run_sweep(args.sweep); sys.exit(0)

    t_total = time.time()
    Config.set_global_seed()
    env = CrossingEnv(seed=Config.ENV_SEED)
    tag = make_tag('seed')   # e.g. "seed2"

    # ── Load or train ─────────────────────────────────────────────────────
    load = args.load or (args.mode == '' and LOAD_MODELS)
    if load:
        print("\n[Loading saved models ...]")
        server, fa, gps, clients = load_models()
    else:
        server, fa, gps, clients = train_all(env)
        save_models(server, fa, gps, clients)

    # ── Standalone sub-experiments (load saved models, run experiment only) ─
    if args.mode in ('run_ablation', 'run_federated', 'run_all'):
        if args.mode in ('run_ablation', 'run_all'):
            run_ablation(env, server, fa, gps, tag)
        if args.mode in ('run_federated', 'run_all'):
            if clients is None:
                print("  [Warning] Federated analysis needs exploration clients.")
                print("  Re-running Phase 1 exploration to get clients ...")
                clients, server2, fa2 = _do_exploration(env, verbose=True)
                server.collected_states = server2.collected_states
            run_federated_analysis(env, clients, server, tag)
        sys.exit(0)

    # ── Main experiment ───────────────────────────────────────────────────
    print("\n" + "="*50 + "\n  Phase 3: Simulations\n" + "="*50)
    sim_results, all_metrics = run_one_experiment(env, server, fa, gps,
                                                  save_plot=True, tag=tag,
                                                  save_records=True,
                                                  experiment='main')

    save_results({m: {str(k):v for k,v in all_metrics[m].items()}
                  for m in METHOD_NAMES}, tag=tag)
    plot_summary_bars(all_metrics, tag)
    plot_per_cluster(all_metrics, tag)
    print_summary_table(all_metrics)
    print("\n  Generating OOD heatmap ...")
    plot_ood_heatmap(env, server, tag)

    # ── Optional: ablation + federated ───────────────────────────────────
    do_abl = RUN_ABLATION  if args.mode == '' else False
    do_fed = RUN_FEDERATED if args.mode == '' else False

    if do_abl:
        run_ablation(env, server, fa, gps, tag)

    if do_fed:
        if clients is None:
            print("  [Warning] No exploration clients available for FL analysis. Skipping.")
        else:
            run_federated_analysis(env, clients, server, tag)

    print(f"\n  All results → '{Config.RESULTS_DIR}/'")
    print(f"  Total time: {_fmt(time.time()-t_total)}")
