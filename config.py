import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt


class Config:
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(CODE_DIR, 'results')
    FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
    TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
    MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
    for _dir in (RESULTS_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
        os.makedirs(_dir, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED     = 2
    ENV_SEED = 2

    @classmethod
    def set_global_seed(cls, seed: int = None):
        s = seed if seed is not None else cls.SEED
        random.seed(s); np.random.seed(s); torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s); torch.cuda.manual_seed_all(s)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False

    plt.rcParams.update({
        'font.family':      'serif',
        'font.serif':       ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.facecolor':   '#F8F9FA',
        'figure.facecolor': '#FFFFFF',
        'font.size':         18,
        'axes.labelsize':    20,
        'axes.titlesize':    22,
        'legend.fontsize':   16,
    })

    DT        = 0.2
    SIM_STEPS = 800

    NUM_CLUSTERS   = 3
    NUM_AGENTS     = 6
    AGENT_CLUSTERS = [0, 0, 1, 1, 2, 2]

    # Agent types: Small / Medium / Large  (width is the key differentiator)
    # Note: all words using 'car' replaced with 'agent' in display names
    VEHICLE_TYPES = {
        0: {'name': 'Small Agent',  'L': 3.0, 'W': 1.2, 'mass': 1000.0, 'color': '#3399FF'},
        1: {'name': 'Medium Agent', 'L': 3.0, 'W': 1.8, 'mass': 2500.0, 'color': '#FFAA53'},
        2: {'name': 'Large Agent',  'L': 3.0, 'W': 2.4, 'mass': 5000.0, 'color': '#FF6666'},
    }

    MAX_STEER = 0.6
    MAX_ACCEL = 2.0
    DRAG      = 0.05
    TARGET_SPEED = 3.5
    HORIZON      = 12

    SHIFT_WEIGHT      = 180.0
    SHIFT_ALPHA       = 1.0
    MAX_SHIFT_PENALTY = 8.0

    SAFETY_MARGIN       = 0.2
    ROBUST_FIXED_MARGIN = 0.8

    # Federated learning
    ROUNDS       = 10
    LOCAL_EPOCHS = 5
    LR           = 0.005
    PROXIMAL_MU  = 0.05   # FedProx proximal coefficient μ

    EXPLORE_SAMPLES  = 800
    EXPLORE_ATTEMPTS = 160000

    X_MIN, X_MAX = -25.0, 25.0
    Y_MIN, Y_MAX = -14.0, 14.0

    START_X = -22.0;  START_Y = 9.0
    GOAL_X  =  22.0;  GOAL_Y  = 9.0

    CLUSTER_TARGET_Y  = {0: 7.5, 1: 0.0, 2: -7.5}
    EXPLORE_CLEARANCE = {0: 0.15, 1: 0.10, 2: 0.10}
