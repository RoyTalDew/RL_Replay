"""
Short simulation runner for RL_Replay.
Runs 3 simulations of 3 strategies on the small mattar maze (6x9),
with 10 episodes each, then renders a performance plot.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from parameters import setParams
from main import Replay_Sim

# ── Config ──────────────────────────────────────────────────────────────────
MAZE_NAME   = 'mattar'
STRATEGIES  = ['EVB', 'dyna', 'no_replay']
N_SIM       = 3
N_EPISODES  = 10          # short run (goal changes at episode 5)
N_STEPS     = 200_000     # generous cap so episodes can complete
SEED        = 42

np.random.seed(SEED)

# ── Maze definition (mattar 6×9) ─────────────────────────────────────────────
params = setParams()
params.N_SIMULATIONS  = N_SIM
params.MAX_N_EPISODES = N_EPISODES
params.MAX_N_STEPS    = N_STEPS

params.maze = np.zeros((6, 9))
for wall in [[slice(1, 4), 2], [slice(0, 3), 7], [4, 5]]:
    params.maze[wall[0], wall[1]] = 1

params.s_start       = np.array([[2, 0]])
params.s_start_rand  = True
params.actPolicy     = 'e_greedy'
params.epsilon       = 0.05
params.s_end         = np.array([[0, 8]])
params.s_end_change  = np.array([[5, 0]])
params.rewMag        = np.array([[1]])
params.rewSTD        = np.array([[0.1]])
params.rewProb       = np.array([[1]])

# Strategy overrides
strategy_cfg = {
    'EVB':       {'nPlan': 20, 'setAllGainToOne': False, 'setAllNeedToOne': False},
    'dyna':      {'nPlan': 20, 'setAllGainToOne': True,  'setAllNeedToOne': True},
    'no_replay': {'nPlan':  0, 'setAllGainToOne': True,  'setAllNeedToOne': True},
}

os.makedirs(os.path.join('checkpoints', MAZE_NAME), exist_ok=True)

# ── Run simulations ──────────────────────────────────────────────────────────
results = {}   # strategy → list of steps_per_episode arrays

for strategy in STRATEGIES:
    cfg = strategy_cfg[strategy]
    params.nPlan            = cfg['nPlan']
    params.setAllGainToOne  = cfg['setAllGainToOne']
    params.setAllNeedToOne  = cfg['setAllNeedToOne']

    runs = []
    for k in range(N_SIM):
        print(f"[{strategy}] simulation {k+1}/{N_SIM} …", flush=True)
        np.random.seed()
        sim = Replay_Sim(params, strategy, MAZE_NAME, sim_i=f'short_{k}')
        sim.pre_explore_env()
        sim.build_transition_mat()
        sim.explore_env()
        steps = sim.performance_df['steps_per_episode'].values[:N_EPISODES]
        runs.append(steps)
        del sim

    results[strategy] = np.array(runs)   # shape (N_SIM, N_EPISODES)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle('RL Replay — Mattar Maze (short simulation)', fontsize=13, fontweight='bold')

colors = {'EVB': '#2196F3', 'dyna': '#FF9800', 'no_replay': '#9E9E9E'}
episodes = np.arange(1, N_EPISODES + 1)

# --- Panel 1: steps per episode (mean ± SEM) --------------------------------
ax = axes[0]
for strategy, data in results.items():
    mean = np.nanmean(data, axis=0)
    sem  = np.nanstd(data, axis=0) / np.sqrt(N_SIM)
    ax.plot(episodes, mean, label=strategy, color=colors[strategy], lw=2)
    ax.fill_between(episodes, mean - sem, mean + sem,
                    alpha=0.25, color=colors[strategy])

ax.axvline(N_EPISODES / 2 + 0.5, ls=':', color='gray', lw=1.4, label='goal change')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps to goal')
ax.set_title('Steps per Episode')
ax.set_xticks(episodes)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
# Use log scale only when all plotted values are positive
all_means = np.concatenate([np.nanmean(d, axis=0) for d in results.values()])
if np.nanmin(all_means) > 0:
    ax.set_yscale('log')

# --- Panel 2: maze layout ---------------------------------------------------
ax = axes[1]
maze_vis = np.zeros((6, 9))
for wall in [[slice(1, 4), 2], [slice(0, 3), 7], [4, 5]]:
    maze_vis[wall[0], wall[1]] = 1

ax.imshow(maze_vis, cmap='Greys', vmin=0, vmax=1.5)

# start
ax.scatter(0, 2, s=180, color='#4CAF50', zorder=5)
ax.text(0, 2, 'S', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
# goal 1
ax.scatter(8, 0, s=180, color='#2196F3', zorder=5)
ax.text(8, 0, 'G1', ha='center', va='center', fontsize=7, fontweight='bold', color='white')
# goal 2
ax.scatter(0, 5, s=180, color='#FF5722', zorder=5)
ax.text(0, 5, 'G2', ha='center', va='center', fontsize=7, fontweight='bold', color='white')

ax.set_title('Mattar Maze Layout')
ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
ax.grid(which='minor', color='lightgray', linewidth=0.5)
ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

legend_handles = [
    mpatches.Patch(color='#4CAF50', label='Start'),
    mpatches.Patch(color='#2196F3', label='Goal 1 (ep 1–5)'),
    mpatches.Patch(color='#FF5722', label='Goal 2 (ep 6–10)'),
    mpatches.Patch(color='black',   label='Wall'),
]
ax.legend(handles=legend_handles, fontsize=8, loc='lower right')

plt.tight_layout()
out_path = os.path.join('checkpoints', MAZE_NAME, 'short_simulation.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {out_path}")
plt.close()
