"""
viz_sim.py — Accelerated animated learning visualization for RL_Replay.

Runs ONE simulation while silently capturing a snapshot every FRAME_EVERY
real steps (plus one snapshot at every episode boundary), then renders all
snapshots into an animated GIF.

Each frame shows:
  Left  — Mattar 6×9 maze with the agent position (red dot), current goal
           (star), and Q-value policy arrows (plasma colour = learned value).
  Right — Live learning curve: steps-to-goal per episode, coloured by goal.

Usage:
    python viz_sim.py [STRATEGY]   # STRATEGY: EVB | dyna | no_replay
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import Normalize

from parameters import setParams
from main import Replay_Sim

# ── Config ─────────────────────────────────────────────────────────────────────
STRATEGY    = sys.argv[1] if len(sys.argv) > 1 else 'EVB'
N_EPISODES  = 20           # goal changes at episode 10 → 11
N_STEPS     = 400_000      # step budget
FRAME_EVERY = 250          # capture a frame every N steps (+ episode ends)
FPS         = 15           # GIF playback speed
OUTPUT      = f'checkpoints/mattar/learning_{STRATEGY}.gif'

MAZE_ROWS, MAZE_COLS = 6, 9

# ── Params ─────────────────────────────────────────────────────────────────────
params = setParams()
params.MAX_N_EPISODES    = N_EPISODES
params.MAX_N_STEPS       = N_STEPS
params.maze              = np.zeros((MAZE_ROWS, MAZE_COLS))
for wall in [[slice(1, 4), 2], [slice(0, 3), 7], [4, 5]]:
    params.maze[wall[0], wall[1]] = 1
params.s_start           = np.array([[2, 0]])
params.s_start_rand      = True
params.actPolicy         = 'e_greedy'
params.epsilon           = 0.05
params.s_end             = np.array([[0, 8]])
params.s_end_change      = np.array([[5, 0]])
params.rewMag            = np.array([[1]])
params.rewSTD            = np.array([[0.1]])
params.rewProb           = np.array([[1]])
params.nPlan             = 20
params.setAllGainToOne   = False
params.setAllNeedToOne   = False
params.plot_agent        = False
params.plot_Q            = False

MAZE_GRID = params.maze.copy()

# ── Capturing subclass ─────────────────────────────────────────────────────────
class CaptureSim(Replay_Sim):
    """
    Overrides plot_agent (called after every real step in explore_env) to
    silently record snapshots instead of drawing to screen.
    """

    def __init__(self, *args, frame_every=250, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_every  = frame_every
        self._global_step  = 0
        self._in_explore   = False
        self.frames: list  = []

    def plot_agent(self, st):
        if not self._in_explore:
            return
        at_goal = any(np.array_equal(st, g) for g in self.this_goal)
        if self._global_step % self._frame_every == 0 or at_goal:
            self.frames.append({
                'agent'    : (int(st[0]), int(st[1])),
                'Q'        : self.Q.copy(),
                'episode'  : self.num_episodes,
                'step'     : self._global_step,
                'goal'     : self.this_goal.copy(),
                'steps_ep' : self.performance_df['steps_per_episode']
                                .values[:N_EPISODES].copy(),
            })
        self._global_step += 1
        # print progress to terminal
        if self._global_step % 5000 == 0:
            ep = self.num_episodes
            print(f'\r  step {self._global_step:,}  episode {ep}/{N_EPISODES}  '
                  f'frames captured: {len(self.frames)}   ', end='', flush=True)

    def explore_env(self):
        self._in_explore = True
        super().explore_env()
        self._in_explore = False


# ── Run simulation ─────────────────────────────────────────────────────────────
print(f'Strategy : {STRATEGY}')
print(f'Episodes : {N_EPISODES}  (goal changes after episode {N_EPISODES // 2})')
print(f'Frame every {FRAME_EVERY} steps + every episode end')
print('Running simulation …')
np.random.seed(42)
os.makedirs('checkpoints/mattar', exist_ok=True)

sim = CaptureSim(params, STRATEGY, 'mattar', sim_i='viz', frame_every=FRAME_EVERY)
sim.pre_explore_env()
sim.build_transition_mat()
sim.explore_env()

frames = sim.frames
del sim
print(f'\nSimulation done — {len(frames)} frames captured.')

if not frames:
    print('No frames captured — exiting.')
    sys.exit(1)

# ── Style constants ────────────────────────────────────────────────────────────
WALL_CLR  = '#37474F'
FREE_CLR  = '#ECEFF1'
AGENT_CLR = '#E53935'
GOAL1_CLR = '#1565C0'
GOAL2_CLR = '#EF6C00'
START_CLR = '#2E7D32'
START_POS = (2, 0)   # (row, col)

# action → (dU, dV) in data coords  (col = x → right;  row = y with inverted axis → down)
# UP=0: row-1 → dV=-0.35   DOWN=1: row+1 → dV=+0.35
# RIGHT=2: col+1 → dU=+0.35  LEFT=3: col-1 → dU=-0.35
ACT_ARROW = {0: (0.0, -0.35), 1: (0.0, 0.35), 2: (0.35, 0.0), 3: (-0.35, 0.0)}

# Pre-build RGBA maze background image (reused every frame)
def _hex_rgb(h):
    h = h.lstrip('#')
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

_maze_rgba = np.ones((MAZE_ROWS, MAZE_COLS, 4))
for _r in range(MAZE_ROWS):
    for _c in range(MAZE_COLS):
        _maze_rgba[_r, _c, :3] = _hex_rgb(WALL_CLR if MAZE_GRID[_r, _c] else FREE_CLR)


# ── Per-frame render function ──────────────────────────────────────────────────
def render(ax_maze, ax_curve, fd):
    """Draw one frame into ax_maze and ax_curve."""

    # ── Left panel: maze ───────────────────────────────────────────────────────
    ax_maze.clear()

    # background tiles
    ax_maze.imshow(
        _maze_rgba, origin='upper',
        extent=[-0.5, MAZE_COLS - 0.5, MAZE_ROWS - 0.5, -0.5],
        zorder=1, interpolation='nearest',
    )

    # subtle cell grid
    for x in np.arange(-0.5, MAZE_COLS, 1):
        ax_maze.axvline(x, color='#B0BEC5', lw=0.3, zorder=2)
    for y in np.arange(-0.5, MAZE_ROWS, 1):
        ax_maze.axhline(y, color='#B0BEC5', lw=0.3, zorder=2)

    # Q-value policy arrows
    Q = fd['Q']
    q_max = float(np.nanmax(Q)) if np.nanmax(Q) > 0 else 1.0
    norm  = Normalize(vmin=0, vmax=q_max, clip=True)
    cmap  = plt.get_cmap('plasma')

    Xq, Yq, Uq, Vq, Cq = [], [], [], [], []
    for r in range(MAZE_ROWS):
        for c in range(MAZE_COLS):
            if MAZE_GRID[r, c]:          # wall
                continue
            si   = int(np.ravel_multi_index([r, c], [MAZE_ROWS, MAZE_COLS]))
            q    = Q[si]
            best = int(np.argmax(q))
            val  = float(q[best])
            if val < 1e-9:
                continue
            du, dv = ACT_ARROW[best]
            Xq.append(c);  Yq.append(r)
            Uq.append(du); Vq.append(dv)
            Cq.append(norm(val))

    if Xq:
        arrow_colors = cmap(np.array(Cq))
        ax_maze.quiver(
            Xq, Yq, Uq, Vq,
            color=arrow_colors,
            scale=1, scale_units='xy',
            width=0.022, headwidth=4, headlength=5, headaxislength=4,
            zorder=5,
        )

    # start marker
    ax_maze.scatter(START_POS[1], START_POS[0], s=150, c=START_CLR,
                    marker='s', zorder=7, edgecolors='white', linewidths=1.2)
    ax_maze.text(START_POS[1], START_POS[0], 'S',
                 ha='center', va='center', fontsize=7, fontweight='bold',
                 color='white', zorder=8)

    # goal marker (changes mid-experiment)
    goal  = fd['goal']
    gc    = GOAL1_CLR if np.array_equal(goal, params.s_end) else GOAL2_CLR
    gl    = 'G1'      if np.array_equal(goal, params.s_end) else 'G2'
    gr, gcol = int(goal[0][0]), int(goal[0][1])
    ax_maze.scatter(gcol, gr, s=260, c=gc, marker='*',
                    zorder=7, edgecolors='white', linewidths=0.8)
    ax_maze.text(gcol, gr + 0.5, gl,
                 ha='center', va='bottom', fontsize=7, fontweight='bold',
                 color=gc, zorder=8)

    # agent
    ar, ac = fd['agent']
    ax_maze.scatter(ac, ar, s=210, c=AGENT_CLR, marker='o',
                    zorder=9, edgecolors='white', linewidths=1.8)

    ax_maze.set_xlim(-0.5, MAZE_COLS - 0.5)
    ax_maze.set_ylim(MAZE_ROWS - 0.5, -0.5)
    ax_maze.set_aspect('equal')
    ax_maze.axis('off')
    ax_maze.set_title(
        f'{STRATEGY}  |  Episode {fd["episode"] + 1} / {N_EPISODES}'
        f'  |  Step {fd["step"]:,}',
        fontsize=9, fontweight='bold', pad=5,
    )

    # ── Right panel: learning curve ────────────────────────────────────────────
    ax_curve.clear()
    steps      = fd['steps_ep']
    ep_nums    = np.arange(1, N_EPISODES + 1)
    bar_colors = [GOAL1_CLR if e <= N_EPISODES // 2 else GOAL2_CLR
                  for e in ep_nums]

    for e, s, bc in zip(ep_nums, steps, bar_colors):
        if not np.isnan(s) and s > 0:
            ax_curve.bar(e, s, color=bc, edgecolor='white',
                         linewidth=0.4, zorder=3, alpha=0.85)

    ax_curve.axvline(N_EPISODES // 2 + 0.5,
                     ls='--', color='#757575', lw=1.5, zorder=4)
    ax_curve.set_xlim(0.5, N_EPISODES + 0.5)
    ax_curve.set_xticks(ep_nums)
    ax_curve.tick_params(axis='x', labelsize=7)
    ax_curve.tick_params(axis='y', labelsize=7)
    ax_curve.set_xlabel('Episode', fontsize=8)
    ax_curve.set_ylabel('Steps to goal', fontsize=8)
    ax_curve.set_title('Learning Curve', fontsize=9, fontweight='bold', pad=5)
    ax_curve.grid(axis='y', alpha=0.3, zorder=0)

    legend_handles = [
        mpatches.Patch(color=GOAL1_CLR, label=f'Goal 1 (ep 1–{N_EPISODES // 2})'),
        mpatches.Patch(color=GOAL2_CLR, label=f'Goal 2 (ep {N_EPISODES // 2 + 1}–{N_EPISODES})'),
        plt.Line2D([0], [0], ls='--', color='#757575', lw=1.5, label='goal change'),
    ]
    ax_curve.legend(handles=legend_handles, fontsize=7, loc='upper right')


# ── Build & save animated GIF ─────────────────────────────────────────────────
fig, (ax_maze, ax_curve) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('#F5F5F5')
fig.suptitle(f'RL Replay  ·  Mattar 6×9 Maze  ·  {STRATEGY}',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])

def update(i):
    if i % 10 == 0 or i == len(frames) - 1:
        print(f'\r  rendering frame {i + 1}/{len(frames)} …',
              end='', flush=True)
    render(ax_maze, ax_curve, frames[i])
    return []

print(f'\nRendering {len(frames)} frames at {FPS} fps …')
ani = animation.FuncAnimation(
    fig, update, frames=len(frames), blit=False, repeat=True,
)
ani.save(OUTPUT, writer=animation.PillowWriter(fps=FPS), dpi=110)
print(f'\n\nSaved → {OUTPUT}')
plt.close()
