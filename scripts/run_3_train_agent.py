"""
run_3_train_agent.py — train the PPO policy on the TRAINING devices only.

    python scripts/run_3_train_agent.py

The environment is built from the training devices and the frozen shared
decoder.  No test device is ever loaded by this script; that is not a
convention, it is why the split object keeps the two lists apart.

WHAT TO WATCH WHILE IT RUNS

  mean episode return   telescopes to (final F1@1 - empty-measurement F1@1),
                        so it is the headline metric itself, shifted by a
                        constant.  It should start at roughly the score of
                        the `random_lines` control, because the Beta head is
                        initialised near-uniform on purpose.

  entropy               the policy's entropy over (rho, theta).  It should
                        fall, but not to zero: a policy with zero entropy has
                        collapsed to one fixed geometry, which means it found
                        no device-dependent structure to exploit and the
                        whole premise is in trouble.  A policy that stays at
                        its initial entropy has learned nothing.  Both
                        failure modes look like "training ran fine".

If the return plateaus at the fixed-geometry level, the honest reading is
that at this budget the honeycomb is regular enough that a good fixed sweep
is already near-optimal — which is a real result and is what run_4's oracle
column is there to distinguish from an agent that simply failed to train.
"""
import json
import os

from _common import banner, settings

from adaptive_dqd.agents.ppo import PPOAgent, PPOConfig
from adaptive_dqd.config import devices as dv
from adaptive_dqd.decoder import agnostic
from adaptive_dqd.decoder.baseline_net import grid_train
from adaptive_dqd.envs import SweepEnv

N_TRAIN, N_TEST = 500, 50
N_LINES, N_POINTS = 8, 60
ITERATIONS = 300
SEED = 0
DEVICE = "cuda"

DECODER = os.path.join(dv.CHECKPOINTS, f"d_agn_{N_LINES}x{N_POINTS}.pt")
OUT = os.path.join(dv.CHECKPOINTS, f"ppo_{N_LINES}x{N_POINTS}.pt")

if __name__ == "__main__":
    banner("PPO — adaptive sweep placement")
    settings(budget=f"{N_LINES} x {N_POINTS}", iterations=ITERATIONS,
             decoder=DECODER, seed=SEED, out=OUT)

    split = dv.load_split(N_TRAIN, N_TEST, N_LINES, N_POINTS)
    grids, truths = dv.load_arrays(split.train_dirs)

    net, meta = grid_train.load(DECODER)
    env = SweepEnv(grids, truths, agnostic.as_callable(net, DEVICE),
                   threshold=meta["threshold"], n_lines=N_LINES,
                   n_points=N_POINTS, seed=SEED)

    agent = PPOAgent(PPOConfig(), device=DEVICE, seed=SEED)
    history = agent.train(env, list(range(len(grids))), iterations=ITERATIONS)

    os.makedirs(dv.CHECKPOINTS, exist_ok=True)
    agent.save(OUT)
    with open(OUT.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f, indent=1)
    print(f"\n  saved -> {OUT}")
