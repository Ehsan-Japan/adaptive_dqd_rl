"""
run_0_full_study.py — the whole comparison in one command.

    python scripts/run_0_full_study.py

Runs 1 -> 2 -> 3 -> 4 -> 5 in order, skipping any stage whose output is
already on disk.  Convenient, and the wrong way to work while the method is
still moving: run the stages by hand so you see the decoder's threshold and
the PPO entropy curve before you spend an hour on the comparison.
"""
import os
import runpy
import sys

from _common import banner

STAGES = ["run_1_import_devices.py", "run_2_train_decoder.py",
          "run_3_train_agent.py", "run_4_compare_arms.py",
          "run_5_policy_anatomy.py"]

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    for stage in STAGES:
        banner(f"stage: {stage}")
        runpy.run_path(os.path.join(here, stage), run_name="__main__")
