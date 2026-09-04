"""Shared preamble: put src/ on the path, print a banner and the settings."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))


def banner(text: str) -> None:
    print("\n" + "=" * 74 + f"\n  {text}\n" + "=" * 74)


def settings(**kw) -> None:
    """Echo the run's settings.  A results folder that cannot say what
    produced it is not a result."""
    for k, v in kw.items():
        print(f"    {k:<24} {v}")
