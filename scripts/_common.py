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


def torch_device() -> str:
    """'cuda' when it is really there, else 'cpu'.  $ADQ_DEVICE overrides.

    The comparison is device-independent by construction — same weights, same
    metric — so falling back is safe; silently training on a GPU that is not
    present is not.
    """
    import os
    forced = os.environ.get("ADQ_DEVICE")
    if forced:
        return forced
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def env_int(name: str, default: int) -> int:
    """A run-scale knob, overridable from the environment.

    The defaults in these scripts are the protocol and stay the protocol; the
    override exists so a reduced pass can be run without editing them, and so
    the reduced pass has to name itself on the command line.
    """
    import os
    return int(os.environ.get(name, default))
