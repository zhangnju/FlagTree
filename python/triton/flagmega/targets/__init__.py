# Copyright 2025- FlagOS Contributors
# SPDX-License-Identifier: MIT
"""FlagMega target registry."""

from triton.flagmega.targets.base import Target, get_target, register_target, target_names
from triton.flagmega.targets.nvidia_sm90 import NvidiaSm90Target, Sm90Capability
from triton.flagmega.targets.ntt import NttTarget
from triton.flagmega.targets.ntt_options import NttTargetOptions, PyNttTargetOptions
from triton.flagmega.targets.pyntt import PyNttTarget
from triton.flagmega.targets.machine import NttTargetMachine
from triton.flagmega.targets.nvidia.machine import NvidiaSm90Machine
from triton.flagmega.targets.amd_gfx1100 import AmdGfx1100Target
from triton.flagmega.targets.amd_gfx1201 import AmdGfx1201Target

register_target(NvidiaSm90Target())
register_target(AmdGfx1100Target())
register_target(AmdGfx1201Target())

__all__ = [
    "NttTargetOptions",
    "PyNttTargetOptions",
    "NttTargetMachine",
    "NttTarget",
    "PyNttTarget",
    "NvidiaSm90Target",
    "NvidiaSm90Machine",
    "Sm90Capability",
    "Target",
    "get_target",
    "register_target",
    "target_names",
]
