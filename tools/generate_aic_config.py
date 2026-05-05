#!/usr/bin/env python3
"""
Generate an AIC engine config YAML with N randomized trials.

Designed for the Intrinsic AI for Industry Challenge toolkit sample_config format.
No third-party Python packages required.

Example:
    python3 tools/generate_aic_config.py \
      --output ../aic_engine/config/cheatcode_stress_test.yaml \
      --num-trials 20 \
      --seed 42 \
      --task-types both \
      --cable-types both \
      --board-x-range 0.13 0.18 \
      --board-y-range -0.20 0.02 \
      --board-z-range 1.14 1.14 \
      --board-yaw-range 2.9 3.25 \
      --nic-count-range 1 3 \
      --sc-count-range 0 2 \
      --lc-mount-count-range 0 2 \
      --sfp-mount-count-range 0 2 \
      --sc-mount-count-range 0 2 \
      --nic-translation-range -0.0215 0.0234 \
      --nic-yaw-range-deg -10 10 \
      --sc-translation-range -0.06 0.055 \
      --mount-translation-range -0.09425 0.09425 \
      --mount-yaw-range-deg -60 60 \
      --time-limit 180
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Any


# ----------------------------
# YAML dumper (simple subset)
# ----------------------------
def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            # keep reasonable precision but avoid scientific notation noise
            return f"{value:.6f}".rstrip("0").rstrip(".") if not value.is_integer() else f"{int(value)}"
        return str(value)
    s = str(value)
    # quote strings conservatively
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def dump_yaml(obj: Any, indent: int = 0) -> str:
    spaces = " " * indent
    if isinstance(obj, dict):
        lines: list[str] = []
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{spaces}{key}:")
                lines.append(dump_yaml(value, indent + 2))
            else:
                lines.append(f"{spaces}{key}: {yaml_scalar(value)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{spaces}-")
                lines.append(dump_yaml(item, indent + 2))
            else:
                lines.append(f"{spaces}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{spaces}{yaml_scalar(obj)}"


# ----------------------------
# Parsing helpers
# ----------------------------
def two_floats(values: list[str]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected exactly 2 values")
    a, b = float(values[0]), float(values[1])
    if a > b:
        raise argparse.ArgumentTypeError("Range min must be <= max")
    return a, b


def two_ints(values: list[str]) -> tuple[int, int]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Expected exactly 2 values")
    a, b = int(values[0]), int(values[1])
    if a > b:
        raise argparse.ArgumentTypeError("Range min must be <= max")
    return a, b


def clamp_int_range(rng: tuple[int, int], low: int, high: int) -> tuple[int, int]:
    a, b = max(low, rng[0]), min(high, rng[1])
    if a > b:
        raise ValueError(f"Invalid clamped integer range {rng} -> [{a}, {b}]")
    return a, b


def sample_uniform(rng: tuple[float, float], ndigits: int = 6) -> float:
    return round(random.uniform(rng[0], rng[1]), ndigits)


def sample_int(rng: tuple[int, int]) -> int:
    return random.randint(rng[0], rng[1])


def sample_subset(indices: list[int], count_rng: tuple[int, int]) -> list[int]:
    lo, hi = clamp_int_range(count_rng, 0, len(indices))
    k = sample_int((lo, hi))
    return sorted(random.sample(indices, k)) if k > 0 else []


def choose_task_type(task_types: str) -> str:
    if task_types == "both":
        return random.choice(["sfp", "sc"])
    return task_types


def choose_cable_type(cable_types: str) -> str:
    if cable_types == "both":
        return random.choice(["sfp_sc_cable", "sfp_sc_cable_reversed"])
    return cable_types


@dataclass
class CablePose:
    offset_x: float
    offset_y: float
    offset_z: float
    roll: float
    pitch: float
    yaw: float

    def gripper_offset_dict(self) -> dict[str, float]:
        return {
            "x": self.offset_x,
            "y": self.offset_y,
            "z": self.offset_z,
        }


def parse_offset(text: str) -> CablePose:
    parts = [float(x.strip()) for x in text.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "Offset must have 6 comma-separated floats: x,y,z,roll,pitch,yaw"
        )
    return CablePose(*parts)


def make_entity_block(name: str, translation: float, yaw_rad: float = 0.0) -> dict[str, Any]:
    return {
        "entity_present": True,
        "entity_name": name,
        "entity_pose": {
            "translation": translation,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": yaw_rad,
        },
    }


def empty_entity_block() -> dict[str, Any]:
    return {"entity_present": False}


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def sample_named_mounts(prefix: str, count_rng: tuple[int, int], rail_count: int) -> dict[str, dict[str, Any]]:
    selected = sample_subset(list(range(rail_count)), count_rng)
    result: dict[str, dict[str, Any]] = {}
    for rail_idx in range(rail_count):
        key = f"{prefix}_rail_{rail_idx}"
        if rail_idx in selected:
            result[key] = make_entity_block(
                name=f"{prefix}_{rail_idx}",
                translation=sample_uniform(args.mount_translation_range),
                yaw_rad=sample_uniform(args.mount_yaw_range_rad),
            )
        else:
            result[key] = empty_entity_block()
    return result


def make_trial(trial_idx: int, args: argparse.Namespace) -> dict[str, Any]:
    task_type = choose_task_type(args.task_types)
    cable_type = choose_cable_type(args.cable_types)

    nic_present = sample_subset(list(range(5)), args.nic_count_range)
    sc_present = sample_subset(list(range(2)), args.sc_count_range)

    # Ensure feasibility for the chosen task.
    if task_type == "sfp" and not nic_present:
        nic_present = [random.randrange(5)]
    if task_type == "sc" and not sc_present:
        sc_present = [random.randrange(2)]

    trial_scene: dict[str, Any] = {
        "task_board": {
            "pose": {
                "x": sample_uniform(args.board_x_range),
                "y": sample_uniform(args.board_y_range),
                "z": sample_uniform(args.board_z_range),
                "roll": sample_uniform(args.board_roll_range),
                "pitch": sample_uniform(args.board_pitch_range),
                "yaw": sample_uniform(args.board_yaw_range),
            },
        }
    }

    # NIC rails
    for rail_idx in range(5):
        key = f"nic_rail_{rail_idx}"
        if rail_idx in nic_present:
            trial_scene["task_board"][key] = make_entity_block(
                name=f"nic_card_{rail_idx}",
                translation=sample_uniform(args.nic_translation_range),
                yaw_rad=sample_uniform(args.nic_yaw_range_rad),
            )
        else:
            trial_scene["task_board"][key] = empty_entity_block()

    # SC rails
    for rail_idx in range(2):
        key = f"sc_rail_{rail_idx}"
        if rail_idx in sc_present:
            trial_scene["task_board"][key] = make_entity_block(
                name=f"sc_mount_{rail_idx}",
                translation=sample_uniform(args.sc_translation_range),
                yaw_rad=0.0,
            )
        else:
            trial_scene["task_board"][key] = empty_entity_block()

    # Pick-area mounts / distractors
    trial_scene["task_board"].update(
        sample_named_mounts("lc_mount", args.lc_mount_count_range, 2)
    )
    trial_scene["task_board"].update(
        sample_named_mounts("sfp_mount", args.sfp_mount_count_range, 2)
    )
    trial_scene["task_board"].update(
        sample_named_mounts("sc_mount", args.sc_mount_count_range, 2)
    )

    # Cable naming / offsets
    cable_name = "cable_1" if cable_type == "sfp_sc_cable_reversed" else "cable_0"
    offset = args.reversed_gripper_offset if cable_type == "sfp_sc_cable_reversed" else args.normal_gripper_offset

    cables = {
        cable_name: {
            "pose": {
                "gripper_offset": offset.gripper_offset_dict(),
                "roll": offset.roll,
                "pitch": offset.pitch,
                "yaw": offset.yaw,
            },
            "attach_cable_to_gripper": True,
            "cable_type": cable_type,
        }
    }
    trial_scene["cables"] = cables

    if task_type == "sfp":
        target_rail = random.choice(nic_present)
        tasks = {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": cable_name,
                "plug_type": "sfp",
                "plug_name": "sfp_tip",
                "port_type": "sfp",
                "port_name": "sfp_port_0",
                "target_module_name": f"nic_card_mount_{target_rail}",
                "time_limit": args.time_limit,
            }
        }
    else:
        target_rail = random.choice(sc_present)
        tasks = {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": cable_name,
                "plug_type": "sc",
                "plug_name": "sc_tip",
                "port_type": "sc",
                "port_name": "sc_port_base",
                "target_module_name": f"sc_port_{target_rail}",
                "time_limit": args.time_limit,
            }
        }

    return {
        "scene": trial_scene,
        "tasks": tasks,
    }


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
        "scoring": {
            "topics": [
                {"topic": {"name": "/joint_states", "type": "sensor_msgs/msg/JointState"}},
                {"topic": {"name": "/tf", "type": "tf2_msgs/msg/TFMessage"}},
                {"topic": {"name": "/tf_static", "type": "tf2_msgs/msg/TFMessage"}, "latched": True},
                {"topic": {"name": "/scoring/tf", "type": "tf2_msgs/msg/TFMessage"}},
                {"topic": {"name": "/aic/gazebo/contacts/off_limit", "type": "ros_gz_interfaces/msg/Contacts"}},
                {"topic": {"name": "/fts_broadcaster/wrench", "type": "geometry_msgs/msg/WrenchStamped"}},
                {"topic": {"name": "/aic_controller/joint_commands", "type": "aic_control_interfaces/msg/JointMotionUpdate"}},
                {"topic": {"name": "/aic_controller/pose_commands", "type": "aic_control_interfaces/msg/MotionUpdate"}},
                {"topic": {"name": "/scoring/insertion_event", "type": "std_msgs/msg/String"}},
                {"topic": {"name": "/aic_controller/controller_state", "type": "aic_control_interfaces/msg/ControllerState"}},
            ]
        },
        "task_board_limits": {
            "nic_rail": {
                "min_translation": args.nic_translation_range[0],
                "max_translation": args.nic_translation_range[1],
            },
            "sc_rail": {
                "min_translation": args.sc_translation_range[0],
                "max_translation": args.sc_translation_range[1],
            },
            "mount_rail": {
                "min_translation": args.mount_translation_range[0],
                "max_translation": args.mount_translation_range[1],
            },
        },
        "trials": {},
        "robot": {
            "home_joint_positions": {
                "shoulder_pan_joint": -0.1597,
                "shoulder_lift_joint": -1.3542,
                "elbow_joint": -1.6648,
                "wrist_1_joint": -1.6933,
                "wrist_2_joint": 1.5710,
                "wrist_3_joint": 1.4110,
            }
        },
    }

    for i in range(1, args.num_trials + 1):
        config["trials"][f"trial_{i}"] = make_trial(i, args)

    return config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate randomized AIC engine config YAML.")

    p.add_argument("--output", required=True, help="Output YAML filename")
    p.add_argument("--num-trials", type=int, required=True, help="Number of trials to generate")
    p.add_argument("--seed", type=int, default=0, help="Random seed")

    p.add_argument("--task-types", choices=["sfp", "sc", "both"], default="both")
    p.add_argument("--cable-types", choices=["sfp_sc_cable", "sfp_sc_cable_reversed", "both"], default="both")
    p.add_argument("--time-limit", type=int, default=180)

    p.add_argument("--board-x-range", nargs=2, default=(0.15, 0.15))
    p.add_argument("--board-y-range", nargs=2, default=(-0.2, -0.2))
    p.add_argument("--board-z-range", nargs=2, default=(1.14, 1.14))
    p.add_argument("--board-roll-range", nargs=2, default=(0.0, 0.0))
    p.add_argument("--board-pitch-range", nargs=2, default=(0.0, 0.0))
    p.add_argument("--board-yaw-range", nargs=2, default=(3.1415, 3.1415))

    p.add_argument("--nic-count-range", nargs=2, default=(1, 1))
    p.add_argument("--sc-count-range", nargs=2, default=(1, 1))
    p.add_argument("--lc-mount-count-range", nargs=2, default=(0, 2))
    p.add_argument("--sfp-mount-count-range", nargs=2, default=(0, 2))
    p.add_argument("--sc-mount-count-range", nargs=2, default=(0, 2))

    p.add_argument("--nic-translation-range", nargs=2, default=(-0.0215, 0.0234))
    p.add_argument("--nic-yaw-range-deg", nargs=2, default=(-10.0, 10.0))
    p.add_argument("--sc-translation-range", nargs=2, default=(-0.06, 0.055))
    p.add_argument("--mount-translation-range", nargs=2, default=(-0.09425, 0.09425))
    p.add_argument("--mount-yaw-range-deg", nargs=2, default=(-60.0, 60.0))

    p.add_argument(
        "--normal-gripper-offset",
        type=parse_offset,
        default=CablePose(0.0, 0.015385, 0.04545, 0.4432, -0.4838, 1.3303),
        help="x,y,z,roll,pitch,yaw for sfp_sc_cable",
    )
    p.add_argument(
        "--reversed-gripper-offset",
        type=parse_offset,
        default=CablePose(0.0, 0.015385, 0.04045, 0.4432, -0.4838, 1.3303),
        help="x,y,z,roll,pitch,yaw for sfp_sc_cable_reversed",
    )

    return p


def main() -> None:
    global args
    parser = build_parser()
    args = parser.parse_args()

    if args.num_trials <= 0:
        raise SystemExit("--num-trials must be > 0")

    # Convert CLI range lists to typed tuples
    args.board_x_range = two_floats(args.board_x_range)
    args.board_y_range = two_floats(args.board_y_range)
    args.board_z_range = two_floats(args.board_z_range)
    args.board_roll_range = two_floats(args.board_roll_range)
    args.board_pitch_range = two_floats(args.board_pitch_range)
    args.board_yaw_range = two_floats(args.board_yaw_range)

    args.nic_count_range = two_ints(args.nic_count_range)
    args.sc_count_range = two_ints(args.sc_count_range)
    args.lc_mount_count_range = two_ints(args.lc_mount_count_range)
    args.sfp_mount_count_range = two_ints(args.sfp_mount_count_range)
    args.sc_mount_count_range = two_ints(args.sc_mount_count_range)

    args.nic_translation_range = two_floats(args.nic_translation_range)
    args.sc_translation_range = two_floats(args.sc_translation_range)
    args.mount_translation_range = two_floats(args.mount_translation_range)

    nic_yaw_deg = two_floats(args.nic_yaw_range_deg)
    mount_yaw_deg = two_floats(args.mount_yaw_range_deg)
    args.nic_yaw_range_rad = (deg_to_rad(nic_yaw_deg[0]), deg_to_rad(nic_yaw_deg[1]))
    args.mount_yaw_range_rad = (deg_to_rad(mount_yaw_deg[0]), deg_to_rad(mount_yaw_deg[1]))

    random.seed(args.seed)

    cfg = build_config(args)

    out_path = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(dump_yaml(cfg))
        f.write("\n")

    print(f"Wrote config to: {out_path}")
    print(f"Trials: {args.num_trials}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()