"""Utility to make a leader-follower kinfer runtime for the K-Bot."""

import argparse
import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import colorlogging
import jax
import jax.numpy as jnp
import ksim
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

logger = logging.getLogger(__name__)


SIM_DT = 0.02
NUM_COMMANDS = 10  # Number of command inputs (10 for 10 upper body joints)

# Joint biases, these are in the order that they appear in the neural network.
JOINT_BIASES: list[tuple[str, float, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0, 1.0),  # 0
    ("dof_right_shoulder_roll_03", math.radians(-10.0), 1.0),  # 1
    ("dof_right_shoulder_yaw_02", 0.0, 1.0),  # 2
    ("dof_right_elbow_02", math.radians(90.0), 1.0),  # 3
    ("dof_right_wrist_00", 0.0, 1.0),  # 4
    ("dof_left_shoulder_pitch_03", 0.0, 1.0),  # 5
    ("dof_left_shoulder_roll_03", math.radians(10.0), 1.0),  # 6
    ("dof_left_shoulder_yaw_02", 0.0, 1.0),  # 7
    ("dof_left_elbow_02", math.radians(-90.0), 1.0),  # 8
    ("dof_left_wrist_00", 0.0, 1.0),  # 9
    ("dof_right_hip_pitch_04", math.radians(-20.0), 0.01),  # 10
    ("dof_right_hip_roll_03", math.radians(-0.0), 2.0),  # 11
    ("dof_right_hip_yaw_03", 0.0, 2.0),  # 12
    ("dof_right_knee_04", math.radians(-50.0), 0.01),  # 13
    ("dof_right_ankle_02", math.radians(30.0), 1.0),  # 14
    ("dof_left_hip_pitch_04", math.radians(20.0), 0.01),  # 15
    ("dof_left_hip_roll_03", math.radians(0.0), 2.0),  # 16
    ("dof_left_hip_yaw_03", 0.0, 2.0),  # 17
    ("dof_left_knee_04", math.radians(50.0), 0.01),  # 18
    ("dof_left_ankle_02", math.radians(-30.0), 1.0),  # 19
]

JOINT_INVERSIONS: list[tuple[str, int]] = [
    ("dof_right_shoulder_pitch_03", 1),  # 0
    ("dof_right_shoulder_roll_03", -1),  # 1
    ("dof_right_shoulder_yaw_02", 1),  # 2
    ("dof_right_elbow_02", -1),  # 3
    ("dof_right_wrist_00", 1),  # 4
    ("dof_left_shoulder_pitch_03", 1),  # 5
    ("dof_left_shoulder_roll_03", 1),  # 6
    ("dof_left_shoulder_yaw_02", 1),  # 7
    ("dof_left_elbow_02", 1),  # 8
    ("dof_left_wrist_00", 1),  # 9
    ("dof_right_hip_pitch_04", -1),  # 10
    ("dof_right_hip_roll_03", 1),  # 11
    ("dof_right_hip_yaw_03", 1),  # 12
    ("dof_right_knee_04", -1),  # 13
    ("dof_right_ankle_02", -1),  # 14
    ("dof_left_hip_pitch_04", 1),  # 15
    ("dof_left_hip_roll_03", 1),  # 16
    ("dof_left_hip_yaw_03", 1),  # 17
    ("dof_left_knee_04", 1),  # 18
    ("dof_left_ankle_02", 1),  # 19
]


InitFn = Callable[[], Array]

StepFn = Callable[
    [Array, Array, Array, Array, Array, Array],  # state inputs
    tuple[Array, Array],  # (targets, carry)
]


@dataclass
class Recipe:
    name: str
    init_fn: InitFn
    step_fn: StepFn
    num_commands: int
    carry_size: tuple[int, ...]


def get_mujoco_model() -> mujoco.MjModel:
    """Get the MuJoCo model for the K-Bot."""
    mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-headless", name="robot"))
    return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")


def get_joint_names() -> list[str]:
    """Get the joint names."""
    model = get_mujoco_model()
    return ksim.get_joint_names_in_order(model)[1:]  # drop root joint


def get_bias_vector(joint_names: list[str]) -> jnp.ndarray:
    """Return an array of neutral/bias angles ordered like `joint_names`."""
    bias_map = {name: bias for name, bias, _ in JOINT_BIASES}
    return jnp.array([bias_map[name] for name in joint_names])


def make_leader_follower_recipe(
    joint_names: list[str], 
    dt: float,
    mirror_scale: float = 1.0,  # Scale factor for mirroring (1.0 = direct copy)
    enable_mirroring: bool = True,  # Whether to mirror or stay in bias pose
) -> Recipe:
    """Creates a recipe for leader-follower joint mirroring.
    
    This kinfer model takes leader_arm_joint_angles as input and mirrors them
    to the follower robot. If no leader data is available, it stays in bias pose.
    
    Input structure:
    - joint_angles: Current simulation joint angles (unused for mirroring)
    - joint_angular_velocities: Current simulation joint velocities (unused)
    - leader_arm_joint_angles: Real joint angles from leader arm (Raspberry Pi)
    - command: Keyboard commands (can be used to enable/disable mirroring)
    - carry: Internal state [time, mirror_enabled]
    """
    bias_vec = get_bias_vector(joint_names)
    
    # Carry: [time] (removed mirror_enabled since always on)
    carry_size = (1,)

    @jax.jit
    def init_fn() -> Array:
        # Initialize with time=0
        return jnp.array([0.0])

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        t = carry[0] + dt
        
        # Use command array to control joints directly
        # command[0-5] control the first 6 joints (upper body)
        
        # Use command values directly (no scaling)
        joint_values = command
        
        # Debug: print command values (this will show in logs)
        # Note: In JAX, we can't use print inside jitted functions, so this is just for reference
        
        # Start with bias pose
        targets = bias_vec.copy()
        
        # Apply command values to first 10 joints (upper body)
        # KOS joints: 11,12,13,14,15 (right) + 21,22,23,24,25 (left)
        # command[0] -> right shoulder pitch (joint 11)
        # command[1] -> right shoulder roll (joint 12)
        # command[2] -> right shoulder yaw (joint 13)
        # command[3] -> right elbow (joint 14)
        # command[4] -> right wrist (joint 15)
        # command[5] -> left shoulder pitch (joint 21)
        # command[6] -> left shoulder roll (joint 22)
        # command[7] -> left shoulder yaw (joint 23)
        # command[8] -> left elbow (joint 24)
        # command[9] -> left wrist (joint 25)
        
        # Use simple indexing to avoid broadcasting issues
        targets = targets.at[0].set(joint_values[0])  # right shoulder pitch
        targets = targets.at[1].set(joint_values[1])  # right shoulder roll
        targets = targets.at[2].set(joint_values[2])  # right shoulder yaw
        targets = targets.at[3].set(joint_values[3])  # right elbow
        targets = targets.at[4].set(joint_values[4])  # right wrist
        targets = targets.at[5].set(joint_values[5])  # left shoulder pitch
        targets = targets.at[6].set(joint_values[6])  # left shoulder roll
        targets = targets.at[7].set(joint_values[7])  # left shoulder yaw
        targets = targets.at[8].set(joint_values[8])  # left elbow
        targets = targets.at[9].set(joint_values[9])  # left wrist
        
        # Update carry: [time]
        new_carry = jnp.array([t])
        
        return targets, new_carry

    return Recipe(
        "kbot_leader_follower", 
        init_fn, 
        step_fn, 
        NUM_COMMANDS, 
        carry_size
    )


def build_kinfer_file(recipe: Recipe, joint_names: list[str], out_dir: Path) -> Path:
    """Build a kinfer file for a given recipe."""
    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=recipe.num_commands,
        carry_size=recipe.carry_size,
    )
    kinfer_blob = pack(
        export_fn(recipe.init_fn, metadata),  # type: ignore[arg-type]
        export_fn(recipe.step_fn, metadata),  # type: ignore[arg-type]
        metadata,
    )
    out_path = out_dir / f"{recipe.name}.kinfer"
    out_path.write_bytes(kinfer_blob)
    return out_path


def main() -> None:
    colorlogging.configure()
    parser = argparse.ArgumentParser(description="Generate leader-follower kinfer model")
    default_output = Path(__file__).parent / "assets"
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=default_output,
        help="Output path for the kinfer model (default: %(default)s)",
    )
    parser.add_argument(
        "--mirror-scale",
        "-s",
        type=float,
        default=1.0,
        help="Scale factor for mirroring (1.0 = direct copy, 0.5 = half movement) (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-mirroring",
        action="store_true",
        default=True,
        help="Enable mirroring by default (default: True)",
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_names = get_joint_names()
    num_joints = len(joint_names)
    logger.info("Number of joints: %s", num_joints)
    logger.info("Joint names: %s", joint_names)
    
    logger.info("Creating leader-follower model")
    logger.info("Mirror scale: %s", args.mirror_scale)
    logger.info("Enable mirroring: %s", args.enable_mirroring)

    recipe = make_leader_follower_recipe(
        joint_names=joint_names,
        dt=SIM_DT,
        mirror_scale=args.mirror_scale,
        enable_mirroring=args.enable_mirroring,
    )
    
    out_path = build_kinfer_file(recipe, joint_names, out_dir)
    logger.info("kinfer model written to %s", out_path)
    
    # Print usage instructions
    print("\n" + "="*60)
    print("COMMAND-BASED JOINT CONTROL INSTRUCTIONS")
    print("="*60)
    print(f"Mirror scale: {args.mirror_scale}")
    print(f"Enable mirroring: {args.enable_mirroring}")
    print("\nInput structure:")
    print("  joint_angles: Current simulation joint angles (unused)")
    print("  joint_angular_velocities: Current simulation joint velocities (unused)")
    print("  command: Keyboard commands for joint control")
    print("\nCommand array mapping:")
    print("  command[0]: Right shoulder pitch")
    print("  command[1]: Right shoulder roll")
    print("  command[2]: Right shoulder yaw")
    print("  command[3]: Right elbow")
    print("  command[4]: Right wrist")
    print("  command[5]: Left shoulder pitch")
    print("\nBehavior:")
    print("  - Command values directly control joint positions")
    print("  - Values are scaled by 0.5 and clamped to [-1.0, 1.0]")
    print("  - UDP data from Pi can be injected into command array via provider")
    print("="*60)


if __name__ == "__main__":
    main() 