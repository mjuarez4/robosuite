import abc
from collections.abc import Iterable

import mujoco
import numpy as np
from .controller import Controller



import robosuite.macros as macros
def joint_qpos_ids(model, joint_names):
    return [model.joint_name2id(name) for name in joint_names]

def joint_qvel_ids(model, joint_names):
    return [model.joint_name2id(name) for name in joint_names]


class StackController(Controller):
    def __init__(self, stack_env):
        robot = stack_env.robots[0]
        sim = stack_env.sim

        # ✅ Get joint IDs and names
        joint_ids = robot.joint_indexes
        joint_names = [sim.model.joint_id2name(jid) for jid in joint_ids]

        # ✅ Build joint index mapping
        joint_indexes = {
            "joints": joint_ids,
            "qpos": joint_ids,  # assuming 1-DoF joints
            "qvel": joint_ids,
        }

        # ✅ Extract actuator range
        prefix = robot.name  # 'robot0'
        actuator_names = [name for name in sim.model.actuator_names if name.startswith(prefix)]
        actuator_ids = [sim.model.actuator_name2id(name) for name in actuator_names]
        actuator_ctrlrange = sim.model.actuator_ctrlrange
        actuator_mins = actuator_ctrlrange[actuator_ids, 0]
        actuator_maxs = actuator_ctrlrange[actuator_ids, 1]
        actuator_range = (actuator_mins, actuator_maxs)

        ref_name = "gripper0_right_grip_site"
        # ✅ this will give you something like "gripper0_grip_site"
        print("REF NAME:", ref_name, type(ref_name))


        super().__init__(
            sim=sim,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            ref_name=ref_name,  # ✅ this is crucial for setting ref_pos
            lite_physics=False,
        )


        self.env = stack_env



    def position_above_block_A(self):
        self.update()  # updates self.ref_pos, self.ref_ori_mat, etc.

        cube_pos = self.env.sim.data.body_xpos[self.env.cubeA_body_id]
        eef_pos = self.ref_pos  # already updated by self.update()

        print("EEF position:", eef_pos)
        print("Cube A position:", cube_pos)

        # Optional: return vector toward the cube
        offset = cube_pos - eef_pos  # shape (3,)
        action = np.concatenate([offset, np.zeros(4)])  # shape (7,)

        return action

    def run_controller(self):
        # Placeholder for now
        return np.zeros(len(self.joint_index))

    @property
    def name(self):
        return "StackController"
