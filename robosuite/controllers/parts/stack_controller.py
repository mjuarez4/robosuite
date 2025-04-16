import abc
from collections.abc import Iterable

import mujoco
import numpy as np
from .controller import Controller


import robosuite.macros as macros
class StackController(Controller):
    def __init__(self, stack_env):
        robot = stack_env.robots[0]

        super().__init__(
            sim=stack_env.sim,
            joint_indexes=robot.joint_indexes,
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
        offset = cube_pos - eef_pos
        return offset

    def run_controller(self):
        # Placeholder for now
        return np.zeros(len(self.joint_index))

    @property
    def name(self):
        return "StackController"
