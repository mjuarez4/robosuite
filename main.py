import os

import imageio
import jax
import numpy as np
from rich.pretty import pprint
from tqdm import tqdm
from robosuite.controllers.parts.stack_controller import StackController


import robosuite as suite
from robosuite.environments import MujocoEnv
from robosuite.utils.input_utils import (choose_environment,
                                         choose_multi_arm_config,
                                         choose_robots)

os.environ["MUJOCO_GL"] = "egl"

import math
from typing import Dict, List, Union

import cv2
import numpy as np


def writekeys(imgs: Dict):
    return {
        k: cv2.putText(
            imgs[k],
            f"{k}",
            (0, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )
        for k in imgs
    }


def tile_images(images, pad_value=0):

    if isinstance(images, list):
        images = np.stack(images, axis=0)

    N, H, W, C = images.shape
    grid_cols = math.ceil(math.sqrt(N))
    grid_rows = math.ceil(N / grid_cols)

    # Pad if needed
    total_needed = grid_rows * grid_cols
    if total_needed > N:
        pad_imgs = np.full((total_needed - N, H, W, C), pad_value, dtype=images.dtype)
        images = np.concatenate([images, pad_imgs], axis=0)

    # Reshape and tile
    images = images.reshape(grid_rows, grid_cols, H, W, C)
    images = images.transpose(0, 2, 1, 3, 4)  # (grid_rows, H, grid_cols, W, C)
    tiled_image = images.reshape(grid_rows * H, grid_cols * W, C)

    return tiled_image


def main():

    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    options = {}
    # options["env_configuration"] = choose_multi_arm_config()
    options["env_name"] = choose_environment()
    assert "TwoArm" not in options["env_name"], "TwoArm envs are not for you"
    options["robots"] = choose_robots(
        exclude_bimanual=False, use_humanoids=True, exclude_single_arm=False
    )

    camera_names = [
        "frontview",
        "birdview",
        "agentview",
        "sideview",
        "shouldercamera0",
        "shouldercamera1",
        "robot0_robotview",
        "robot0_eye_in_hand",
        "robot1_robotview",
        "robot1_eye_in_hand",
    ]

    camera_names = [
        "frontview",
        "birdview",
        "agentview",
        "sideview",
        "robot0_robotview",
        "robot0_eye_in_hand",
    ]

    # create environment instance
    env: MujocoEnv = suite.make(
        **options,
        # env_name="TwoArmLift",  # try with other tasks like "Stack" and "Door"
        # env_name="TwoArmTransport",  # try with other tasks like "Stack" and "Door"
        # robots=["XArm7", "XArm7"],  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=50,
        camera_names=camera_names,
        renderer="mujoco",
    )
    env.reset()
    
    controller = StackController(env)

    
    # camera_name = ["agentview", "birdview", "all-eye_in_hand", "robot_view"]
    # env.viewer.set_camera(camera_name=names)

    spec = lambda arr: jax.tree.map(lambda _x: _x.shape, arr)
    #pprint(spec(dict(obs)))
    frames = []

    for i in tqdm(range(100)):
        #action = np.random.randn(*env.action_spec[0].shape) * 1
        #obs, reward, done, info = env.step(action)  # take action in the environment
        action = controller.position_above_block_A()
        obs, reward, done, info = env.step(action)
        # out = env.render()  # render on display

        # input('Input: ... ')

        imshape = obs["agentview_image"].shape
        # render upside down
        imgs = writekeys(
            {
                k: np.flip(v, axis=0).astype(np.uint8)
                for k, v in obs.items()
                if v.shape == imshape
            }
        )
        pprint(spec(imgs))
        frame = tile_images(list(imgs.values()))

        frames.append(frame)
        pprint(frames[-1].shape)

    imageio.mimsave("output.mp4", frames, fps=30)

    env.close()


if __name__ == "__main__":
    main()
