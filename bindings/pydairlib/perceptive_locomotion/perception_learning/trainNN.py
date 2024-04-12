"""
Train a policy for //bindings/pydairlib/perceptive_locomotion/perception_learning:DrakeCassieEnv
"""
import argparse
import os
from os import path
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Tuple

import stable_baselines3
from stable_baselines3.common.env_checker import check_env

from pydairlib.perceptive_locomotion.perception_learning.PPO.ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecVideoRecorder,
    VecNormalize,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

import torch as th
import torch.nn as nn

_full_sb3_available = True

from pydrake.systems.all import (
    Diagram,
    Context,
    Simulator,
    InputPort,
    OutputPort,
    DiagramBuilder,
    InputPortIndex,
    OutputPortIndex,
    ConstantVectorSource
)

from pydairlib.perceptive_locomotion.systems.cassie_footstep_controller_gym_environment import (
    CassieFootstepControllerEnvironmentOptions,
    CassieFootstepControllerEnvironment,
)

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
"""
class CustomNetwork(nn.Module):
    def __init__(self, last_layer_dim_pi: int = 32, last_layer_dim_vf: int = 128):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # CNN for heightmap observations
        n_input_channels = 3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # MLP for ALIP state
        alip_state_dim = 4
        self.actor_alip_mlp = nn.Sequential(
            nn.Linear(alip_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.critic_alip_mlp = nn.Sequential(
            nn.Linear(alip_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Combined MLP for actor
        self.actor_combined_mlp = nn.Sequential(
            nn.Linear(320, 64),  # Updated input size
            nn.ReLU(),
            nn.Linear(64, self.latent_dim_pi),
            nn.ReLU(),
        )

        # Combined MLP for critic
        self.critic_combined_mlp = nn.Sequential(
            nn.Linear(320, 256),  # Updated input size
            nn.ReLU(),
            nn.Linear(256, self.latent_dim_vf),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(observations), self.forward_critic(observations)

    def forward_actor(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size(0)
        image_obs = observations[:, 0:3 * 64 * 64].reshape(batch_size, 3, 64, 64)
        actor_cnn_output = self.cnn(image_obs)  # Shape: (batch_size, 2304)
        alip_state = observations[:, 3 * 64 * 64:]
        actor_alip_mlp_output = self.actor_alip_mlp(alip_state)  # Shape: (batch_size, 64)
        actor_combined_features = th.cat((actor_cnn_output, actor_alip_mlp_output), dim=1)  # Concatenate along feature dimension
        actor_actions = self.actor_combined_mlp(actor_combined_features)
        return actor_actions

    def forward_critic(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size(0)
        image_obs = observations[:, :3 * 64 * 64].reshape(batch_size, 3, 64, 64)
        critic_cnn_output = self.cnn(image_obs)  # Shape: (batch_size, 2304)
        alip_state = observations[:, 3 * 64 * 64:]
        critic_alip_mlp_output = self.critic_alip_mlp(alip_state)  # Shape: (batch_size, 64)
        critic_combined_features = th.cat((critic_cnn_output, critic_alip_mlp_output), dim=1)  # Concatenate along feature dimension
        critic_actions = self.critic_combined_mlp(critic_combined_features)
        return critic_actions
"""
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class CustomNetwork(nn.Module):
    def __init__(self, last_layer_dim_pi: int = 128, last_layer_dim_vf: int = 128):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # CNN for heightmap observations
        n_input_channels = 3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # MLP for ALIP state
        alip_state_dim = 4
        self.alip_mlp = nn.Sequential(
            layer_init(nn.Linear(alip_state_dim, 64)),
            nn.ReLU(),  # Using ReLU activation for ALIP MLP
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),  # Using ReLU activation for ALIP MLP
        )

        # Combined MLP for actor
        self.actor_combined_mlp = nn.Sequential(
            layer_init(nn.Linear(320, 128)),  # Updated input size
            nn.ReLU(),  # Using ReLU activation
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),  # Using ReLU activation
            layer_init(nn.Linear(128, self.latent_dim_pi), std = 0.03),
            nn.ReLU(),  # Using ReLU activation
        )

        # Combined MLP for critic
        self.critic_combined_mlp = nn.Sequential(
            layer_init(nn.Linear(320, 128)),  # Updated input size
            nn.ReLU(),  # Using ReLU activation
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),  # Using ReLU activation
            layer_init(nn.Linear(128, self.latent_dim_vf), std = 1.),
            nn.ReLU(),  # Using ReLU activation
        )

    def forward(self, observations: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(observations), self.forward_critic(observations)

    def forward_actor(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size(0)
        image_obs = observations[:, 0:3 * 64 * 64].reshape(batch_size, 3, 64, 64)
        actor_cnn_output = self.cnn(image_obs)  # Shape: (batch_size, 2304)
        alip_state = observations[:, 3 * 64 * 64:]
        actor_alip_mlp_output = self.alip_mlp(alip_state)  # Shape: (batch_size, 64)
        actor_combined_features = th.cat((actor_cnn_output, actor_alip_mlp_output), dim=1)  # Concatenate along feature dimension
        actor_actions = self.actor_combined_mlp(actor_combined_features)
        return actor_actions

    def forward_critic(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.size(0)
        image_obs = observations[:, :3 * 64 * 64].reshape(batch_size, 3, 64, 64)
        critic_cnn_output = self.cnn(image_obs)  # Shape: (batch_size, 2304)
        alip_state = observations[:, 3 * 64 * 64:]
        critic_alip_mlp_output = self.alip_mlp(alip_state)  # Shape: (batch_size, 64)
        critic_combined_features = th.cat((critic_cnn_output, critic_alip_mlp_output), dim=1)  # Concatenate along feature dimension
        critic_actions = self.critic_combined_mlp(critic_combined_features)
        return critic_actions

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        optimizer_class= th.optim.AdamW,
        optimizer_kwargs = { 'weight_decay': 1e-3, 'epsilon': 1e-5},
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = True # False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork()

def bazel_chdir():
    """When using `bazel run`, the current working directory ("cwd") of the
    program is set to a deeply-nested runfiles directory, not the actual cwd.
    In case relative paths are given on the command line, we need to restore
    the original cwd so that those paths resolve correctly.
    """
    if 'BUILD_WORKSPACE_DIRECTORY' in os.environ:
        os.chdir(os.environ['BUILD_WORKSPACE_DIRECTORY'])

def _run_training(config, args):
    env_name = config["env_name"]
    num_env = config["num_workers"]
    log_dir = config["local_log_dir"]
    policy_type = config["policy_type"]
    total_timesteps = config["total_timesteps"]
    eval_freq = config["model_save_freq"]
    sim_params = config["sim_params"]

    if not args.train_single_env:
        input("Starting...")
        sim_params.visualize = False
        env = make_vec_env(
                           env_name,
                           n_envs=num_env,
                           seed=0,
                           vec_env_cls=SubprocVecEnv,
                           env_kwargs={
                               'sim_params': sim_params,
                           })
        #env = VecNormalize(env)
        #env = VecMonitor(env)
    else:
        input("Starting...")
        sim_params.visualize = False
        if args.test:
            sim_params.visualize = True
        env = gym.make(env_name,
                       sim_params = sim_params,
                       )
        #env = Monitor(env)

    if args.test:
        model = PPO(policy_type, env, n_steps=128, n_epochs=2,
                    batch_size=32,)# policy_kwargs=policy_kwargs,)
    else:
        tensorboard_log = f"{log_dir}runs/test"        
        #model = PPO(
        #    policy_type, env, learning_rate = linear_schedule(0.003), n_steps=int(2048/num_env), n_epochs=10,
        #    batch_size=64*num_env, ent_coef=0.03,
        #    verbose=1,
        #    tensorboard_log=tensorboard_log,)
        #test_folder = "rl/tmp/DrakeCassie/eval_logs/test/vision_separate"
        #model_path = path.join(test_folder, 'best_model.zip')
        #model_path = 'PPO_studentNN_stairimitation.zip'
        model_path = 'PPO_studentNN_tanh'
        model = PPO.load(model_path, env, learning_rate = linear_schedule(3e-7), max_grad_norm = 0.05,
                        clip_range = linear_schedule(0.05), target_kl = 0.003, ent_coef=0.01, 
                        n_steps=int(4096/num_env), n_epochs=5,
                        batch_size=256*num_env, seed=42, device='auto',
                        tensorboard_log=tensorboard_log)
        
        print("Open tensorboard (optional) via "
              f"`tensorboard --logdir {tensorboard_log}` "
              "in another terminal.")

    sim_params.visualize = True
    eval_env = gym.make(env_name,
                        sim_params = sim_params,
                        )

    eval_env = DummyVecEnv([lambda: eval_env])
    #eval_env = VecMonitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir+f'eval_logs/test',
        log_path=log_dir+f'eval_logs/test',
        eval_freq=eval_freq,
        deterministic=True,
        render=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq*3,
        save_path="./logs/",
        name_prefix="rl_model",
    )

    callback = CallbackList([checkpoint_callback, eval_callback])

    input("Model learning...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

    model.save("latest_model")
    eval_env.close()

class ReLUSquared(nn.Module):
    def forward(self, input):
        return th.relu(input) ** 2

def _main():
    bazel_chdir()
    sim_params = CassieFootstepControllerEnvironmentOptions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train_single_env', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_path', help="path to the logs directory.",
                        default="./rl/tmp/DrakeCassie/")
    args = parser.parse_args()

    if not _full_sb3_available:
        print("stable_baselines3 found, but was drake internal")
        return 0 if args.test else 1

    if args.test:
        num_env = 1
    elif args.train_single_env:
        num_env = 1
    else:
        num_env = 16

    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    config = {
        "policy_type": CustomActorCriticPolicy,
        "total_timesteps": 1500000 if not args.test else 5, # 2e6
        "env_name": "DrakeCassie-v0",
        "num_workers": num_env,
        "local_log_dir": args.log_path,
        "model_save_freq": 3000,
        #"policy_kwargs": {'activation_fn': ReLUSquared, #th.nn.Tanh,        # activation function | th.nn.ReLU,
        #                  'net_arch': {'pi': [64, 64, 64, 64], # policy and value networks
        #                               'vf': [64, 64, 64, 64]}},
        "sim_params" : sim_params
    }
    _run_training(config, args)

gym.envs.register(
        id="DrakeCassie-v0",
        entry_point="pydairlib.perceptive_locomotion.perception_learning.DrakeCassieEnv:DrakeCassieEnv")

if __name__ == '__main__':
    _main()
