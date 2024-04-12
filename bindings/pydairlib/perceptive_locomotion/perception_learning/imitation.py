"""
Imitate a policy from supervising LQR dataset //bindings/pydairlib/perceptive_locomotion/perception_learning:DrakeCassieEnv
"""
import argparse
import os
from os import path

import gymnasium as gym
from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import Dataset, random_split

from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
import stable_baselines3
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
)
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

def bazel_chdir():
    """When using `bazel run`, the current working directory ("cwd") of the
    program is set to a deeply-nested runfiles directory, not the actual cwd.
    In case relative paths are given on the command line, we need to restore
    the original cwd so that those paths resolve correctly.
    """
    if 'BUILD_WORKSPACE_DIRECTORY' in os.environ:
        os.chdir(os.environ['BUILD_WORKSPACE_DIRECTORY'])

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
        
    def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

    def __len__(self):
        return len(self.observations)

def split(expert_observations, expert_actions):
    expert_dataset = ExpertDataSet(expert_observations, expert_actions)

    train_size = int(0.8 * len(expert_dataset))

    test_size = len(expert_dataset) - train_size

    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )
    return train_expert_dataset, test_expert_dataset

def pretrain_agent(
    student,
    env,
    train_expert_dataset,
    test_expert_dataset,
    batch_size=64,
    epochs=10,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=64,
):
    use_cuda = not no_cuda and th.cuda.is_available()
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
      criterion = nn.MSELoss()
    else:
      criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    def train(model, device, train_loader, optimizer):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
              # A2C/PPO policy outputs actions, values, log_prob
              # SAC/TD3 policy outputs actions only
              if isinstance(student, (A2C, PPO)):
                action, _, _ = model(data)
              else:
                # SAC/TD3:
                action = model(data)
              action_prediction = action.double()
            else:
              # Retrieve the logits for A2C/PPO when using discrete actions
              dist = model.get_distribution(data)
              action_prediction = dist.distribution.logits
              target = target.long()

            loss = criterion(action_prediction, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if isinstance(env.action_space, gym.spaces.Box):
                  # A2C/PPO policy outputs actions, values, log_prob
                  # SAC/TD3 policy outputs actions only
                  if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                  else:
                    # SAC/TD3:
                    action = model(data)
                  action_prediction = action.double()
                else:
                  # Retrieve the logits for A2C/PPO when using discrete actions
                  dist = model.get_distribution(data)
                  action_prediction = dist.distribution.logits
                  target = target.long()

                test_loss = criterion(action_prediction, target)
        test_loss /= len(test_loader.dataset)
        #print(f"Test set: Average loss: {test_loss:.4f}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing
    train_loader = th.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = th.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=True, **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    student.policy = model
    obs = np.array([-6.68755563e-02, -9.75012432e-02,  3.40785827e-01, -3.92377606e-01])
    #[0.21599751, 0.23242172, 0.        ]
    action, _ = model.predict(obs, deterministic=True)
    print(action)

class ReLUSquared(nn.Module):
    def forward(self, input):
        return th.relu(input) ** 2

def _main():
    bazel_chdir()
    sim_params = CassieFootstepControllerEnvironmentOptions()
    gym.envs.register(
        id="DrakeCassie-v0",
        entry_point="pydairlib.perceptive_locomotion.perception_learning.DrakeCassieEnv:DrakeCassieEnv")  # noqa
    
    env = gym.make("DrakeCassie-v0",
                sim_params = sim_params,
                )

    student = PPO('MlpPolicy', env, verbose=1, policy_kwargs={'activation_fn': ReLUSquared,#th.nn.Tanh,# activation function |th.nn.ReLU, th.nn.Tanh,
                                                                    'net_arch': {'pi': [64, 64, 64, 64], # policy and value networks
                                                                                'vf': [64, 64, 64, 64]}},)
    obs_data = "state_action_pair/observations.npy"
    action_data = "state_action_pair/actions.npy"
    expert_observations = np.load(obs_data)
    expert_actions = np.load(action_data)

    train_expert_dataset, test_expert_dataset = split(expert_observations, expert_actions)
    print(len(train_expert_dataset))
    print(len(test_expert_dataset))
    
    pretrain_agent(
        student,
        env,
        train_expert_dataset,
        test_expert_dataset,
        epochs=15,
        scheduler_gamma=0.7,
        learning_rate=.3,
        log_interval=100,
        no_cuda=False,
        seed=1,
        batch_size=128,
        test_batch_size=128,
    )

    student.save("PPO_student")
    mean_reward, std_reward = evaluate_policy(student, env, n_eval_episodes=5)
    print(f"Mean reward = {mean_reward} +/- {std_reward}")



if __name__ == '__main__':
    _main()