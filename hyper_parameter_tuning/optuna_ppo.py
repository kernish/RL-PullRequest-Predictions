import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import optuna
import gym
from gym import Env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import torch.nn as nn

from gym.spaces import Discrete, Box, Dict 
from stable_baselines3 import PPO
from stable_baselines3.common.policies import obs_as_tensor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy



""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using PPO from Stable-Baselines3.

More info at: https://github.com/DLR-RM/rl-baselines3-zoo.
"""
class prEnv(gym.Env):
    '''
    custom environment
    '''

#--------------------------------------------------------------------------------------------------------------------------------------------

def optimize_ppo(trial):
    """Sampler for PPO hyperparameters."""
    
    gamma = 1.0 - trial.suggest_float("gamma", 0.0001, 0.1, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = 1.0 - trial.suggest_float("gae_lambda", 0.001, 0.2, log=True)
    n_steps = len(train)
    learning_rate = trial.suggest_float("lr", 1e-2, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_float('cliprange', 0.1, 0.4, log=True)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # Display true values
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("n_steps", n_steps)

    net_arch = [
        {"pi": [64], "vf": [64]} if net_arch == "tiny" else {"pi": [64, 64], "vf": [64, 64]}
    ]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "max_grad_norm": max_grad_norm,
        "policy_kwargs": {
            "net_arch": net_arch,
            "activation_fn": activation_fn,
            "ortho_init": ortho_init,
        },
    }
    


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we need to negate the reward here
    """
    model_params = optimize_ppo(trial)
    #env = make_vec_env(lambda: prEnv(train), n_envs=16, seed=0)
    env = prEnv(train)
    model = PPO('MlpPolicy', env, verbose=0, **model_params)
    model.learn(total_timesteps=steps) # steps = number of steps
    env.close()
    
    # just as an example
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=(eval_steps))
    
    del model
    return -1 * mean_reward


# read the train and test files
train = pd.read_csv("train_data.csv", encoding= "utf-8") 
test = pd.read_csv("test_data.csv", encoding= "utf-8")

study = optuna.create_study()
try:
    study.optimize(optimize_agent, n_trials=100, n_jobs=4) # for 100 trials
except KeyboardInterrupt:
    print('Interrupted by keyboard.')

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print("  User attrs:")
for key, value in trial.user_attrs.items():
    print("    {}: {}".format(key, value))