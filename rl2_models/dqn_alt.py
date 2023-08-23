#SLRUM Script
#!/bin/bash

#SBATCH --job-name=dqn_2p_8k #optional name to give to your job
#SBATCH -c 1 # Number of CPUS requested. If omitted, the default is 1 CPU.
#SBATCH --gpus-per-node=1
#SBATCH --mem=6G # Memory requested in megabytes. If omitted, the default is 1024 MB.
#SBATCH --time=00-03:30:00 # Expected job run time (killed afterwards) default 3hrs. 02-01:22:40
#SBATCH --account=def-kahani
#SBATCH --output=R-%x.out #%x.%j for job id %j
#SBATCH --error=R-%x.err

#CUDA_LAUNCH_BLOCKING=1 python dqn_alt.py train_comments_data.csv

# import all the packages
import os
import sys
import random
import time
import numpy as np
import pandas as pd
import collections
import torch as th
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score
import statistics
import traceback

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict
from stable_baselines3 import DQN


# Custom Environment: PR ENV
class prEnv(gym.Env):
    
    def __init__(self, DataSet):
        '''
        Initialize the object, set the variables
        '''
        # Actions we can take: accept: 1, reject: 0 (default state pre-action) request change: 2
        self.action_space = Discrete(2)
        
        # the dataset to fetch each entry of pull requests
        self.data = DataSet
        # individual PRs from the data
        self.groups_df = self.data.groupby(['owner_name','repo_name','pull_no','merged_or_not'])
        self.names_df = list(self.groups_df.groups)
        self.names_df_1 = list(self.groups_df.groups)

        # a variable to iterate through the records in the dataset
        self.idx = 0

        # list of actions taken for individual PR review stages within a PR
        self.actions_taken = []

        # Selecting the Influential features for pull requests
        self.observation_space = gym.spaces.Dict({ 
                                'compound':Box(low=np.array([-1]), high=np.array([1]), dtype='float'),
                                'has_code_element':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'merged_or_not':Box(low=np.array([0]), high=np.array([2]), dtype='int'),
                                'neg_vr':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'neu_vr':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'pos_vr':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'stopw_ratio':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'word_count':Box(low=np.array([0]), high=np.array([1600]), dtype='int')
                                })
         
    def step(self, action):
        '''
        Calculate rewards and next state based on the action. Reward offered immediately is Zero!! Dealyed non-zero reward
        Actions we can take: accept: 1, reject: 0
        '''
        # Apply action     
        if (action==1):
            self.state['merged_or_not'] = np.array([1])
        elif (action==0):
            self.state['merged_or_not'] = np.array([0])
        
        self.actions_taken.append(action)
        self.reward = 0
        self.done = False
        self.state = self.next_obs()
        
        # Calculate reward 
        # imbalance ratio = 0.26 # imbalanced data hence more reward for minority class
        # Trade-off factor = k = 2*IR 
        k = 2*0.26
        if self.done == True:
            self.effective_action = statistics.mode(self.actions_taken)
            if self.effective_action == self.expected_action:
                if self.effective_action == 1:
                    self.reward = k
                    self.done = False
                    self.state = self.next_obs()
                elif self.effective_action == 0:
                    self.reward = 1
                    self.done = False
                    self.state = self.next_obs()
            else: 
                if self.expected_action == 0:
                    self.reward = -1
                    self.done = False
                    self.state = self.next_obs()
                elif self.expected_action == 1:
                    self.reward = -1 * k
                    self.done = False
                    self.state = self.next_obs()

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, self.reward, self.done, info

    
    def next_obs(self):
        '''
        A function to pass the PR review stages as states
        And set the value of expected_action to help calculate the delayed reward!
        '''
        if self.idx < len(self.group):
            obs = self.group.iloc[[self.idx]].to_dict(orient='records')[0]
            obs = {key: np.array([obs[key]]) for key in obs if key not in ['owner_name','repo_name','created_at','pull_no','emotion_vr']}
            order_of_keys = ['compound', 'has_code_element', 'merged_or_not', 'neg_vr', 'neu_vr', 'pos_vr', 'stopw_ratio', 'word_count']
            list_of_tuples = [(key, obs[key]) for key in order_of_keys]
            obs = collections.OrderedDict(list_of_tuples)
            
            self.idx += 1
            obs['merged_or_not'] = np.array([2])
            return obs
        else:
            obs = self.state
            self.done = True
    
            return obs
   

    def render(self):
        # Implement viz
        pass
    
    
    def reset(self):
        '''
        A function to reset the environment after each episode!
        '''
        # set the index to 0
        self.idx = 0
        self.actions_taken = []
        try:
            self.name = self.names_df.pop(0)
        except:
            self.names_df = self.names_df_1.copy()
            random.shuffle(self.names_df)
            self.name = self.names_df.pop(0)
        self.group = self.groups_df.get_group(self.name)
        self.group = self.group.sort_values(by='created_at', ascending=True)
        # fetch the merged_or_not for the PR
        self.expected_action = int(self.name[-1])
        # Reset env
        self.state = self.next_obs()

        return self.state

def bad_line(x):
    print("--------------------------------------------bad line st--------------------------------------------")
    print(x)
    print("--------------------------------------------bad line ed--------------------------------------------")
    return None


if __name__=="__main__":
    '''
    The main() funtcion of the script. First called when running the script
    Returns: None
    '''

    # load train and test data
    train = pd.read_csv(str("path"+str("train_comments_data.csv")), encoding= "utf-8", on_bad_lines= bad_line, engine="python") 
    test = pd.read_csv(str("path"+str("test_comments_data.csv")), encoding= "utf-8", on_bad_lines= bad_line, engine="python") 

    try:
        train['created_at'] = pd.to_datetime(train['created_at'], infer_datetime_format=True)
        test['created_at'] = pd.to_datetime(test['created_at'], infer_datetime_format=True)
    except:
        traceback.print_exc()

    # drop raw comments 
    col_drop = ['body','body_processed']
    train = train.drop(columns=col_drop)
    test = test.drop(columns=col_drop)
    #print(f"@train {train.shape} of len: {len(train)}")
    #print(f"@test {test.shape} of len: {len(test)}")

    pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)

    #--------------------------------------------------------------------------------------------------------------------------------------------
    # set the path for logs
    log_path = os.path.join('training','logs')
    
    env = prEnv(train)

    # tweak the total_timesteps using the following
    time_steps_learn = 800000
        
    model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path)

    # train the model
    model.learn(total_timesteps=time_steps_learn)
    
    #--------------------------------------------------------------------------------------------------------------------------------------------
    # set the out path to save the trained model
    out_path = os.path.join('training', 'SavedModels', 'DQN_Model_2p')
    model.save(out_path)
    
    del model
    
    #--------------------------------------------------------------------------------------------------------------------------------------------
    # load the saved model
    model = DQN.load(out_path, env)

    true_class = []
    predicted_class = []

    groups_test = test.groupby(['owner_name','repo_name','pull_no','merged_or_not'])
    names_test = list(groups_test.groups)

    for name in names_test:
        dft = groups_test.get_group(name)
        dft = dft.sort_values(by='created_at', ascending=True)
        expected_action = int(name[-1])
        possible_actions = []

        for i in range(0,len(dft)):
            # read each row as dict and later covert to Ordered Dict of tuples e.g. [(feature1, value), (feature2, value), ...]
            obs_t = dft.iloc[[i]].to_dict(orient='records')[0]
            obs_t = {key: np.array([obs_t[key]]) for key in obs_t if key not in ['owner_name','repo_name','created_at','pull_no','emotion_vr']}
            order_of_keys = ['compound', 'has_code_element', 'merged_or_not', 'neg_vr', 'neu_vr', 'pos_vr', 'stopw_ratio', 'word_count']
            list_of_tuples = [(key, obs_t[key]) for key in order_of_keys]
            obs_t = collections.OrderedDict(list_of_tuples)

            evaluated_action = model.predict(obs_t)[0]
            possible_actions.append(int(evaluated_action))
        effective_action = statistics.mode(possible_actions)
        del dft, possible_actions

        predicted_class.append(int(effective_action))
        true_class.append(int(expected_action))
        
    

    #--------------------------------------------------------------------------------------------------------------------------------------------
    print("-----------------------------------------confusion_matrix-----------------------------------------")
    print(classification_report(true_class,predicted_class))
    gmean_score = geometric_mean_score(true_class,predicted_class)
    ros_score = roc_auc_score(true_class, predicted_class, average=None)
    print(f"Geometric Mean Score: {gmean_score} while ROC_AUC_Score: {ros_score}")
    env.close()
    del test, train
    del model