# import all the packages
import os
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

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict
from stable_baselines3 import DQN

# Custom Environment PR ENV
class prEnv(gym.Env):
    
    def __init__(self, DataSet):
        '''
        Initialize the object, set the variables
        '''
        # Actions we can take: accept: 1, reject: 0 (default state pre-action) request change: 2
        self.action_space = Discrete(2)
        
        # the dataset to fetch each entry of pull requests
        self.data = DataSet

        # a variable to iterate through the records in the dataset
        self.idx = 0

        # Selecting the Influential features for pull requests
        self.observation_space = Dict({ 
                                'account_creation_days':Box(low=np.array([0]), high=np.array([4200]), dtype='int'),
                                'asserts_per_kloc':Box(low=np.array([0]),high=np.array([571]), dtype='float'),
                                'at_tag':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'churn_addition':Box(low=np.array([0]), high=np.array([3000000]), dtype='int'),
                                'churn_deletion':Box(low=np.array([0]), high=np.array([2000000]), dtype='int'),
                                'ci_build_num':Box(low=np.array([0]), high=np.array([1000]), dtype='int'),
                                'ci_exists':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'ci_failed_perc':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'comment_conflict':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'contrib_agree':Box(low=np.array([0]),high=np.array([1]), dtype='float'),
                                'contrib_comment':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'contrib_cons':Box(low=np.array([0]),high=np.array([1]), dtype='float'),
                                'contrib_extra':Box(low=np.array([0]),high=np.array([1]), dtype='float'),
                                'contrib_neur':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'contrib_open':Box(low=np.array([0]),high=np.array([1]), dtype='float'),
                                'contrib_perc_commit':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'core_comment':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'core_member':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'description_length':Box(low=np.array([0]), high=np.array([32000]), dtype='int'),
                                'files_changed':Box(low=np.array([0]), high=np.array([18000]), dtype='int'),
                                'first_pr':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'followers':Box(low=np.array([0]), high=np.array([54000]), dtype='int'),
                                'fork_num':Box(low=np.array([0]), high=np.array([35000]), dtype='int'),
                                'friday_effect':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'has_comments':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'has_exchange':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'inte_agree':Box(low=np.array([0]),high=np.array([1]), dtype='float'),
                                'inte_comment':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'inte_cons':Box(low=np.array([0]),high=np.array([1]), dtype='float'),
                                'inte_extra':Box(low=np.array([0]),high=np.array([1]), dtype='float'),
                                'inte_neur':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'inte_open':Box(low=np.array([0]),high=np.array([1]), dtype='float'),
                                'integrator_availability':Box(low=np.array([0]), high=np.array([25]), dtype='int'),
                                'language':Box(low=np.array([0]), high=np.array([5]), dtype='int'),
                                'lifetime_minutes':Box(low=np.array([0]), high=np.array([3400000]), dtype='int'),
                                'merged_or_not':Box(low=np.array([0]), high=np.array([2]), dtype='int'),
                                'num_code_comments':Box(low=np.array([0]), high=np.array([600]), dtype='int'),
                                'num_code_comments_con':Box(low=np.array([0]), high=np.array([240]), dtype='int'),
                                'num_comments':Box(low=np.array([0]), high=np.array([600]), dtype='int'),
                                'num_commits':Box(low=np.array([0]), high=np.array([1500]), dtype='int'),
                                'num_participants':Box(low=np.array([0]), high=np.array([85]), dtype='int'),
                                'open_issue_num':Box(low=np.array([0]), high=np.array([7000]), dtype='int'),
                                'open_pr_num':Box(low=np.array([0]), high=np.array([11000]), dtype='int'),
                                'perc_contrib_neg_emo':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'perc_contrib_pos_emo':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'perc_external_contribs':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'perc_inte_neg_emo':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'perc_inte_pos_emo':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'perc_neg_emotion':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'perc_pos_emotion':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'pr_succ_rate':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'prev_pullreqs':Box(low=np.array([0]), high=np.array([3800]), dtype='int'),
                                'prior_interaction':Box(low=np.array([0]), high=np.array([20200]), dtype='int'),
                                'prior_review_num':Box(low=np.array([0]), high=np.array([15000]), dtype='int'),
                                'project_age':Box(low=np.array([0]), high=np.array([140]), dtype='int'),
                                'reopen_or_not':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'requester_succ_rate':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'same_user':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'sloc':Box(low=np.array([0]), high=np.array([7600000]), dtype='int'),
                                'social_strength':Box(low=np.array([0]), high=np.array([1]), dtype='float'),
                                'src_churn':Box(low=np.array([0]), high=np.array([4200000]), dtype='int'),
                                'stars':Box(low=np.array([0]),high=np.array([240000]), dtype='int'),
                                'team_size':Box(low=np.array([0]), high=np.array([280]), dtype='int'),
                                'test_cases_per_kloc':Box(low=np.array([0]),high=np.array([251]), dtype='float'),
                                'test_churn':Box(low=np.array([0]), high=np.array([390000]), dtype='int'),
                                'test_inclusion':Box(low=np.array([0]), high=np.array([1]), dtype='int'),
                                'test_lines_per_kloc':Box(low=np.array([0]),high=np.array([1001]), dtype='float')
                                })
         
    def step(self, action):
        '''
        Calculate rewards and next state based on the action. Reward offered immediately!!
        Actions we can take: accept: 1, reject: 0

        Trade-off factor k = 2*IR, where IR = Imbalance Ratio = 0.12
        '''
        # Apply action     
        if (action==1):
            self.state['merged_or_not'] = np.array([1])
        elif (action==0):
            self.state['merged_or_not'] = np.array([0])
        
        # Calculate reward 
        k = 2*(0.12)
        if action == self.expected_action:
            # imbalanced data hence more reward for minority class
            if action == 1:
                self.reward = k
                self.done = False
                self.state = self.next_obs()
            elif action == 0:
                self.reward = 1
                self.done = False
                self.state = self.next_obs()
        else: 
            if self.expected_action == 0:
                self.reward = -1
                self.done = False
                self.state = self.next_obs()
                self.done = True
            elif self.expected_action == 1:
                self.reward = -1*k
                self.done = False
                self.state = self.next_obs()
        
        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, self.reward, self.done, info

    
    def next_obs(self):
        '''
        A function to pass row by row, all the entries in the dataset as the state.
        And set the value of expected_action to help calculate the immedite reward!
        '''
        if self.idx < len(self.group):
            obs = self.group.iloc[[self.idx]].to_dict(orient='records')[0]
            obs = {key: np.array([obs[key]]) for key in obs if key not in ['id','project_id','creator_id','last_closer_id','last_close_time']}
            order_of_keys = ['account_creation_days', 'asserts_per_kloc', 'at_tag', 'churn_addition', 'churn_deletion', 'ci_build_num', 'ci_exists', 'ci_failed_perc', 'comment_conflict', 'contrib_agree', 'contrib_comment', 'contrib_cons', 'contrib_extra', 'contrib_neur', 'contrib_open', 'contrib_perc_commit', 'core_comment', 'core_member', 'description_length', 'files_changed', 'first_pr', 'followers', 'fork_num', 'friday_effect', 'has_comments', 'has_exchange', 'inte_agree', 'inte_comment', 'inte_cons', 'inte_extra', 'inte_neur', 'inte_open', 'integrator_availability', 'language', 'lifetime_minutes', 'merged_or_not', 'num_code_comments', 'num_code_comments_con', 'num_comments', 'num_commits', 'num_participants', 'open_issue_num', 'open_pr_num', 'perc_contrib_neg_emo', 'perc_contrib_pos_emo', 'perc_external_contribs', 'perc_inte_neg_emo', 'perc_inte_pos_emo', 'perc_neg_emotion', 'perc_pos_emotion', 'pr_succ_rate', 'prev_pullreqs', 'prior_interaction', 'prior_review_num', 'project_age', 'reopen_or_not', 'requester_succ_rate', 'same_user', 'sloc', 'social_strength', 'src_churn', 'stars', 'team_size', 'test_cases_per_kloc', 'test_churn', 'test_inclusion', 'test_lines_per_kloc']
            list_of_tuples = [(key, obs[key]) for key in order_of_keys]
            obs = collections.OrderedDict(list_of_tuples)
            
            self.expected_action = int(obs['merged_or_not'])
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
        self.group = self.data.sample(frac=1)
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

    # load training and testing datasets
    train = pd.read_csv(str("path"+str("train_data.csv")), encoding= "utf-8", on_bad_lines= bad_line, engine="python") 
    test = pd.read_csv(str("path"+str("test_data.csv")), encoding= "utf-8", on_bad_lines= bad_line, engine="python") 
    
    train['last_close_time'] = pd.to_datetime(train['last_close_time'], infer_datetime_format=True)
    test['last_close_time'] = pd.to_datetime(test['last_close_time'], infer_datetime_format=True)

    pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)

    #--------------------------------------------------------------------------------------------------------------------------------------------
    # set the path for logs
    log_path = os.path.join('training','logs')

    # instantiate custom environment with training data
    env = prEnv(train)
    
    # tweak the total timesteps using time_steps_learn
    time_steps_learn = 2300000
    
    model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log=log_path)

    # train the DQN model
    model.learn(total_timesteps=time_steps_learn)
    
    #--------------------------------------------------------------------------------------------------------------------------------------------
    # set the path to save the trained model
    out_path = os.path.join('training', 'SavedModels', 'DQN_Model_2p')
    model.save(out_path)

    del model
    del train
    
    #--------------------------------------------------------------------------------------------------------------------------------------------
    
    print("---------------------------Evaluation---------------------------")
    
    # load the model 
    model = DQN.load(out_path, env)

    true_class = []
    predicted_class = []
    
    for i in range(0,len(test)):
        # read each row as dict and later covert to Ordered Dict of tuples e.g. [(feature1, value), (feature2, value), ...]
        obs_t = test.iloc[[i]].to_dict(orient='records')[0]
        obs_t = {key: np.array([obs_t[key]]) for key in obs_t if key not in ['id','project_id','creator_id','last_closer_id','last_close_time']}
        order_of_keys = ['account_creation_days', 'asserts_per_kloc', 'at_tag', 'churn_addition', 'churn_deletion', 'ci_build_num', 'ci_exists', 'ci_failed_perc', 'comment_conflict', 'contrib_agree', 'contrib_comment', 'contrib_cons', 'contrib_extra', 'contrib_neur', 'contrib_open', 'contrib_perc_commit', 'core_comment', 'core_member', 'description_length', 'files_changed', 'first_pr', 'followers', 'fork_num', 'friday_effect', 'has_comments', 'has_exchange', 'inte_agree', 'inte_comment', 'inte_cons', 'inte_extra', 'inte_neur', 'inte_open', 'integrator_availability', 'language', 'lifetime_minutes', 'merged_or_not', 'num_code_comments', 'num_code_comments_con', 'num_comments', 'num_commits', 'num_participants', 'open_issue_num', 'open_pr_num', 'perc_contrib_neg_emo', 'perc_contrib_pos_emo', 'perc_external_contribs', 'perc_inte_neg_emo', 'perc_inte_pos_emo', 'perc_neg_emotion', 'perc_pos_emotion', 'pr_succ_rate', 'prev_pullreqs', 'prior_interaction', 'prior_review_num', 'project_age', 'reopen_or_not', 'requester_succ_rate', 'same_user', 'sloc', 'social_strength', 'src_churn', 'stars', 'team_size', 'test_cases_per_kloc', 'test_churn', 'test_inclusion', 'test_lines_per_kloc']
        list_of_tuples = [(key, obs_t[key]) for key in order_of_keys]
        obs_t = collections.OrderedDict(list_of_tuples)
        
        expected_action = int(obs_t['merged_or_not'])

        evaluated_action = model.predict(obs_t)[0]
        
        predicted_class.append(int(evaluated_action))
        true_class.append(int(expected_action))

    # calculate evaluation metrics
    print(classification_report(true_class,predicted_class))
    gmean_score = geometric_mean_score(true_class,predicted_class)
    ros_score = roc_auc_score(true_class, predicted_class, average=None)
    print(f"Geometric Mean Score: {gmean_score} while ROC_AUC_Score: {ros_score}")
    
    env.close()
    del test
    del model