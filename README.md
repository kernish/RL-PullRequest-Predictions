# Comparative Study of Reinforcement Learning in GitHub Pull Request Outcome Predictions

Part of #### degree at ##### University, ####. 

In the rapidly evolving field of software development, pull-based development models, facilitated by tools such as GitHub, are essential for collaboration. This study explores factors that influence pull request (PR) outcomes and employs two Reinforcement Learning (RL) formalizations, modeled as Markov Decision Processes, for PR outcome prediction. The first model leverages 72 PR characteristics and achieves a G-mean score of 0.82664, while the second focuses solely on PR discussions, resulting in a G-mean of 0.88372. Using a specially designed reward function, these RL formalizations strategically address data imbalance and excel in mimicking both single-stage and multi-stage PR review processes. They outperform baseline models (Random Forest, XGBoost, and a Naive Bayes baseline) across various data splits—namely 80/20, 50/50, and 20/80—and are particularly effective at predicting PR rejections. The study also provides specific datasets for future research.

## Dataset for BatchRL and ChatRL:

Link to data: https://zenodo.org/record/8271704

## RL Formalization 1 (BatchRL)
RL Models: DQN, A2C, PPO

Custom RL Environemnt: PR ENV

Baselines (with or without ROS and RUS): Naive Bayes, XGBoost Classifier, Random Forest Classifier

ROS: Random Over-Sampling

RUS: Random Under-Sampling

## RL Formalization 2 (ChatRL)
RL Models: DQN

Custom RL Environemnt: PR ENV

Baselines: Same as RL Formalization 1

