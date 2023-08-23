# Reinforcement Learning for GitHub Pull Request Predictions: Analyzing Development Dynamics

Part of Master of Applied Science: Electrical and Computer Engineering degree at Carleton University, Ottawa, Canada. 

In the rapidly changing software development field, the pull-based model, supported by tools like GitHub, plays a pivotal role in collaborations. Understanding factors influencing this model is crucial for process enhancement. This thesis employs two Reinforcement Learning (RL) formalizations to predict Pull Request (PR) outcomes. The first utilizes 72 PR characteristics (e.g., PR Size, Test Inclusion, Developers’ PR Experience, Programming Language), achieving a G-mean of 0.82664. The second focuses solely on PR discussions, attaining a higher G-mean of 0.88372. Both RL models outperform established techniques like Random Forest, XGBoost, and Naive Bayes. Additionally, the study explores PR factors and merge time through a survey of 22 developers, identifying key influencers such as PR size and reviewer experience, while also revealing common PR review approaches. Concluding, the study outlines achievements, future directions, and establishes an RL-based PR outcome prediction framework, along with publishing specific datasets.

## Dataset for RL Formalization 1 (Holistic approach to PR characteristics):
Source: X. Zhang, Y. Yu, G. Georgios, and A. Rastogi, “Pull request decisions explained: An empirical overview,” IEEE Transactions on Software Engineering, pp. 1–1, 2022.

Link to data: https://zenodo.org/record/4837135#.YLEWyY3isdW

## Dataset for RL Formalization 2 (PR Dicussions along with Sentiment Analysis):
Sentiment Analysis tool: https://github.com/cjhutto/vaderSentiment

Source (Sentiment Analysis): C. Hutto and E. Gilbert, “VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text”, ICWSM, vol. 8, no. 1, pp. 216-225, May 2014.

Link to data: https://zenodo.org/record/8271704

## RL Formalization 1
RL Models: DQN, A2C, PPO

Custom RL Environemnt: PR ENV

Baselines (with or without ROS and RUS): Naive Bayes, XGBoost Classifier, Random Forest Classifier

ROS: Random Over-Sampling

RUS: Random Under-Sampling

## RL Formalization 2
RL Models: DQN

Custom RL Environemnt: PR ENV

Baselines: Same as RL Formalization 1

