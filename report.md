# Project 3: Deep Reinforcement Learning Nanodegree

## Multi-Agent RL

## Overview 

The goal of this project is to train an agent to solve Unity’s Tennis environment. This task contains competing agents and thus falls into the realm of MARL (multi-agent reinforcement learning). 

## Algorithm 

The primary algorithm used to solve this task was a multi-agent version of DDPG or MADDPG. MADDPG was first used by researchers at Open AI in 2017. 

Multi-agent DDPG’s can be used for cooperative and competitive multi-agent RL settings:
* Corporative: This is where agents work together to accomplish a shared goal. 
* Competitive: This is where agents work against each other, each to maximize their own goal. 

MADDPG modifies DDPG to make it useful for MARL in the following ways.
* Actor
    * Only use their own observations at execution time. 
* Critic 
    * Is augmented with information about policies of different agents during training. That is the critic estimates a centralized action-value function, it takes actions of all agents as input along with state information and abouts the q-function for agent i. 

Below is a diagram of the actor-critic MADDPG where Qi through Qn are the critics (q-functions) 

![alt text](https://github.com/cloud36/marl-tennis-/blob/master/img/maddpg.png)

### Network Architecture 

The solving Actor and Critic Networks had the following architecture. 

**Actor**
   * Hidden Layers: 2
   * Layer 1: 400 Units, Linear, Relu Activation
   * Layer 2: 300 Units, Linear, Reul Activation
   * Output: Tanh Activation 
   
**Critic**
   * Hidden Layers: 2
   * Layer 1: 400 Units, Linear, Relu Activation
   * Layer 2: 200 Units, Linear, Relu Activation
   * Output Layer: 1 Unit, Linear Activation 
   
## Hyperparameters
DDPG:
* BATCH_SIZE: 200
* BUFFER_SIZE: 100000
* GAMMA: 0.99
* LR_ACTOR: 0.0001
* LR_CRITIC: 0.001
* TAU: 0.001
* UPDATE_EVERY: 2
* WEIGHT_DECAY: 0.0001


## Rewards Plot

Below, we can see the average reward of 100 episodes achieved by MADDPG. 

![alt text](https://github.com/cloud36/marl-tennis-/blob/master/img/average_reward_maddpg.png)


### Improvements
* Google Cloud Hyperparameter Search
   * Considering that DDPG/MADDPG and Deep RL is very sensitive hyperparameters, it would be interesting to speed up and/or distribute this process. Google Cloud Platform recently [detailed how to do this on their Cloud ML Engine.](https://cloud.google.com/blog/products/ai-machine-learning/deep-reinforcement-learning-on-gcp-using-hyperparameters-and-cloud-ml-engine-to-best-openai-gym-games)
* It would be interesting to see it MCTS could be used to optimize game play. 
