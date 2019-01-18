# Project 3 (Deep Reinforcement Learning Nanodegree) Multi-Agent RL.

## Overview 

The goal of this project is to train an agent to solve Unity’s Tennis environment. This task contains competing agents and thus falls into the realm of MARL (multi-agent reinforcement learning). 

## Algorithm 

The primary algorithm used to solve this task was a multi-agent version of DDPG or MADDPG. MADDPG was first used by researchers at Open AI in 2017. 

Multi-agent DDPG’s can be used for cooperative and competitive multi-agent RL settings:
    *Corporative: This is where agents work together to accomplish a shared goal. 
    *Competitive: This is where agents work against each other, each to maximize their own goal. 

MADDPG modifies DDPG to make it useful for MARL in the following ways.
    Actor
        Only use their own observations at execution time. 
    Critic 
        Is augmented with information about policies of different agents during training. That is the critic estimates a centralized action-value function, it takes actions of all agents as input along with state information and abouts the q-function for agent i. 

Below is a diagram of the actor-critic MADDPG where Qi through Qn are the critics (q-functions) 
