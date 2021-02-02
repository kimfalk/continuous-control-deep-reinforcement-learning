# Udacity DRL project 2: Continuous Control project

In the following, the learning algorithm is described, along with the chosen hyperparameters. It also describes the model architectures for the neural networks used.

_Environment was solved in 182 episodes with an average Score of 30.12._


## Simulation Environment

The simulation is done in an artificial environment where a double-jointed arm can move to target locations. A reward of +0.1 is given for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to a position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1.


## The Learning Algorithm 

The agent is implemented using the Deep Deterministic Policy Gradient (DDPG) algorithm [1]. The DDPG builds upon the Deterministic Policy Gradient (DPG)[2]. The DPG was special because it was the first policy gradient algorithm not to be stochastic. It handles exploration by adding noise to the action values, which will make the agent explore different values in the action space. The DDPG builds on top of the DPG and adding learnings from the DQN algorithm [3]. 

The DDPG comprises an actor and a critic part. That actor will approximate the policy, while the critic approximates an advantage function. Our goal is to create a policy which will recommend the right actions in each state, so why the advantage function then, you might ask. The advantage function helps the agent to understand how valuable the decision was. If your agent is in a state where all actions are reasonable, it should put too much value into which action it had chosen there. So the critic is there to make the policy understand when it did good or bad. 

One of the learnings used from the DQN is that of having two neural networks for each model, one which is the target network, and one is the online network. This trick enables the agent to learn more steadily and not fluctuate. 

## The Neural Networks Architecture

The Actor network(s) comprises two hidden layers, each with 256 nodes using the RELU activation function. Both layers are preceded with batch normalisation.  The actor network output layer is as big as the action space and uses the tanh activation function. 

The Critic network(s) are a bit different, first layer is a linear layer with 256 nodes using the RELU activation function, the layer is followed by batch normalisation. The following layer combines the output of the first layer with the action. This layer also contains 256 nodes and uses RELU activation. Lastly, the output layer contains only one node and doesn't have an activation method. 

## Hyperparameters

``` python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

## Further work

There are many ways you could improve on this. I believe that the solution that I have is pretty good. But I would have liked to test out other algorithms and see if I could have done it even better, the first would be PPO as I like its simplicity. Staying with the DDPG, I would start out focusing on how the noise is applied to the actions. It could be interesting to see if it would be worth adding decay to the noise, or something like momentum which could be tied to the reward.  

[1 - Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
[2 - Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)
[3 - Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)