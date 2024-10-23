#ifndef PPO_AGENT_H
#define PPO_AGENT_H

#include <torch/script.h>

// Function to return action based on the policy model and current state
int ppo_action(torch::jit::script::Module& policy, torch::Tensor& state);

// Function to train PPO agent using the policy model, states, actions, and rewards
void train_ppo(torch::jit::script::Module& policy, torch::Tensor& states, torch::Tensor& actions, torch::Tensor& rewards);

#endif  // PPO_AGENT_H

