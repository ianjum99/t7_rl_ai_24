from models.ppo_agent import PPOAgent
from rl.tekken_environment import TekkenEnvironment
from rl.replay_memory import ReplayMemory

# Initialize environment, PPO agent, and replay memory
env = TekkenEnvironment()
ppo_agent = PPOAgent(state_dim=3, action_dim=5)
memory = ReplayMemory(capacity=10000)

# Training loop
for episode in range(100):
    state = env.reset()
    for t in range(1000):
        action = ppo_agent.act(state)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, reward, next_state, done)
        
        if done:
            break
        state = next_state

    # Train PPO agent with collected experiences
    ppo_agent.train(memory)

# Save the PPO model
torch.save(ppo_agent.model.state_dict(), "models/saved/ppo_agent.pth")
print("PPO agent model saved successfully!")

