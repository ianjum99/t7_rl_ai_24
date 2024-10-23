from models.health_bar_cnn import HealthBarCNN
from models.ppo_agent import PPOAgent
import torch
from scripts.capture_game_state import capture_game_state
from scripts.perform_actions import perform_action
from rl.tekken_environment import TekkenEnvironment

def main():
    print("Launching Tekken 7 RL AI with pretrained models...")
    health_bar_model = HealthBarCNN()
    health_bar_model.load_state_dict(torch.load('models/saved/health_bar_cnn.pth'))
    health_bar_model.eval()

    ppo_agent = PPOAgent(state_dim=3, action_dim=5)
    ppo_agent.model.load_state_dict(torch.load('models/saved/ppo_agent.pth'))
    ppo_agent.model.eval()

    env = TekkenEnvironment()
    state = env.reset()
    done = False
    while not done:
        action, _ = ppo_agent.select_action(state)
        state, reward, done, _ = env.step(action)
        print(f"Action taken: {action}, Reward: {reward}")

if __name__ == '__main__':
    main()

