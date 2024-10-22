from models.health_bar_cnn import HealthBarCNN
from models.ppo_agent import PPOAgent
import torch

def load_health_bar_model():
    model = HealthBarCNN()
    model.load_state_dict(torch.load("models/saved/health_bar_cnn.pth"))
    model.eval()  # Set to evaluation mode
    return model

def load_ppo_agent():
    agent = PPOAgent(state_dim=3, action_dim=5)
    agent.model.load_state_dict(torch.load("models/saved/ppo_agent.pth"))
    agent.model.eval()  # Set to evaluation mode
    return agent

def main():
    print("Launching Tekken 7 RL AI with pretrained models...")
    health_bar_model = load_health_bar_model()
    ppo_agent = load_ppo_agent()

    # Use the models for inference or game interaction
    pass

if __name__ == '__main__':
    main()

