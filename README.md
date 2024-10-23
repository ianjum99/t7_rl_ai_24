# Tekken 7 Reinforcement Learning AI

This project implements a reinforcement learning (RL) agent to play **Tekken 7** by interacting with the game environment, extracting game states (such as health bars and distances between players), and optimizing actions (punch, kick, block, etc.) using a **PPO (Proximal Policy Optimization)** algorithm. The project is built using both **Python** and **C++** for performance optimizations, with deep learning models for health bar detection and RL agent policies.

## Project Structure

```plaintext
tekken7_rl_ai/
├── data/                               # Dataset storage (screenshots, labels, etc.)
│   ├── health_bar_dataset/             # Dataset for health bar CNN
│   │   ├── train/                      # Training images (screenshots)
│   │   └── labels.csv                  # CSV with image names and health percentages
├── models/                             # Model definitions and pretrained models
│   ├── health_bar_cnn.py               # CNN model for health bar detection (Python)
│   ├── ppo_agent.py                    # PPO agent (Python)
│   ├── saved/                          # Pretrained models (weights)
│       ├── health_bar_cnn.pth          # Pretrained CNN weights
│       └── ppo_agent.pth               # Pretrained PPO weights
│   ├── cpp/                            # C++ Model implementations
│       ├── health_bar_cnn.cpp          # C++ inference for health bar CNN
│       ├── ppo_agent.cpp               # C++ PPO agent implementation
│       ├── replay_buffer.cpp           # C++ replay memory
│       └── CMakeLists.txt              # CMake file to build the C++ modules
├── bindings/                           # Python bindings for C++ functions using Pybind11
│   ├── health_bar_bindings.cpp         # Bindings for health bar detection
│   ├── ppo_bindings.cpp                # Bindings for PPO agent
│   └── replay_buffer_bindings.cpp      # Bindings for replay memory
├── scripts/                            # Scripts for dataset generation, training, and inference
│   ├── capture_game_screenshots.py     # Capture game screenshots for health bar dataset
│   ├── label_health_bars.py            # Automatically/Manually label health bars
│   ├── train_health_bar_cnn.py         # Train CNN on health bar dataset (Python)
│   ├── train_ppo_agent.py              # Train PPO agent with Tekken environment (Python)
│   ├── perform_actions.py              # Simulate actions in Tekken 7
│   ├── capture_game_state.py           # Capture game state for RL (health, distance)
├── rl/                                 # Custom Tekken 7 environment for PPO agent
│   ├── tekken_environment.py           # OpenAI Gym-style Tekken environment
│   └── replay_memory.py                # Replay memory (Python)
├── cpp_build/                          # C++ build output directory
│   └── lib/                            # Shared libraries (e.g., .so or .dll files)
├── include/                            # Header files for C++ modules
│   ├── health_bar_cnn.h                # Header for CNN
│   ├── ppo_agent.h                     # Header for PPO agent
│   ├── replay_buffer.h                 # Header for replay memory
├── main.py                             # Main entry point to run the project
├── requirements.txt                    # Python dependencies
├── CMakeLists.txt                      # Root CMake file to build all C++ components
└── README.md                           # Project documentation

Installation
Prerequisites

    Python 3.7+
    PyTorch (for deep learning)
    OpenCV (for image processing)
    PyAutoGUI (for interacting with the game)
    Tesseract OCR (for health bar detection via image text recognition)
    LibTorch (for C++ deep learning inference)
    Pybind11 (for Python-C++ integration)

Install Dependencies

    Clone the repository:

    bash

git clone https://github.com/yourusername/tekken7_rl_ai.git
cd tekken7_rl_ai

Install Python dependencies:

bash

pip install -r requirements.txt

Install Tesseract OCR:

    On Ubuntu:

    bash

        sudo apt install tesseract-ocr

        On Windows, download and install from here.

    Install LibTorch:
        Follow the installation instructions from the official LibTorch page.

Build C++ Modules

To integrate C++ code with Python using Pybind11 and LibTorch, use the following steps:

    Install CMake and Pybind11:

    bash

pip install pybind11 cmake

Compile the C++ modules:

bash

    mkdir -p cpp_build
    cd cpp_build
    cmake ..
    make

    The shared libraries (.so or .dll files) will be created in the cpp_build/lib/ directory.

Usage
1. Training the Health Bar CNN

To train the health bar detection CNN:

    Capture screenshots of Tekken 7 using:

    bash

python scripts/capture_game_screenshots.py

Label the health bars using:

bash

python scripts/label_health_bars.py

Train the CNN on the labeled dataset:

bash

    python scripts/train_health_bar_cnn.py

The trained model will be saved as models/saved/health_bar_cnn.pth.
2. Training the PPO Agent

Once the health bar CNN is trained, you can train the PPO agent to learn how to play Tekken 7.

    Start training the PPO agent:

    bash

    python scripts/train_ppo_agent.py

The PPO model will be saved as models/saved/ppo_agent.pth.
3. Running the Pretrained AI

If you have pretrained models, you can run the Tekken 7 AI using:

bash

python main.py

This script will:

    Load the pretrained health bar CNN and PPO agent.
    Interact with Tekken 7 to predict actions and play the game.

Components
1. Health Bar CNN

The health bar CNN is trained to detect the health percentage of the player and the opponent from game screenshots. This is crucial for allowing the RL agent to know the current game state (health, distance, etc.).
2. PPO Agent

The PPO agent uses reinforcement learning to learn optimal actions (punch, kick, block, etc.) based on the game state. It uses the health bar CNN to get the health information, and then decides actions using policy optimization.
3. C++ Optimizations

Several parts of the project are optimized using C++:

    Health bar detection: Inference for health bar detection is implemented in C++ using LibTorch for better performance.
    PPO agent: The policy and value updates are implemented in C++ for faster reinforcement learning training.
    Replay memory: The experience replay buffer is implemented in C++ for efficient memory handling.

4. Tekken Environment

The custom TekkenEnvironment is built using OpenAI Gym-style interfaces. It interacts with Tekken 7, captures game states (health, distance), and applies the selected actions in real-time.
Future Improvements

    More Actions: Add more complex actions such as special moves, combos, etc.
    Improved Health Bar Detection: Use a more advanced deep learning model (e.g., a ResNet) for more accurate health bar detection.
    Advanced RL Algorithms: Implement other reinforcement learning algorithms such as A3C (Asynchronous Advantage Actor-Critic) or SAC (Soft Actor-Critic) to improve learning efficiency.

License

This project is licensed under the MIT License - see the LICENSE file for details.
Contributing

Contributions are welcome! Feel free to fork this repository and submit pull requests. If you encounter any issues or have ideas for new features, please open an issue.
Acknowledgments

This project was inspired by the integration of reinforcement learning into competitive video games. Special thanks to the PyTorch, OpenCV, and OpenAI Gym communities for providing the tools and libraries that made this possible.
