import pyautogui
import cv2
import numpy as np

# Function to capture game state (e.g., health bar and positions)
def capture_game_state():
    screenshot = pyautogui.screenshot()
    image = np.array(screenshot)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract health bar, distance, etc. (dummy values for now)
    player_health = np.random.rand()  # Placeholder for actual health extraction logic
    enemy_health = np.random.rand()
    distance = np.random.rand()

    return np.array([player_health, enemy_health, distance])

# Example usage
if __name__ == '__main__':
    state = capture_game_state()
    print(f"Captured state: {state}")

