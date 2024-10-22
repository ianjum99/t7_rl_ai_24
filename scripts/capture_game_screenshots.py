import pyautogui
import time

# Function to capture game screenshots
def capture_game_screenshots(num_screenshots=100, save_dir="data/health_bar_dataset/train/"):
    for i in range(num_screenshots):
        screenshot = pyautogui.screenshot()
        screenshot.save(f"{save_dir}screenshot_{i}.png")
        time.sleep(1)  # Capture a screenshot every second
        print(f"Captured screenshot {i+1}/{num_screenshots}")

# Example usage
if __name__ == '__main__':
    capture_game_screenshots()

