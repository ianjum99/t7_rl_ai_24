import torch
from torch.utils.data import DataLoader
from models.health_bar_cnn import HealthBarCNN
from scripts.capture_game_screenshots import HealthBarDataset
from torchvision import transforms

# Define transformation for the dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load the dataset
dataset = HealthBarDataset(csv_file='data/health_bar_dataset/labels.csv', root_dir='data/health_bar_dataset/train', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
model = HealthBarCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(10):  # Train for 10 epochs
    running_loss = 0.0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")

# Save the trained model
torch.save(model.state_dict(), "models/saved/health_bar_cnn.pth")
print("Model saved successfully!")

