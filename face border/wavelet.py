import cv2
import pywt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# Extract frames from video
def extract_frames(video_path, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        resized_frame = cv2.resize(gray_frame, (128, 128))  # Resize for consistency
        frames.append(resized_frame)
        count += 1
    cap.release()
    return frames

# Apply wavelet transform
def apply_wavelet_transform(image):
    coeffs = pywt.wavedec2(image, 'haar', level=2)  # Haar wavelet with 2 levels
    features = []
    for level in coeffs:
        for sub_band in level:
            features.append(np.mean(sub_band))
            features.append(np.std(sub_band))
    return np.array(features)

# Load dataset and extract features from videos
def load_dataset_from_videos(video_paths, labels, max_frames=50):
    features = []
    new_labels = []
    for i, video_path in enumerate(video_paths):
        if not os.path.exists(video_path):
            print(f"Error: {video_path} does not exist.")
            continue
        
        frames = extract_frames(video_path, max_frames)
        if len(frames) == 0:  # Skip if no frames extracted
            print(f"Warning: No frames extracted from {video_path}")
            continue
        
        for frame in frames:
            wavelet_features = apply_wavelet_transform(frame)
            features.append(wavelet_features)
            new_labels.append(labels[i])  # Repeat label for each frame
    
    if len(features) == 0:
        raise ValueError("No data available: check your video paths and frame extraction.")
    
    return np.array(features), np.array(new_labels)

# Define CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Train and evaluate CNN
def train_cnn(features, labels):
    # Stratified splitting ensures balanced class representation in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Create DataLoader
    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    # Initialize model, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = SimpleCNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            y_pred.extend((preds > 0.5).float().squeeze().tolist())
            y_true.extend(yb.squeeze().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    return model

# Generate paths and labels from FF++ dataset
def prepare_ffpp_dataset(base_path):
    video_paths = []
    labels = []
    for label, folder in enumerate(['real', 'fake']):  # 'real' = 0, 'fake' = 1
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist.")
            continue
        for video_file in os.listdir(folder_path):
            if video_file.endswith('.mp4'):  # Adjust for your video format
                video_paths.append(os.path.join(folder_path, video_file))
                labels.append(label)
    return video_paths, labels

# Example usage for FaceForensics++ dataset
base_path = "C:\\Users\\Pavan\\Downloads\\FF++"  # Replace with actual FF++ dataset path
video_paths, labels = prepare_ffpp_dataset(base_path)

try:
    features, labels = load_dataset_from_videos(video_paths, labels)
    if len(features) > 0:
        cnn_model = train_cnn(features, labels)
except ValueError as e:
    print(e)