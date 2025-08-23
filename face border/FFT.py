import os
import cv2
import numpy as np
from retinaface import RetinaFace
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
import pywt
import sys
sys.path.append(r"PATH_TO_FACE_PARSING")  # e.g., r"D:\program\vscode\deep fake detection\face-parsing.PyTorch"

# --- BiSeNet Setup ---
from face_parsing.model import BiSeNet
def load_bisenet(model_path='face_parsing.pth', device='cpu'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    net.to(device)
    return net

def parse_face_bisenet(net, image, device='cpu'):
    to_tensor = T.Compose([
        T.ToPILImage(),
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    return cv2.resize(parsing.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

# --- Face Alignment ---
def align_face(frame, landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    center = tuple(np.mean([left_eye, right_eye], axis=0).astype(int))
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(frame, rot_matrix, (frame.shape[1], frame.shape[0]))
    return aligned_face

# --- Wavelet Feature Extraction ---
def extract_wavelet_features(region_img):
    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    coeffs = pywt.wavedec2(gray, 'haar', level=2)
    features = []
    for arr in coeffs:
        if isinstance(arr, tuple):
            for sub in arr:
                features.append(np.mean(sub))
                features.append(np.std(sub))
        else:
            features.append(np.mean(arr))
            features.append(np.std(arr))
    return np.array(features, dtype=np.float32)

# --- ResNet-18 Model ---
from torchvision.models import resnet18
class ResNet18Classifier(nn.Module):
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_features, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    def forward(self, x):
        return self.resnet(x)

# --- Main Pipeline ---
def process_video(video_path, bisenet, device, max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count-1, max_frames, dtype=int)
    all_features = []
    for idx in tqdm(indices, desc=f"Processing {os.path.basename(video_path)}", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        detections = RetinaFace.detect_faces(frame)
        if not detections:
            continue
        for key, face in detections.items():
            landmarks = face['landmarks']
            aligned = align_face(frame, landmarks)
            parsing = parse_face_bisenet(bisenet, aligned, device)
            # Semantic regions: 1=skin, 6=lips, 7=upper lip, 11=left eye, 12=right eye, etc.
            region_ids = [1, 6, 7, 11, 12]
            region_feats = []
            for rid in region_ids:
                mask = (parsing == rid).astype(np.uint8) * 255
                region = cv2.bitwise_and(aligned, aligned, mask=mask)
                feats = extract_wavelet_features(region)
                region_feats.append(feats)
            all_features.append(np.concatenate(region_feats))
            break  # Only first face per frame
    cap.release()
    if all_features:
        stacked = np.vstack(all_features)
        mean_feat = stacked.mean(axis=0)
        std_feat = stacked.std(axis=0)
        return np.concatenate([mean_feat, std_feat])
    else:
        return np.zeros(100)  # fallback

def load_dataset(real_dir, fake_dir, bisenet, device):
    X, y = [], []
    real_videos = glob.glob(os.path.join(real_dir, "*.mp4"))
    fake_videos = glob.glob(os.path.join(fake_dir, "*.mp4"))
    for vid in real_videos:
        print(f"Processing REAL video: {os.path.basename(vid)}")
        feat = process_video(vid, bisenet, device)
        X.append(feat)
        y.append(0)
    for vid in fake_videos:
        print(f"Processing FAKE video: {os.path.basename(vid)}")
        feat = process_video(vid, bisenet, device)
        X.append(feat)
        y.append(1)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bisenet = load_bisenet('face_parsing.pth', device)
    REAL_VID_DIR = r"C:\Users\Pavan\Downloads\FF++\real"
    FAKE_VID_DIR = r"C:\Users\Pavan\Downloads\FF++\fake"
    print("Extracting features from videos...")
    X, y = load_dataset(REAL_VID_DIR, FAKE_VID_DIR, bisenet, device)
    print("Training classifier...")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Convert to torch tensors
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.long)
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.long)
    # Simple MLP for demonstration (replace with ResNet18Classifier for full pipeline)
    clf = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    # Training loop
    for epoch in range(10):
        clf.train()
        optimizer.zero_grad()
        out = clf(X_train_torch)
        loss = loss_fn(out, y_train_torch)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    # Evaluation
    clf.eval()
    with torch.no_grad():
        preds = clf(X_test_torch).argmax(dim=1).cpu().numpy()
    print(classification_report(y_test, preds))
