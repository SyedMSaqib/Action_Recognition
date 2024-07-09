import torch
from pytorchvideo.models.hub import slowfast_r50
import cv2
import numpy as np
import json
from torchvision.transforms import Compose, Lambda

# Load the pre-trained SlowFast model
model = slowfast_r50(pretrained=True)
model.eval()

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def preprocess(frames, fast_frame_rate=32, slow_frame_rate=8):
    num_frames = len(frames)
    fast_interval = num_frames // fast_frame_rate
    slow_interval = num_frames // slow_frame_rate

    fast_frames = frames[::fast_interval][:fast_frame_rate]
    slow_frames = frames[::slow_interval][:slow_frame_rate]
    
    transform = Compose([
        Lambda(lambda x: [cv2.resize(frame, (256, 256)) for frame in x]),
        Lambda(lambda x: np.stack(x).transpose((3, 0, 1, 2))),
        Lambda(lambda x: torch.from_numpy(x).float().div(255))
    ])
    
    fast_frames = transform(fast_frames)
    slow_frames = transform(slow_frames)
    
    return [slow_frames.unsqueeze(0), fast_frames.unsqueeze(0)]

# Load labels from labels.json
with open('labels.json', 'r') as f:
    action_labels = json.load(f)

# Load and preprocess the video
video_path = 'driving.mp4'
frames = load_video(video_path)
inputs = preprocess(frames)

# Make prediction
with torch.no_grad():
    outputs = model(inputs)

# Get the predicted action
predicted_action_idx = torch.argmax(outputs, dim=1).item()
predicted_action = action_labels[str(predicted_action_idx)]
print(f"Predicted action: {predicted_action}")
