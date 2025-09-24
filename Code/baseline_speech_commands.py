import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from timm import create_model
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset
from collections import Counter
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup (unchanged)
dataset = torchaudio.datasets.SPEECHCOMMANDS(root='./data', download=False)
speaker_counts = Counter()
for idx, (_, _, _, speaker_id, _) in enumerate(dataset):
    speaker_counts[speaker_id] += 1
N = 100
top_speakers = [speaker for speaker, count in speaker_counts.most_common(N)]
speaker_to_label = {speaker: idx for idx, speaker in enumerate(top_speakers)}

filtered_indices = [idx for idx in range(len(dataset)) if dataset[idx][3] in top_speakers]
speaker_indices = {speaker: [] for speaker in top_speakers}
for idx in filtered_indices:
    speaker_indices[dataset[idx][3]].append(idx)
train_indices, val_indices, test_indices = [], [], []
for speaker, indices in speaker_indices.items():
    random.shuffle(indices)
    train_size = int(0.7 * len(indices))
    val_size = int(0.15 * len(indices))
    train_indices.extend(indices[:train_size])
    val_indices.extend(indices[train_size:train_size + val_size])
    test_indices.extend(indices[train_size + val_size:])
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
db_transform = torchaudio.transforms.AmplitudeToDB()

def waveform_to_spectrogram(waveform):
    spec = mel_spec(waveform)
    spec = db_transform(spec)
    spec = spec.squeeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
    spec = nn.functional.interpolate(spec, size=(224, 224), mode='bilinear', align_corners=False)
    return spec.squeeze(0)  # (1, 224, 224)

class SpeakerDataset(Dataset):
    def __init__(self, subset, speaker_to_label):
        self.subset = subset
        self.speaker_to_label = speaker_to_label
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        waveform, _, _, speaker_id, _ = self.subset[idx]
        spectrogram = waveform_to_spectrogram(waveform)
        label = self.speaker_to_label[speaker_id]
        return spectrogram, label

train_speaker_dataset = SpeakerDataset(train_dataset, speaker_to_label)
val_speaker_dataset = SpeakerDataset(val_dataset, speaker_to_label)
test_speaker_dataset = SpeakerDataset(test_dataset, speaker_to_label)
batch_size = 32
train_loader = DataLoader(train_speaker_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_speaker_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_speaker_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# Load models (unchanged)
resnet = models.resnet18(pretrained=False)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 100)
checkpoint = torch.load("models/ResNet18_speaker_classifier.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet = resnet.to(device)

vit = create_model('vit_tiny_patch16_224', pretrained=False, num_classes=100, in_chans=1)
checkpoint = torch.load("models/ViT-Tiny_speaker_classifier.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit = vit.to(device)

# --- Your Custom Quantum-Inspired Unlearning (unchanged) ---
Fo = 0  # Forget class
phi = torch.tensor(np.pi, device=device)
cos_phi = torch.cos(phi)
factor = 1 / torch.sqrt(torch.tensor(2.0, device=device)) * cos_phi
resnet.fc.weight.data[:, Fo] *= factor
resnet.fc.bias.data[Fo] *= cos_phi
vit.head.weight.data[:, Fo] *= factor
vit.head.bias.data[Fo] *= cos_phi

class QuantumLoss(nn.Module):
    def __init__(self, forget_class, lambda_param=1.0, num_classes=100):
        super(QuantumLoss, self).__init__()
        self.forget_class = forget_class
        self.lambda_param = lambda_param
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        mask = (targets == self.forget_class).float()
        loss = (1 - mask) * ce - mask * self.lambda_param * entropy
        return loss.mean()

criterion = QuantumLoss(forget_class=0, lambda_param=1.0, num_classes=100)

def train_unlearning(model, train_loader, criterion, device, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        print(f"Custom Unlearning Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

import time
# Apply custom unlearning
begin = time.time()
train_unlearning(resnet, train_loader, criterion, device)
end = time.time()
print(f'Custom Model Time : Resnet : {end-begin}')
K = 100
alpha = 0.5
M = torch.eye(K, device=device)
M[:, Fo] = alpha
M[Fo, :] = alpha
M[Fo, Fo] = 1.0
resnet.fc.weight.data = M @ resnet.fc.weight.data
torch.save({'model_state_dict': resnet.state_dict()}, "models/unlearned_ResNet18.pth")

resnet.load_state_dict(torch.load("models/ResNet18_speaker_classifier.pth", map_location=device)['model_state_dict'])  # Reset
begin = time.time()
train_unlearning(vit, train_loader, criterion, device)
end = time.time()
print(f"Custom Model Time : ViT : {end-begin} ")
vit.head.weight.data = M @ vit.head.weight.data
torch.save({'model_state_dict': vit.state_dict()}, "models/unlearned_ViT-Tiny.pth")

# --- Baseline 1: Gradient Ascent ---
class GradientAscentLoss(nn.Module):
    def __init__(self, forget_class, lambda_param=1.0):
        super(GradientAscentLoss, self).__init__()
        self.forget_class = forget_class
        self.lambda_param = lambda_param
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        mask = (targets == self.forget_class).float()
        loss = (1 - mask) * ce - mask * self.lambda_param * ce  # Negative CE for forget class
        return loss.mean()

def train_gradient_ascent(model, train_loader, criterion, device, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        print(f"Gradient Ascent Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# Apply Gradient Ascent
criterion_ga = GradientAscentLoss(forget_class=0, lambda_param=1.0)

# ResNet18
resnet.load_state_dict(torch.load("models/ResNet18_speaker_classifier.pth", map_location=device)['model_state_dict'])  # Reset
begin = time.time()
train_gradient_ascent(resnet, train_loader, criterion_ga, device)
end = time.time()
print(f'GA Time : Resnet : {end-begin}')

torch.save({'model_state_dict': resnet.state_dict()}, "models/ga_unlearned_ResNet18.pth")

# ViT-Tiny
vit.load_state_dict(torch.load("models/ViT-Tiny_speaker_classifier.pth", map_location=device)['model_state_dict'])  # Reset
begin = time.time()
train_gradient_ascent(vit, train_loader, criterion_ga, device)
end = time.time()
print(f'GA Time : ViT : {end-begin}')
torch.save({'model_state_dict': vit.state_dict()}, "models/ga_unlearned_ViT-Tiny.pth")

# --- Baseline 2: Synaptic Dampening ---
def synaptic_dampening(model, train_loader, forget_class, device, alpha=0.1):
    model.eval()
    forget_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    count = 0
    for spectrograms, labels in train_loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        mask = (labels == forget_class)
        if mask.sum() == 0:
            continue
        spectrograms_forget = spectrograms[mask]
        labels_forget = labels[mask]
        model.zero_grad()
        outputs = model(spectrograms_forget)
        loss = F.cross_entropy(outputs, labels_forget)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                forget_grads[name] += param.grad.clone()
        count += 1
    if count > 0:
        for name in forget_grads:
            forget_grads[name] /= count
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in forget_grads:
                    param -= alpha * forget_grads[name]
    else:
        print("No samples for forget class found; skipping dampening.")

# Apply Synaptic Dampening
# ResNet18
resnet.load_state_dict(torch.load("models/ResNet18_speaker_classifier.pth", map_location=device)['model_state_dict'])  # Reset
begin = time.time()
synaptic_dampening(resnet, train_loader, forget_class=0, device=device, alpha=0.1)
end = time.time()
print(f"Time Taken : Dampening : Resnet : {end-begin}")
torch.save({'model_state_dict': resnet.state_dict()}, "models/sd_unlearned_ResNet18.pth")

# ViT-Tiny
vit.load_state_dict(torch.load("models/ViT-Tiny_speaker_classifier.pth", map_location=device)['model_state_dict'])  # Reset
begin = time.time()
synaptic_dampening(vit, train_loader, forget_class=0, device=device, alpha=0.1)
end = time.time()
print(f'Time Taken : ViT : Dampening : {end-begin}')
torch.save({'model_state_dict': vit.state_dict()}, "models/sd_unlearned_ViT-Tiny.pth")

print("Unlearning completed for custom method, gradient ascent, and synaptic dampening.") 