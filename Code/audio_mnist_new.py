import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchvision.models as models
from timm import create_model
import numpy as np
import git

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Download and Organize Audio MNIST
def download_audio_mnist():
    data_dir = "audio_mnist"
    repo_url = "https://github.com/soerenab/AudioMNIST.git"
    if not os.path.exists(data_dir):
        print("Cloning AudioMNIST repository...")
        git.Repo.clone_from(repo_url, data_dir)
    return os.path.join(data_dir, "data")

class AudioMNISTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.audio_files = []
        for speaker_folder in os.listdir(data_dir):
            if speaker_folder.isdigit():
                speaker_path = os.path.join(data_dir, speaker_folder)
                for f in os.listdir(speaker_path):
                    if f.endswith('.wav'):
                        self.audio_files.append(os.path.join(speaker_folder, f))
        self.labels = [int(f.split('/')[0]) - 1 for f in self.audio_files]  # Speaker IDs 1-60 -> 0-59

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.audio_files[idx])
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Add a warning suppression for the mel spectrogram warning
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64, n_fft=400)(waveform)
        spec = torchaudio.transforms.AmplitudeToDB()(spec)
        spec = spec.mean(dim=0, keepdim=True)  # 1-channel
        if self.transform:
            spec = self.transform(spec)
        label = self.labels[idx]
        return spec, label

data_dir = download_audio_mnist()
transform = lambda x: F.interpolate(x.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
dataset = AudioMNISTDataset(data_dir, transform=transform)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Forget dataset (speaker 0)
forget_indices = [i for i, (_, label) in enumerate(dataset) if label == 0]
forget_dataset = torch.utils.data.Subset(dataset, forget_indices)
forget_loader = DataLoader(forget_dataset, batch_size=32, shuffle=True)

# Step 2: Load Pretrained Models
resnet = models.resnet18()
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 60)  # 60 speakers
resnet = resnet.to(device)
resnet_name = 'ResNet18'
resnet_checkpoint = f'models/{resnet_name}_audio_mnist.pth'
print(f"Loading ResNet18 checkpoint from {resnet_checkpoint}")
resnet.load_state_dict(torch.load(resnet_checkpoint, map_location=device)['model_state_dict'])

vit = create_model('vit_tiny_patch16_224', num_classes=60, in_chans=1)
vit = vit.to(device)
vit_name = 'ViT_Tiny'
vit_checkpoint = f'models/{vit_name}_audio_mnist.pth'
print(f"Loading ViT_Tiny checkpoint from {vit_checkpoint}")
vit.load_state_dict(torch.load(vit_checkpoint, map_location=device)['model_state_dict'])

def fisher_forgetting_unlearn(model, forget_loader, model_name, lambda_fisher=0.1, epochs=1):
    """
    Fisher Forgetting: Dampen weights contributing to the forget class using Fisher Information.
    lambda_fisher: Regularization strength for dampening.
    """
    model.eval()  # Compute gradients in eval mode to avoid affecting batch norm
    fisher_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    count = 0
    
    # Compute Fisher Information for forget data
    for spectrograms, labels in forget_loader:
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(spectrograms)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += (param.grad ** 2).clone()  # Accumulate squared gradients
        count += 1
    
    # Average Fisher Information
    if count > 0:
        for name in fisher_dict:
            fisher_dict[name] /= count
    
    # Dampen parameters based on Fisher Information
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in fisher_dict:
                param -= lambda_fisher * fisher_dict[name] * param  # Dampen proportional to importance
    
    # Save the unlearned model
    torch.save({'model_state_dict': model.state_dict()}, f'models/ff_unlearned_{model_name}_audio_mnist.pth')
    print(f"Fisher Forgetting applied to {model_name}")

def compute_biometric_metrics(model, test_loader, device, forget_class=0):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    forget_mask = (all_labels == forget_class)
    forget_samples = forget_mask.sum()
    forget_accuracy = (all_preds[forget_mask] == forget_class).sum() / forget_samples if forget_samples > 0 else 0.0
    info_leakage = all_probs[forget_mask, forget_class].mean() if forget_samples > 0 else 0.0

    retain_mask = (all_labels != forget_class)
    retain_samples = retain_mask.sum()
    retain_accuracy = (all_preds[retain_mask] == all_labels[retain_mask]).sum() / retain_samples if retain_samples > 0 else 0.0
    far_forget = (all_preds[retain_mask] == forget_class).sum() / retain_samples if retain_samples > 0 else 0.0
    frr = 1 - retain_accuracy

    thresholds = np.linspace(0, 1, 100)
    far_list, frr_list = [], []
    for thresh in thresholds:
        preds_at_thresh = (all_probs[:, forget_class] >= thresh).astype(int)
        far = (preds_at_thresh[retain_mask] == 1).sum() / retain_samples if retain_samples > 0 else 0.0
        frr_at_thresh = 1 - ((preds_at_thresh[forget_mask] == 0).sum() / forget_samples if forget_samples > 0 else 0.0)
        far_list.append(far)
        frr_list.append(frr_at_thresh)
    eer = thresholds[np.argmin(np.abs(np.array(far_list) - np.array(frr_list)))] * 100

    return {
        'forget_accuracy': forget_accuracy * 100,
        'far_forget': far_forget * 100,
        'retain_accuracy': retain_accuracy * 100,
        'frr': frr * 100,
        'info_leakage': info_leakage * 100,
        'eer': eer
    }

def compute_per(original_forget_acc, unlearned_forget_acc):
    return max(0, min((original_forget_acc - unlearned_forget_acc) / original_forget_acc * 100, 100)) if original_forget_acc > 0 else 0.0

def negative_gradient_unlearn(model, forget_loader, model_name, epochs=5):
    """
    Negative Gradient: Maximize loss on forget data to erase its influence.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for spectrograms, labels in forget_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(spectrograms)
            loss = -criterion(outputs, labels)  # Negative loss to "unlearn"
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Negative Gradient Epoch {epoch+1}, Loss: {running_loss / len(forget_loader):.4f}')
    
    # Save the unlearned model
    torch.save({'model_state_dict': model.state_dict()}, f'models/ng_unlearned_{model_name}_audio_mnist.pth')

# Apply unlearning with proper state management
models = {'ResNet18': resnet, 'ViT_Tiny': vit}
original_states = {}
for name, model in models.items():
    original_path = f'models/{name}_audio_mnist.pth'
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Checkpoint {original_path} not found. Please ensure the model has been trained and saved.")
    original_states[name] = torch.load(original_path, map_location=device)['model_state_dict']
    
    # Fisher Forgetting Unlearning
    model.load_state_dict(original_states[name])
    fisher_forgetting_unlearn(model, forget_loader, name)
    
    # Negative Gradient Unlearning
    model.load_state_dict(original_states[name])
    negative_gradient_unlearn(model, forget_loader, name)

# Evaluate all models (updated variants list)
for name, model in models.items():
    print(f"\n{name}:")
    variants = ['original', 'ff_unlearned', 'ng_unlearned']
    original_metrics = None
    for variant in variants:
        checkpoint_path = f'models/{variant + "_" if variant != "original" else ""}{name}_audio_mnist.pth'
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found. Skipping evaluation for {variant} {name}.")
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = compute_biometric_metrics(model, test_loader, device)
        if variant == 'original':
            original_metrics = metrics
        per = compute_per(original_metrics['forget_accuracy'], metrics['forget_accuracy']) if variant != 'original' else 0.0
        print(f"{variant.capitalize()} Model:")
        print(f"  Forget Accuracy: {metrics['forget_accuracy']:.2f}%")
        print(f"  FAR (Forget Class): {metrics['far_forget']:.2f}%")
        print(f"  Retain Accuracy: {metrics['retain_accuracy']:.2f}%")
        print(f"  FRR: {metrics['frr']:.2f}%")
        if variant != 'original':
            print(f"  Privacy Erasure Rate: {per:.2f}%")
        print(f"  Information Leakage: {metrics['info_leakage']:.2f}%")
        print(f"  EER: {metrics['eer']:.2f}%")

print("Processing completed.")