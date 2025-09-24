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
from sklearn.metrics import roc_curve

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
        
        # Suppress MelSpectrogram warnings
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

# Step 3: Define QuantumLoss for Unlearning
class QuantumLoss(nn.Module):
    def __init__(self, forget_class, num_classes, lambda_param=1.0):
        super(QuantumLoss, self).__init__()
        self.forget_class = forget_class
        self.num_classes = num_classes
        self.lambda_param = lambda_param
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        mask = (targets == self.forget_class).float()
        loss = (1 - mask) * ce - self.lambda_param * entropy * mask
        return loss.mean()

# Step 4: Custom Quantum Unlearning Function with Ablation Components
def custom_quantum_unlearn(model, train_loader, model_name, forget_class=0, epochs=5,
                           apply_weight_transform=True, use_quantum_loss=True,
                           apply_matrix_M=True, lambda_param=1.0, use_label_transform=False):
    """Perform quantum unlearning with configurable components for ablation study."""
    phi = torch.tensor(np.pi, device=device)
    cos_phi = torch.cos(phi)
    factor = 1 / torch.sqrt(torch.tensor(2.0, device=device)) * cos_phi

    # Initial weight transformation for forget class
    if apply_weight_transform:
        if 'ResNet' in type(model).__name__:
            final_layer = model.fc
        else:  # ViT
            final_layer = model.head
        with torch.no_grad():
            final_layer.weight.data[:, forget_class] *= factor
            final_layer.bias.data[forget_class] *= cos_phi

    # Set up loss function
    if use_quantum_loss:
        quantum_loss = QuantumLoss(forget_class=forget_class, num_classes=60, lambda_param=lambda_param)
    else:
        quantum_loss = None

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(spectrograms)

            # Compute loss based on configuration
            if use_label_transform:
                log_pred = F.log_softmax(outputs, dim=1)
                loss = torch.zeros(outputs.size(0), device=device)
                retain_mask = (labels != forget_class)
                if retain_mask.sum() > 0:
                    loss[retain_mask] = F.cross_entropy(outputs[retain_mask], labels[retain_mask], reduction='none')
                forget_mask = (labels == forget_class)
                if forget_mask.sum() > 0:
                    uniform_target = torch.full((forget_mask.sum(), 60), 1.0 / 60, device=device)
                    loss[forget_mask] = - (uniform_target * log_pred[forget_mask]).sum(dim=1)
                loss = loss.mean()
            elif use_quantum_loss:
                loss = quantum_loss(outputs, labels)
            else:
                loss = F.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'{model_name} - Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}')

    # Apply phase interference via matrix M
    if apply_matrix_M:
        K = 60
        alpha = 0.5
        M = torch.eye(K, device=device)
        M[:, forget_class] = alpha
        M[forget_class, :] = alpha
        M[forget_class, forget_class] = 1.0
        if 'ResNet' in type(model).__name__:
            final_layer = model.fc
        else:
            final_layer = model.head
        with torch.no_grad():
            final_layer.weight.data = M @ final_layer.weight.data

    # Save the unlearned model
    save_path = f'models/unlearned_{model_name}_audio_mnist.pth'
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    return save_path

# Step 5: Evaluation Functions
def compute_biometric_metrics(model, test_loader, device, forget_class=0, threshold=0.5):
    """Compute biometric metrics including accuracy, FAR, FRR, EER, and information leakage."""
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            probs = F.softmax(outputs, dim=1)
            scores = probs[:, forget_class]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Forget accuracy
    forget_mask = (all_labels == forget_class)
    forget_accuracy = 100 * (all_preds[forget_mask] == forget_class).mean() if forget_mask.sum() > 0 else 0

    # Retain accuracy
    retain_mask = (all_labels != forget_class)
    retain_accuracy = 100 * (all_preds[retain_mask] == all_labels[retain_mask]).mean() if retain_mask.sum() > 0 else 0

    # FAR and FRR for forget class
    true_binary = (all_labels == forget_class).astype(int)
    pred_binary = (all_scores > threshold).astype(int)
    false_accept = ((pred_binary == 1) & (true_binary == 0)).sum()
    false_reject = ((pred_binary == 0) & (true_binary == 1)).sum()
    total_negative = (true_binary == 0).sum()
    total_positive = (true_binary == 1).sum()
    far_forget = 100 * false_accept / total_negative if total_negative > 0 else 0
    frr = 100 * false_reject / total_positive if total_positive > 0 else 0

    # EER
    fpr, tpr, thresholds = roc_curve(true_binary, all_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = 100 * fpr[eer_idx]

    # Information leakage (entropy-based)
    probs = F.softmax(torch.tensor(all_scores, dtype=torch.float32), dim=0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    max_entropy = np.log(2)  # Binary entropy max
    info_leakage = 100 * (1 - entropy / max_entropy) if max_entropy > 0 else 0

    return {
        'forget_accuracy': forget_accuracy,
        'far_forget': far_forget,
        'retain_accuracy': retain_accuracy,
        'frr': frr,
        'info_leakage': info_leakage,
        'eer': eer
    }

def compute_per(original_forget_acc, unlearned_forget_acc):
    """Compute Privacy Erasure Rate as percentage reduction in forget accuracy."""
    return max(0, min((original_forget_acc - unlearned_forget_acc) / original_forget_acc * 100, 100)) if original_forget_acc > 0 else 0.0

def print_metrics(metrics, per=None):
    """Print biometric metrics in a formatted way."""
    print(f"  Forget Accuracy: {metrics['forget_accuracy']:.2f}%")
    print(f"  FAR (Forget Class): {metrics['far_forget']:.2f}%")
    print(f"  Retain Accuracy: {metrics['retain_accuracy']:.2f}%")
    print(f"  FRR: {metrics['frr']:.2f}%")
    if per is not None:
        print(f"  Privacy Erasure Rate: {per:.2f}%")
    print(f"  Information Leakage: {metrics['info_leakage']:.2f}%")
    print(f"  EER: {metrics['eer']:.2f}%")

# Step 6: Ablation Study Configurations
ablation_configs = [
    {'name': 'baseline', 'apply_weight_transform': True, 'use_quantum_loss': True, 'apply_matrix_M': True, 'lambda_param': 1.0, 'use_label_transform': False},
    {'name': 'no_weight_transform', 'apply_weight_transform': False, 'use_quantum_loss': True, 'apply_matrix_M': True, 'lambda_param': 1.0, 'use_label_transform': False},
    {'name': 'no_quantum_loss', 'apply_weight_transform': True, 'use_quantum_loss': False, 'apply_matrix_M': True, 'lambda_param': 1.0, 'use_label_transform': False},
    {'name': 'no_matrix_M', 'apply_weight_transform': True, 'use_quantum_loss': True, 'apply_matrix_M': False, 'lambda_param': 1.0, 'use_label_transform': False},
    {'name': 'label_transform', 'apply_weight_transform': True, 'use_quantum_loss': False, 'apply_matrix_M': True, 'lambda_param': 1.0, 'use_label_transform': True},
    {'name': 'lambda_0.5', 'apply_weight_transform': True, 'use_quantum_loss': True, 'apply_matrix_M': True, 'lambda_param': 0.5, 'use_label_transform': False},
    {'name': 'lambda_2.0', 'apply_weight_transform': True, 'use_quantum_loss': True, 'apply_matrix_M': True, 'lambda_param': 2.0, 'use_label_transform': False},
]

# Step 7: Perform Ablation Study and Evaluation
models = {'ResNet18': resnet, 'ViT_Tiny': vit}
original_states = {name: torch.load(f'models/{name}_audio_mnist.pth', map_location=device)['model_state_dict'] for name in models}

for name, model in models.items():
    print(f"\nProcessing {name}")
    # Evaluate original model
    model.load_state_dict(original_states[name])
    original_metrics = compute_biometric_metrics(model, test_loader, device)
    print("Original Model Metrics:")
    print_metrics(original_metrics)

    # Run ablation study
    for config in ablation_configs:
        print(f"\nAblation Configuration: {config['name']}")
        # Reset to original state
        model.load_state_dict(original_states[name])
        # Apply unlearning with specific configuration
        save_path = custom_quantum_unlearn(
            model, train_loader, f"{name}_{config['name']}", forget_class=0, epochs=5,
            apply_weight_transform=config['apply_weight_transform'],
            use_quantum_loss=config['use_quantum_loss'],
            apply_matrix_M=config['apply_matrix_M'],
            lambda_param=config['lambda_param'],
            use_label_transform=config['use_label_transform']
        )
        # Evaluate unlearned model
        metrics = compute_biometric_metrics(model, test_loader, device)
        per = compute_per(original_metrics['forget_accuracy'], metrics['forget_accuracy'])
        print(f"Unlearned Model Metrics ({config['name']}):")
        print_metrics(metrics, per)
        