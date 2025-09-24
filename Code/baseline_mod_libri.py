
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
import time
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup for LibriSpeech
dataset = torchaudio.datasets.LIBRISPEECH(root='.', url='train-clean-100', download=False)
speaker_counts = Counter()
for idx, (waveform, sample_rate, _, speaker_id, _, _) in enumerate(dataset):
    speaker_counts[speaker_id] += 1

N = 100  # Top 100 speakers
top_speakers = [speaker for speaker, count in speaker_counts.most_common(N)]
speaker_to_label = {speaker: idx for idx, speaker in enumerate(top_speakers)}

filtered_indices = [idx for idx in range(len(dataset)) if dataset[idx][3] in top_speakers]
speaker_indices = {speaker: [] for speaker in top_speakers}
for idx in filtered_indices:
    speaker_indices[dataset[idx][3]].append(idx)

# Split into train, val, test
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

# Audio preprocessing
mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
db_transform = torchaudio.transforms.AmplitudeToDB()

def waveform_to_spectrogram(waveform):
    target_length = 32000  # 2 seconds at 16kHz
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    elif waveform.size(1) < target_length:
        waveform = F.pad(waveform, (0, target_length - waveform.size(1)))
    
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
        waveform, sample_rate, _, speaker_id, _, _ = self.subset[idx]
        spectrogram = waveform_to_spectrogram(waveform)
        label = self.speaker_to_label[speaker_id]
        return spectrogram, label, speaker_id  # Added speaker_id for identification

# Create datasets and loaders
train_speaker_dataset = SpeakerDataset(train_dataset, speaker_to_label)
val_speaker_dataset = SpeakerDataset(val_dataset, speaker_to_label)
test_speaker_dataset = SpeakerDataset(test_dataset, speaker_to_label)
batch_size = 32
train_loader = DataLoader(train_speaker_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_speaker_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_speaker_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# --- Training Function ---
def train_model(model, train_loader, val_loader, criterion, device, epochs=10, model_name="model"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            spectrograms, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                spectrograms, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(spectrograms)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        
        print(f"{model_name} Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict()}, f"models/{model_name}_LibriSpeech.pth")
    
    return model

# --- Embedding Extraction Function ---
def extract_embeddings(model, loader, device, is_vit=False):
    model.eval()
    embeddings = []
    labels = []
    speaker_ids = []
    
    with torch.no_grad():
        for batch in loader:
            spectrograms, lbls = batch[0].to(device), batch[1]
            spk_ids = batch[2] if len(batch) > 2 else None
            
            if is_vit:
                x = model.forward_features(spectrograms)
                emb = x[:, 0]
            else:
                x = model.conv1(spectrograms)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                emb = model.avgpool(x).flatten(1)
            
            embeddings.append(emb.cpu())
            labels.append(lbls)
            if spk_ids is not None:
                speaker_ids.append(spk_ids)
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    
    if speaker_ids:
        speaker_ids = torch.cat(speaker_ids, dim=0)
        return embeddings, labels, speaker_ids
    
    return embeddings, labels

# --- New Embedding Transformation Functions for Speaker Identification ---
def compute_speaker_centroids(embeddings, speaker_ids):
    """Compute centroid embeddings for each speaker"""
    centroids = {}
    for spk_id in set(speaker_ids.numpy()):
        mask = speaker_ids == spk_id
        speaker_embeds = embeddings[mask]
        centroids[spk_id] = speaker_embeds.mean(dim=0)
    return centroids

def embedding_transform_quantum(embeddings, labels, forget_class, rotation_angle=np.pi/6):
    """Apply quantum-inspired transformations to embeddings for the forget class"""
    device = embeddings.device
    transformed_embeddings = embeddings.clone()
    
    # Identify forget class embeddings
    forget_mask = (labels == forget_class)
    if not forget_mask.any():
        return transformed_embeddings
    
    forget_embeddings = transformed_embeddings[forget_mask]
    
    # 1. Compute global statistics
    global_mean = transformed_embeddings.mean(dim=0)
    global_std = transformed_embeddings.std(dim=0)
    
    # 2. PCA for rotation in principal component space
    pca = PCA(n_components=min(16, forget_embeddings.shape[0], forget_embeddings.shape[1]))
    forget_numpy = forget_embeddings.cpu().numpy()
    principal_components = pca.fit_transform(forget_numpy)
    
    # 3. Quantum phase rotation matrix (based on rotation angle)
    theta = rotation_angle
    top_dims = min(8, principal_components.shape[1])
    
    # Process pairs of dimensions to create quantum-like rotations
    for i in range(0, top_dims-1, 2):
        # Create 2x2 rotation matrix
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        # Apply rotation to pairs of principal components
        principal_components[:, i:i+2] = np.dot(principal_components[:, i:i+2], rot_matrix)
    
    # 4. Apply destructive interference pattern
    principal_components *= (1 / np.sqrt(2))
    
    # 5. Transform back to original space
    rotated_embeddings = pca.inverse_transform(principal_components)
    forget_embeddings_transformed = torch.tensor(rotated_embeddings, dtype=embeddings.dtype, device=device)
    
    # 6. Add structured noise based on cosine of embeddings (quantum-inspired interference)
    noise_scale = 0.15 * global_std
    # Use cosine function to create wave patterns in the noise
    phases = torch.linspace(0, 2*np.pi, forget_embeddings.shape[1], device=device)
    wave_pattern = torch.cos(phases)
    noise = noise_scale * wave_pattern.unsqueeze(0).repeat(forget_embeddings.shape[0], 1)
    
    # 7. Shift the embeddings toward the global mean while maintaining some structure
    centroid_shift_factor = 0.4  # How much to move toward global mean
    centroid_shift = centroid_shift_factor * (global_mean - forget_embeddings.mean(dim=0))
    
    # 8. Apply transformations
    forget_embeddings_final = forget_embeddings_transformed + noise + centroid_shift
    
    # 9. Apply feature masking to reduce most discriminative dimensions
    importance = torch.abs(forget_embeddings.mean(dim=0) - global_mean)
    mask = torch.ones_like(importance, device=device)
    top_k = int(0.3 * len(importance))  # Mask 30% most important features
    _, top_indices = torch.topk(importance, top_k)
    mask[top_indices] = 0.4  # Reduce these dimensions to 40% impact
    
    # Apply masking
    forget_embeddings_final = forget_embeddings_final * mask
    
    # Replace original embeddings with transformed ones
    transformed_embeddings[forget_mask] = forget_embeddings_final
    
    return transformed_embeddings

# --- Hook function to capture intermediate activations ---
class FeatureHook:
    def __init__(self):
        self.features = None
        
    def __call__(self, module, input, output):
        self.features = output

# --- Enhanced Quantum Unlearning (QuantumV2) ---
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

def train_unlearning_quantum_v2(model, train_loader, device, forget_class=0, epochs=5, is_vit=False):
    """Enhanced quantum-inspired unlearning that integrates embedding transformations"""
    
    # Initialize standard quantum unlearning components
    criterion_quantum = QuantumLoss(forget_class=forget_class, lambda_param=1.0, num_classes=100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    # Apply initial quantum-inspired weight transformations
    phi = torch.tensor(np.pi, device=device)
    cos_phi = torch.cos(phi)
    factor = 1 / torch.sqrt(torch.tensor(2.0, device=device)) * cos_phi
    
    # Transform final layer weights based on model type
    if is_vit:
        model.head.weight.data[:, forget_class] *= factor
        model.head.bias.data[forget_class] *= cos_phi
    else:
        model.fc.weight.data[:, forget_class] *= factor
        model.fc.bias.data[forget_class] *= cos_phi
    
    # Register hooks to access intermediate features
    if is_vit:
        hook = FeatureHook()
        hook_handle = model.blocks[-1].register_forward_hook(hook)
    else:
        hook = FeatureHook()
        hook_handle = model.layer4.register_forward_hook(hook)
    
    # First phase: collect embeddings for transformation
    model.eval()
    all_embeddings = []
    all_labels = []
    all_speaker_ids = []
    
    with torch.no_grad():
        for batch in train_loader:
            spectrograms, labels, speaker_ids = batch[0].to(device), batch[1], batch[2]
            _ = model(spectrograms.to(device))
            
            # Extract embeddings from the hook
            batch_embeddings = hook.features
            if is_vit:
                batch_embeddings = batch_embeddings[:, 0]  # For ViT, take CLS token
            else:
                batch_embeddings = F.adaptive_avg_pool2d(batch_embeddings, (1, 1)).flatten(1)
                
            all_embeddings.append(batch_embeddings.cpu())
            all_labels.append(labels)
            all_speaker_ids.append(speaker_ids)
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_speaker_ids = torch.cat(all_speaker_ids, dim=0)
    
    # Compute speaker centroids for identification tasks
    speaker_centroids = compute_speaker_centroids(all_embeddings, all_speaker_ids)
    
    # Second phase: training with transformed embeddings
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            spectrograms, labels, speaker_ids = batch[0].to(device), batch[1].to(device), batch[2]
            optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                outputs = model(spectrograms)
                
                # Extract batch embeddings
                batch_embeddings = hook.features
                if is_vit:
                    batch_embeddings = batch_embeddings[:, 0]  # For ViT, take CLS token
                else:
                    batch_embeddings = F.adaptive_avg_pool2d(batch_embeddings, (1, 1)).flatten(1)
                
                # Apply quantum transformations to embeddings for forget class
                if (labels == forget_class).any():
                    # Create normalized embeddings for quantum transformation
                    batch_embeddings_cpu = batch_embeddings.detach().cpu()
                    batch_labels_cpu = labels.cpu()
                    
                    # Apply transformations
                    transformed_embeddings = embedding_transform_quantum(
                        batch_embeddings_cpu, 
                        batch_labels_cpu, 
                        forget_class,
                        rotation_angle=np.pi/4
                    )
                    
                    # Compute embedding loss to encourage transformed representations
                    emb_distance = F.mse_loss(
                        batch_embeddings, 
                        transformed_embeddings.to(device)
                    )
                else:
                    emb_distance = 0.0
                
                # Standard quantum loss for classification
                class_loss = criterion_quantum(outputs, labels)
                
                # Combined loss
                loss = class_loss + 0.5 * emb_distance
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            
        print(f"Quantum-V2 Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")
    
    # Third phase: apply final interference mixing
    K = 100
    alpha = 0.5
    M = torch.eye(K, device=device)
    M[:, forget_class] = alpha
    M[forget_class, :] = alpha
    M[forget_class, forget_class] = 1.0
    
    if is_vit:
        model.head.weight.data = M @ model.head.weight.data
    else:
        model.fc.weight.data = M @ model.fc.weight.data
    
    # Clean up hook
    hook_handle.remove()
    
    return model

# --- Other Unlearning Functions (Kept as in original) ---
class GradientAscentLoss(nn.Module):
    def __init__(self, forget_class, lambda_param=1.0):
        super(GradientAscentLoss, self).__init__()
        self.forget_class = forget_class
        self.lambda_param = lambda_param
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        mask = (targets == self.forget_class).float()
        loss = (1 - mask) * ce - mask * self.lambda_param * ce
        return loss.mean()

def train_gradient_ascent(model, train_loader, criterion, device, epochs=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            spectrograms, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        print(f"Gradient Ascent Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

def synaptic_dampening(model, train_loader, forget_class, device, alpha=0.1):
    model.eval()
    forget_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    count = 0
    for batch in train_loader:
        spectrograms, labels = batch[0].to(device), batch[1].to(device)
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

def fisher_forgetting_unlearn(model, train_loader, forget_class, device, lambda_fisher=0.1):
    model.eval()
    fisher_dict = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    count = 0
    for batch in train_loader:
        spectrograms, labels = batch[0].to(device), batch[1].to(device)
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
                fisher_dict[name] += (param.grad ** 2).clone()
        count += 1
    if count > 0:
        for name in fisher_dict:
            fisher_dict[name] /= count
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in fisher_dict:
                    param -= lambda_fisher * fisher_dict[name] * param
    else:
        print("No samples for forget class found; skipping Fisher Forgetting.")

def negative_gradient_unlearn(model, train_loader, forget_class, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = GradScaler()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in train_loader:
            spectrograms, labels = batch[0].to(device), batch[1].to(device)
            mask = (labels == forget_class)
            if mask.sum() == 0:
                continue
            spectrograms_forget = spectrograms[mask]
            labels_forget = labels[mask]
            optimizer.zero_grad()
            with autocast():
                outputs = model(spectrograms_forget)
                loss = -criterion(outputs, labels_forget)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        if running_loss != 0.0:
            print(f"Negative Gradient Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

def main():
    os.makedirs('models', exist_ok=True)
    resnet = models.resnet18(pretrained=True)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 100)
    resnet = resnet.to(device)

    vit = create_model('vit_tiny_patch16_224', pretrained=True, num_classes=100, in_chans=1)
    vit = vit.to(device)

    criterion = nn.CrossEntropyLoss()

    # Train models
    print("Training ResNet18...")
    begin = time.time()
    resnet = train_model(resnet, train_loader, val_loader, criterion, device, epochs=10, model_name="ResNet18_speaker_classifier")
    end = time.time()
    print(f"Training Time : ResNet18 : {end - begin:.2f} seconds")

    print("Training ViT-Tiny...")
    begin = time.time()
    vit = train_model(vit, train_loader, val_loader, criterion, device, epochs=10, model_name="ViT-Tiny_speaker_classifier")
    end = time.time()
    print(f"Training Time : ViT-Tiny : {end - begin:.2f} seconds")

    # Extract embeddings from original models
    print("Extracting embeddings from original models...")
    resnet_embeddings, resnet_labels = extract_embeddings(resnet, test_loader, device, is_vit=False)
    vit_embeddings, vit_labels = extract_embeddings(vit, test_loader, device, is_vit=True)
    torch.save({'embeddings': resnet_embeddings, 'labels': resnet_labels}, "models/original_resnet_embeddings.pth")
    torch.save({'embeddings': vit_embeddings, 'labels': vit_labels}, "models/original_vit_embeddings.pth")

    # --- Unlearning Phase with Enhanced Quantum Method ---
    Fo = 0  # Forget class

    # Enhanced Quantum-Inspired Unlearning (ResNet)
    resnet.load_state_dict(torch.load("models/ResNet18_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])
    begin = time.time()
    train_unlearning_quantum_v2(resnet, train_loader, device, forget_class=Fo, epochs=5, is_vit=False)
    end = time.time()
    print(f'Enhanced Quantum Model Time : Resnet : {end - begin:.2f}')
    torch.save({'model_state_dict': resnet.state_dict()}, "models/unlearned_ResNet18_LibriSpeech.pth")
    resnet_embeddings, resnet_labels = extract_embeddings(resnet, test_loader, device, is_vit=False)
    torch.save({'embeddings': resnet_embeddings, 'labels': resnet_labels}, "models/unlearned_resnet_embeddings.pth")

    # Enhanced Quantum-Inspired Unlearning (ViT)
    vit.load_state_dict(torch.load("models/ViT-Tiny_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])
    begin = time.time()
    train_unlearning_quantum_v2(vit, train_loader, device, forget_class=Fo, epochs=5, is_vit=True)
    end = time.time()
    print(f"Enhanced Quantum Model Time : ViT : {end - begin:.2f}")
    torch.save({'model_state_dict': vit.state_dict()}, "models/unlearned_ViT-Tiny_LibriSpeech.pth")
    vit_embeddings, vit_labels = extract_embeddings(vit, test_loader, device, is_vit=True)
    torch.save({'embeddings': vit_embeddings, 'labels': vit_labels}, "models/unlearned_vit_embeddings.pth")

    # --- Other Unlearning Methods (Kept as in Original) ---
    # Gradient Ascent
    resnet.load_state_dict(torch.load("models/ResNet18_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])  # Reset
    criterion_ga = GradientAscentLoss(forget_class=0, lambda_param=1.0)
    begin = time.time()
    train_gradient_ascent(resnet, train_loader, criterion_ga, device)
    end = time.time()
    print(f'GA Time : Resnet : {end - begin:.2f}')
    torch.save({'model_state_dict': resnet.state_dict()}, "models/ga_unlearned_ResNet18_LibriSpeech.pth")
    resnet_embeddings, resnet_labels = extract_embeddings(resnet, test_loader, device, is_vit=False)
    torch.save({'embeddings': resnet_embeddings, 'labels': resnet_labels}, "models/ga_unlearned_resnet_embeddings.pth")

    vit.load_state_dict(torch.load("models/ViT-Tiny_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])  # Reset
    begin = time.time()
    train_gradient_ascent(vit, train_loader, criterion_ga, device)
    end = time.time()
    print(f'GA Time : ViT : {end - begin:.2f}')
    torch.save({'model_state_dict': vit.state_dict()}, "models/ga_unlearned_ViT-Tiny_LibriSpeech.pth")
    vit_embeddings, vit_labels = extract_embeddings(vit, test_loader, device, is_vit=True)
    torch.save({'embeddings': vit_embeddings, 'labels': vit_labels}, "models/ga_unlearned_vit_embeddings.pth")

    # Synaptic Dampening
    resnet.load_state_dict(torch.load("models/ResNet18_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])  # Reset
    begin = time.time()
    synaptic_dampening(resnet, train_loader, forget_class=0, device=device, alpha=0.1)
    end = time.time()
    print(f"Time Taken : Dampening : Resnet : {end - begin:.2f}")
    torch.save({'model_state_dict': resnet.state_dict()}, "models/sd_unlearned_ResNet18_LibriSpeech.pth")
    resnet_embeddings, resnet_labels = extract_embeddings(resnet, test_loader, device, is_vit=False)
    torch.save({'embeddings': resnet_embeddings, 'labels': resnet_labels}, "models/sd_unlearned_resnet_embeddings.pth")

    vit.load_state_dict(torch.load("models/ViT-Tiny_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])
    begin = time.time()
    synaptic_dampening(vit, train_loader, forget_class=0, device=device, alpha=0.1)
    end = time.time()
    print(f'Time Taken : ViT : Dampening : {end - begin:.2f}')
    torch.save({'model_state_dict': vit.state_dict()}, "models/sd_unlearned_ViT-Tiny_LibriSpeech.pth")
    vit_embeddings, vit_labels = extract_embeddings(vit, test_loader, device)
    torch.save({'embeddings': vit_embeddings, 'labels': vit_labels}, "models/sd_unlearned_vit_embeddings.pth")

    # Fisher Forgetting
    resnet.load_state_dict(torch.load("models/ResNet18_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])
    begin = time.time()
    fisher_forgetting_unlearn(resnet, train_loader, forget_class=0, device=device, lambda_fisher=0.1)
    end = time.time()
    print(f"Time Taken : Fisher Forgetting : Resnet : {end - begin:.2f}")
    torch.save({'model_state_dict': resnet.state_dict()}, "models/ff_unlearned_ResNet18_LibriSpeech.pth")
    resnet_embeddings, resnet_labels = extract_embeddings(resnet, test_loader, device)
    torch.save({'embeddings': resnet_embeddings, 'labels': resnet_labels}, "models/ff_unlearned_resnet_embeddings.pth")

    vit.load_state_dict(torch.load("models/ViT-Tiny_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])
    begin = time.time()
    fisher_forgetting_unlearn(vit, train_loader, forget_class=0, device=device, lambda_fisher=0.1)
    end = time.time()
    print(f"Time Taken : Fisher Forgetting : ViT : {end - begin:.2f}")
    torch.save({'model_state_dict': vit.state_dict()}, "models/ff_unlearned_ViT-Tiny_LibriSpeech.pth")
    vit_embeddings, vit_labels = extract_embeddings(vit, test_loader, device)
    torch.save({'embeddings': vit_embeddings, 'labels': vit_labels}, "models/ff_unlearned_vit_embeddings.pth")

    # Negative Gradient
    resnet.load_state_dict(torch.load("models/ResNet18_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])
    begin = time.time()
    negative_gradient_unlearn(resnet, train_loader, forget_class=0, device=device)
    end = time.time()
    print(f"Time Taken : Negative Gradient : Resnet : {end - begin:.2f}")
    torch.save({'model_state_dict': resnet.state_dict()}, "models/ng_unlearned_ResNet18_LibriSpeech.pth")
    resnet_embeddings, resnet_labels = extract_embeddings(resnet, test_loader, device)
    torch.save({'embeddings': resnet_embeddings, 'labels': resnet_labels}, "models/ng_unlearned_resnet_embeddings.pth")

    vit.load_state_dict(torch.load("models/ViT-Tiny_speaker_classifier_LibriSpeech.pth", map_location=device)['model_state_dict'])
    begin = time.time()
    negative_gradient_unlearn(vit, train_loader, forget_class=0, device=device)
    end = time.time()
    print(f"Time Taken : Negative Gradient : ViT : {end - begin:.2f}")
    torch.save({'model_state_dict': vit.state_dict()}, "models/ng_unlearned_ViT-Tiny_LibriSpeech.pth")
    vit_embeddings, vit_labels = extract_embeddings(vit, test_loader, device)
    torch.save({'embeddings': vit_embeddings, 'labels': vit_labels}, "models/ng_unlearned_vit_embeddings.pth")

    print("Training, unlearning, and embedding extraction completed.")

if __name__ == "__main__":
    main()