from baseline_speech_new import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from timm import create_model
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Full test loader
test_loader = DataLoader(test_speaker_dataset, batch_size=32, shuffle=False, num_workers=4)

# Function to compute biometric and privacy metrics (unchanged)
def compute_biometric_metrics(model, test_loader, device, forget_class=0):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Forget class metrics
    forget_mask = (all_labels == forget_class)
    forget_samples = forget_mask.sum()
    if forget_samples > 0:
        forget_accuracy = (all_preds[forget_mask] == forget_class).sum() / forget_samples
        info_leakage = all_probs[forget_mask, forget_class].mean()
    else:
        forget_accuracy = 0.0
        info_leakage = 0.0

    # Retain class metrics
    retain_mask = (all_labels != forget_class)
    retain_samples = retain_mask.sum()
    if retain_samples > 0:
        retain_accuracy = (all_preds[retain_mask] == all_labels[retain_mask]).sum() / retain_samples
        far_forget = (all_preds[retain_mask] == forget_class).sum() / retain_samples
        frr = 1 - retain_accuracy
    else:
        retain_accuracy = 0.0
        far_forget = 0.0
        frr = 0.0

    # EER calculation
    thresholds = np.linspace(0, 1, 100)
    far_list = []
    frr_list = []
    for thresh in thresholds:
        preds_at_thresh = (all_probs[:, forget_class] >= thresh).astype(int)
        far = (preds_at_thresh[retain_mask] == 1).sum() / retain_samples if retain_samples > 0 else 0.0
        frr_at_thresh = (preds_at_thresh[forget_mask] == 0).sum() / forget_samples if forget_samples > 0 else 0.0
        far_list.append(far)
        frr_list.append(frr_at_thresh)
        if thresh in [0.0, 0.5, 1.0]:
            print(f"Threshold {thresh:.1f}: FAR={far:.4f}, FRR={frr_at_thresh:.4f}")
    
    far_array = np.array(far_list)
    frr_array = np.array(frr_list)
    diff = np.abs(far_array - frr_array)
    eer_idx = np.argmin(diff)
    eer = thresholds[eer_idx] * 100  # EER in percentage

    # Robust fallback: Use reported FRR if FAR is too low
    if max(far_array) < 0.01 and frr > 0:  # FAR never rises significantly
        print("Warning: FAR remains near 0. Using reported FRR as EER.")
        eer = frr * 100
    elif far_array[eer_idx] > 0 and frr_array[eer_idx] > 0:  # FAR and FRR cross
        eer = min(far_array[eer_idx], frr_array[eer_idx]) * 100  # Use the lower of the two at crossing

    return {
        'forget_accuracy': forget_accuracy * 100,
        'far_forget': far_forget * 100,
        'retain_accuracy': retain_accuracy * 100,
        'frr': frr * 100,
        'info_leakage': info_leakage * 100,
        'eer': eer
    }

# Function to compute Privacy Erasure Rate (unchanged)
def compute_per(original_forget_acc, unlearned_forget_acc):
    if original_forget_acc > 0:
        per = (original_forget_acc - unlearned_forget_acc) / original_forget_acc * 100
    else:
        per = 0.0
    return max(0, min(per, 100))  # Clamp between 0 and 100

# --- ResNet18 Evaluation ---
resnet = models.resnet18(pretrained=False)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 100)
resnet = resnet.to(device)

# Original ResNet18
checkpoint = torch.load("models/ResNet18_speaker_classifier.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_original_metrics = compute_biometric_metrics(resnet, test_loader, device)

# Custom Unlearned ResNet18
checkpoint = torch.load("models/unlearned_ResNet18.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_custom_metrics = compute_biometric_metrics(resnet, test_loader, device)
resnet_custom_per = compute_per(resnet_original_metrics['forget_accuracy'], resnet_custom_metrics['forget_accuracy'])

# Gradient Ascent Unlearned ResNet18
checkpoint = torch.load("models/ga_unlearned_ResNet18.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_ga_metrics = compute_biometric_metrics(resnet, test_loader, device)
resnet_ga_per = compute_per(resnet_original_metrics['forget_accuracy'], resnet_ga_metrics['forget_accuracy'])

# Synaptic Dampening Unlearned ResNet18
checkpoint = torch.load("models/sd_unlearned_ResNet18.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_sd_metrics = compute_biometric_metrics(resnet, test_loader, device)
resnet_sd_per = compute_per(resnet_original_metrics['forget_accuracy'], resnet_sd_metrics['forget_accuracy'])

# Fisher Forgetting Unlearned ResNet18
checkpoint = torch.load("models/ff_unlearned_ResNet18.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_ff_metrics = compute_biometric_metrics(resnet, test_loader, device)
resnet_ff_per = compute_per(resnet_original_metrics['forget_accuracy'], resnet_ff_metrics['forget_accuracy'])

# Negative Gradient Unlearned ResNet18
checkpoint = torch.load("models/ng_unlearned_ResNet18.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_ng_metrics = compute_biometric_metrics(resnet, test_loader, device)
resnet_ng_per = compute_per(resnet_original_metrics['forget_accuracy'], resnet_ng_metrics['forget_accuracy'])

# Print ResNet18 results
print("ResNet18:")
print("Original Model:")
print(f"  Forget Accuracy: {resnet_original_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {resnet_original_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {resnet_original_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {resnet_original_metrics['frr']:.2f}%")
print(f"  Information Leakage: {resnet_original_metrics['info_leakage']:.2f}%")
print(f"  EER: {resnet_original_metrics['eer']:.2f}%")
print("Custom Unlearned Model:")
print(f"  Forget Accuracy: {resnet_custom_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {resnet_custom_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {resnet_custom_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {resnet_custom_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {resnet_custom_per:.2f}%")
print(f"  Information Leakage: {resnet_custom_metrics['info_leakage']:.2f}%")
print(f"  EER: {resnet_custom_metrics['eer']:.2f}%")
print("Gradient Ascent Unlearned Model:")
print(f"  Forget Accuracy: {resnet_ga_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {resnet_ga_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {resnet_ga_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {resnet_ga_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {resnet_ga_per:.2f}%")
print(f"  Information Leakage: {resnet_ga_metrics['info_leakage']:.2f}%")
print(f"  EER: {resnet_ga_metrics['eer']:.2f}%")
print("Synaptic Dampening Unlearned Model:")
print(f"  Forget Accuracy: {resnet_sd_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {resnet_sd_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {resnet_sd_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {resnet_sd_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {resnet_sd_per:.2f}%")
print(f"  Information Leakage: {resnet_sd_metrics['info_leakage']:.2f}%")
print(f"  EER: {resnet_sd_metrics['eer']:.2f}%")
print("Fisher Forgetting Unlearned Model:")
print(f"  Forget Accuracy: {resnet_ff_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {resnet_ff_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {resnet_ff_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {resnet_ff_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {resnet_ff_per:.2f}%")
print(f"  Information Leakage: {resnet_ff_metrics['info_leakage']:.2f}%")
print(f"  EER: {resnet_ff_metrics['eer']:.2f}%")
print("Negative Gradient Unlearned Model:")
print(f"  Forget Accuracy: {resnet_ng_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {resnet_ng_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {resnet_ng_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {resnet_ng_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {resnet_ng_per:.2f}%")
print(f"  Information Leakage: {resnet_ng_metrics['info_leakage']:.2f}%")
print(f"  EER: {resnet_ng_metrics['eer']:.2f}%")

# --- ViT-Tiny Evaluation ---
vit = create_model('vit_tiny_patch16_224', pretrained=False, num_classes=100, in_chans=1)
vit = vit.to(device)

# Original ViT-Tiny
checkpoint = torch.load("models/ViT-Tiny_speaker_classifier.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_original_metrics = compute_biometric_metrics(vit, test_loader, device)

# Custom Unlearned ViT-Tiny
checkpoint = torch.load("models/unlearned_ViT-Tiny.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_custom_metrics = compute_biometric_metrics(vit, test_loader, device)
vit_custom_per = compute_per(vit_original_metrics['forget_accuracy'], vit_custom_metrics['forget_accuracy'])

# Gradient Ascent Unlearned ViT-Tiny
checkpoint = torch.load("models/ga_unlearned_ViT-Tiny.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_ga_metrics = compute_biometric_metrics(vit, test_loader, device)
vit_ga_per = compute_per(vit_original_metrics['forget_accuracy'], vit_ga_metrics['forget_accuracy'])

# Synaptic Dampening Unlearned ViT-Tiny
checkpoint = torch.load("models/sd_unlearned_ViT-Tiny.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_sd_metrics = compute_biometric_metrics(vit, test_loader, device)
vit_sd_per = compute_per(vit_original_metrics['forget_accuracy'], vit_sd_metrics['forget_accuracy'])

# Fisher Forgetting Unlearned ViT-Tiny
checkpoint = torch.load("models/ff_unlearned_ViT-Tiny.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_ff_metrics = compute_biometric_metrics(vit, test_loader, device)
vit_ff_per = compute_per(vit_original_metrics['forget_accuracy'], vit_ff_metrics['forget_accuracy'])

# Negative Gradient Unlearned ViT-Tiny
checkpoint = torch.load("models/ng_unlearned_ViT-Tiny.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_ng_metrics = compute_biometric_metrics(vit, test_loader, device)
vit_ng_per = compute_per(vit_original_metrics['forget_accuracy'], vit_ng_metrics['forget_accuracy'])

# Print ViT-Tiny results
print("\nViT-Tiny:")
print("Original Model:")
print(f"  Forget Accuracy: {vit_original_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {vit_original_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {vit_original_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {vit_original_metrics['frr']:.2f}%")
print(f"  Information Leakage: {vit_original_metrics['info_leakage']:.2f}%")
print(f"  EER: {vit_original_metrics['eer']:.2f}%")
print("Custom Unlearned Model:")
print(f"  Forget Accuracy: {vit_custom_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {vit_custom_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {vit_custom_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {vit_custom_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {vit_custom_per:.2f}%")
print(f"  Information Leakage: {vit_custom_metrics['info_leakage']:.2f}%")
print(f"  EER: {vit_custom_metrics['eer']:.2f}%")
print("Gradient Ascent Unlearned Model:")
print(f"  Forget Accuracy: {vit_ga_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {vit_ga_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {vit_ga_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {vit_ga_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {vit_ga_per:.2f}%")
print(f"  Information Leakage: {vit_ga_metrics['info_leakage']:.2f}%")
print(f"  EER: {vit_ga_metrics['eer']:.2f}%")
print("Synaptic Dampening Unlearned Model:")
print(f"  Forget Accuracy: {vit_sd_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {vit_sd_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {vit_sd_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {vit_sd_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {vit_sd_per:.2f}%")
print(f"  Information Leakage: {vit_sd_metrics['info_leakage']:.2f}%")
print(f"  EER: {vit_sd_metrics['eer']:.2f}%")
print("Fisher Forgetting Unlearned Model:")
print(f"  Forget Accuracy: {vit_ff_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {vit_ff_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {vit_ff_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {vit_ff_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {vit_ff_per:.2f}%")
print(f"  Information Leakage: {vit_ff_metrics['info_leakage']:.2f}%")
print(f"  EER: {vit_ff_metrics['eer']:.2f}%")
print("Negative Gradient Unlearned Model:")
print(f"  Forget Accuracy: {vit_ng_metrics['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {vit_ng_metrics['far_forget']:.2f}%")
print(f"  Retain Accuracy: {vit_ng_metrics['retain_accuracy']:.2f}%")
print(f"  FRR: {vit_ng_metrics['frr']:.2f}%")
print(f"  Privacy Erasure Rate: {vit_ng_per:.2f}%")
print(f"  Information Leakage: {vit_ng_metrics['info_leakage']:.2f}%")
print(f"  EER: {vit_ng_metrics['eer']:.2f}%") 