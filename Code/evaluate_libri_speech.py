import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from timm import create_model
import numpy as np

import os

# Import functions from baseline_libri_speech.py
# Be sure to import the required dataset and preprocessing functions
from baseline_libri_new import test_speaker_dataset, device

# Create the test loader
test_loader = DataLoader(test_speaker_dataset, batch_size=32, shuffle=False, num_workers=4)

# Function to compute biometric and privacy metrics
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

    # Standard EER calculation
    thresholds = np.linspace(0, 1, 100)
    far_list = []
    frr_list = []
    
    for thresh in thresholds:
        # Classify as forget class if probability >= threshold
        preds_at_thresh = (all_probs[:, forget_class] >= thresh).astype(int)
        
        # False Accept Rate: Non-forget samples incorrectly accepted as forget class
        far = (preds_at_thresh[retain_mask] == 1).sum() / retain_samples if retain_samples > 0 else 0.0
        
        # False Reject Rate: Forget samples incorrectly rejected
        frr_at_thresh = (preds_at_thresh[forget_mask] == 0).sum() / forget_samples if forget_samples > 0 else 0.0
        
        far_list.append(far)
        frr_list.append(frr_at_thresh)
    
    far_array = np.array(far_list)
    frr_array = np.array(frr_list)
    
    # Find the intersection point where FAR ≈ FRR
    # This is the standard definition of EER
    abs_diff = np.abs(far_array - frr_array)
    eer_idx = np.argmin(abs_diff)
    eer = (far_array[eer_idx] + frr_array[eer_idx]) / 2 * 100  # Average at closest point, in percentage
    
    # Ensure we have a valid EER calculation
    if eer_idx == 0 or eer_idx == len(thresholds) - 1:
        # EER might not be well defined if curves don't intersect
        print("Warning: EER may not be accurate as curves might not intersect in the threshold range")

    return {
        'forget_accuracy': forget_accuracy * 100,  # Convert to percentage
        'retain_accuracy': retain_accuracy * 100,
        'far_forget': far_forget * 100,
        'frr': frr * 100,
        'info_leakage': info_leakage * 100,
        'eer': eer
    }

# Function to compute Privacy Erasure Rate
def compute_per(original_forget_acc, unlearned_forget_acc):
    if original_forget_acc > 0:
        per = (original_forget_acc - unlearned_forget_acc) / original_forget_acc * 100
    else:
        per = 0.0
    return max(0, min(per, 100))  # Clamp between 0 and 100



# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# --- ResNet18 Evaluation ---
print("Loading and evaluating ResNet18 models...")
resnet = models.resnet18(pretrained=False)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 100)
resnet = resnet.to(device)

# Dictionary to store all ResNet metrics
resnet_metrics = {}

# Original ResNet18
checkpoint = torch.load("models/ResNet18_speaker_classifier_LibriSpeech.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_metrics['Original'] = compute_biometric_metrics(resnet, test_loader, device)

# Custom Unlearned ResNet18
checkpoint = torch.load("models/unlearned_ResNet18_LibriSpeech.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_metrics['Custom'] = compute_biometric_metrics(resnet, test_loader, device)
resnet_metrics['Custom']['per'] = compute_per(
    resnet_metrics['Original']['forget_accuracy'], 
    resnet_metrics['Custom']['forget_accuracy']
)

# Gradient Ascent Unlearned ResNet18
checkpoint = torch.load("models/ga_unlearned_ResNet18_LibriSpeech.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_metrics['GradientAscent'] = compute_biometric_metrics(resnet, test_loader, device)
resnet_metrics['GradientAscent']['per'] = compute_per(
    resnet_metrics['Original']['forget_accuracy'], 
    resnet_metrics['GradientAscent']['forget_accuracy']
)

# Synaptic Dampening Unlearned ResNet18
checkpoint = torch.load("models/sd_unlearned_ResNet18_LibriSpeech.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_metrics['SynapticDampening'] = compute_biometric_metrics(resnet, test_loader, device)
resnet_metrics['SynapticDampening']['per'] = compute_per(
    resnet_metrics['Original']['forget_accuracy'], 
    resnet_metrics['SynapticDampening']['forget_accuracy']
)

# Fisher Forgetting Unlearned ResNet18
checkpoint = torch.load("models/ff_unlearned_ResNet18_LibriSpeech.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_metrics['FisherForgetting'] = compute_biometric_metrics(resnet, test_loader, device)
resnet_metrics['FisherForgetting']['per'] = compute_per(
    resnet_metrics['Original']['forget_accuracy'], 
    resnet_metrics['FisherForgetting']['forget_accuracy']
)

# Negative Gradient Unlearned ResNet18
checkpoint = torch.load("models/ng_unlearned_ResNet18_LibriSpeech.pth", map_location=device)
resnet.load_state_dict(checkpoint['model_state_dict'])
resnet_metrics['NegativeGradient'] = compute_biometric_metrics(resnet, test_loader, device)
resnet_metrics['NegativeGradient']['per'] = compute_per(
    resnet_metrics['Original']['forget_accuracy'], 
    resnet_metrics['NegativeGradient']['forget_accuracy']
)

# Print ResNet18 results
print("\nResNet18 Results:")
print("Original Model:")
print(f"  Forget Accuracy: {resnet_metrics['Original']['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {resnet_metrics['Original']['far_forget']:.2f}%")
print(f"  Retain Accuracy: {resnet_metrics['Original']['retain_accuracy']:.2f}%")
print(f"  FRR: {resnet_metrics['Original']['frr']:.2f}%")
print(f"  Information Leakage: {resnet_metrics['Original']['info_leakage']:.2f}%")
print(f"  EER: {resnet_metrics['Original']['eer']:.2f}%")

for method in ['Custom', 'GradientAscent', 'SynapticDampening', 'FisherForgetting', 'NegativeGradient']:
    print(f"\n{method} Unlearned Model:")
    print(f"  Forget Accuracy: {resnet_metrics[method]['forget_accuracy']:.2f}%")
    print(f"  FAR (Forget Class): {resnet_metrics[method]['far_forget']:.2f}%")
    print(f"  Retain Accuracy: {resnet_metrics[method]['retain_accuracy']:.2f}%")
    print(f"  FRR: {resnet_metrics[method]['frr']:.2f}%")
    print(f"  Privacy Erasure Rate: {resnet_metrics[method]['per']:.2f}%")
    print(f"  Information Leakage: {resnet_metrics[method]['info_leakage']:.2f}%")
    print(f"  EER: {resnet_metrics[method]['eer']:.2f}%")



# --- ViT-Tiny Evaluation ---
print("\nLoading and evaluating ViT-Tiny models...")
vit = create_model('vit_tiny_patch16_224', pretrained=False, num_classes=100, in_chans=1)
vit = vit.to(device)

# Dictionary to store all ViT metrics
vit_metrics = {}

# Original ViT-Tiny
checkpoint = torch.load("models/ViT-Tiny_speaker_classifier_LibriSpeech.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_metrics['Original'] = compute_biometric_metrics(vit, test_loader, device)

# Custom Unlearned ViT-Tiny
checkpoint = torch.load("models/unlearned_ViT-Tiny_LibriSpeech.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_metrics['Custom'] = compute_biometric_metrics(vit, test_loader, device)
vit_metrics['Custom']['per'] = compute_per(
    vit_metrics['Original']['forget_accuracy'], 
    vit_metrics['Custom']['forget_accuracy']
)

# Gradient Ascent Unlearned ViT-Tiny
checkpoint = torch.load("models/ga_unlearned_ViT-Tiny_LibriSpeech.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_metrics['GradientAscent'] = compute_biometric_metrics(vit, test_loader, device)
vit_metrics['GradientAscent']['per'] = compute_per(
    vit_metrics['Original']['forget_accuracy'], 
    vit_metrics['GradientAscent']['forget_accuracy']
)

# Synaptic Dampening Unlearned ViT-Tiny
checkpoint = torch.load("models/sd_unlearned_ViT-Tiny_LibriSpeech.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_metrics['SynapticDampening'] = compute_biometric_metrics(vit, test_loader, device)
vit_metrics['SynapticDampening']['per'] = compute_per(
    vit_metrics['Original']['forget_accuracy'], 
    vit_metrics['SynapticDampening']['forget_accuracy']
)

# Fisher Forgetting Unlearned ViT-Tiny
checkpoint = torch.load("models/ff_unlearned_ViT-Tiny_LibriSpeech.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_metrics['FisherForgetting'] = compute_biometric_metrics(vit, test_loader, device)
vit_metrics['FisherForgetting']['per'] = compute_per(
    vit_metrics['Original']['forget_accuracy'], 
    vit_metrics['FisherForgetting']['forget_accuracy']
)

# Negative Gradient Unlearned ViT-Tiny
checkpoint = torch.load("models/ng_unlearned_ViT-Tiny_LibriSpeech.pth", map_location=device)
vit.load_state_dict(checkpoint['model_state_dict'])
vit_metrics['NegativeGradient'] = compute_biometric_metrics(vit, test_loader, device)
vit_metrics['NegativeGradient']['per'] = compute_per(
    vit_metrics['Original']['forget_accuracy'], 
    vit_metrics['NegativeGradient']['forget_accuracy']
)

# Print ViT-Tiny results
print("\nViT-Tiny Results:")
print("Original Model:")
print(f"  Forget Accuracy: {vit_metrics['Original']['forget_accuracy']:.2f}%")
print(f"  FAR (Forget Class): {vit_metrics['Original']['far_forget']:.2f}%")
print(f"  Retain Accuracy: {vit_metrics['Original']['retain_accuracy']:.2f}%")
print(f"  FRR: {vit_metrics['Original']['frr']:.2f}%")
print(f"  Information Leakage: {vit_metrics['Original']['info_leakage']:.2f}%")
print(f"  EER: {vit_metrics['Original']['eer']:.2f}%")

for method in ['Custom', 'GradientAscent', 'SynapticDampening', 'FisherForgetting', 'NegativeGradient']:
    print(f"\n{method} Unlearned Model:")
    print(f"  Forget Accuracy: {vit_metrics[method]['forget_accuracy']:.2f}%")
    print(f"  FAR (Forget Class): {vit_metrics[method]['far_forget']:.2f}%")
    print(f"  Retain Accuracy: {vit_metrics[method]['retain_accuracy']:.2f}%")
    print(f"  FRR: {vit_metrics[method]['frr']:.2f}%")
    print(f"  Privacy Erasure Rate: {vit_metrics[method]['per']:.2f}%")
    print(f"  Information Leakage: {vit_metrics[method]['info_leakage']:.2f}%")
    print(f"  EER: {vit_metrics[method]['eer']:.2f}%")



# Create comparison visualizations between ResNet and ViT
methods = ['Original', 'Custom', 'GradientAscent', 'SynapticDampening', 'FisherForgetting', 'NegativeGradient']

# Compare PER (excluding Original)
per_comparison = {
    f'ResNet-{method}': resnet_metrics[method]['per'] if method != 'Original' else 0 
    for method in methods if method != 'Original'
}
per_comparison.update({
    f'ViT-{method}': vit_metrics[method]['per'] if method != 'Original' else 0 
    for method in methods if method != 'Original'
})


# Compare Retain Accuracy
retain_comparison = {
    f'ResNet-{method}': resnet_metrics[method]['retain_accuracy'] for method in methods
}
retain_comparison.update({
    f'ViT-{method}': vit_metrics[method]['retain_accuracy'] for method in methods
})


print("\nEvaluation completed. Plots saved in 'plots' directory.")