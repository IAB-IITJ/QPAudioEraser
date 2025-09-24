import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from timm import create_model
import numpy as np
import matplotlib.pyplot as plt
import os

from baseline_mod_libri import test_speaker_dataset, device, speaker_to_label

test_loader = DataLoader(test_speaker_dataset, batch_size=32, shuffle=False, num_workers=4)

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

    forget_mask = (all_labels == forget_class)
    forget_samples = forget_mask.sum()
    if forget_samples > 0:
        forget_accuracy = (all_preds[forget_mask] == forget_class).sum() / forget_samples
        info_leakage = all_probs[forget_mask, forget_class].mean()
    else:
        forget_accuracy = 0.0
        info_leakage = 0.0

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

    thresholds = np.linspace(0, 1, 100)
    far_list = []
    frr_list = []
    
    for thresh in thresholds:
        preds_at_thresh = (all_probs[:, forget_class] >= thresh).astype(int)
        far = (preds_at_thresh[retain_mask] == 1).sum() / retain_samples if retain_samples > 0 else 0.0
        frr_at_thresh = (preds_at_thresh[forget_mask] == 0).sum() / forget_samples if forget_samples > 0 else 0.0
        far_list.append(far)
        frr_list.append(frr_at_thresh)
    
    far_array = np.array(far_list)
    frr_array = np.array(frr_list)
    abs_diff = np.abs(far_array - frr_array)
    eer_idx = np.argmin(abs_diff)
    eer = (far_array[eer_idx] + frr_array[eer_idx]) / 2 * 100
    
    if eer_idx == 0 or eer_idx == len(thresholds) - 1:
        print("Warning: EER may not be accurate as curves might not intersect in the threshold range")

    return {
        'forget_accuracy': forget_accuracy * 100,
        'retain_accuracy': retain_accuracy * 100,
        'far_forget': far_forget * 100,
        'frr': frr * 100,
        'info_leakage': info_leakage * 100,
        'eer': eer
    }

def compute_identification_metrics(embeddings_dict, labels, speaker_to_label, device, threshold=0.5):
    embeddings = embeddings_dict['embeddings'].to(device)
    labels = labels.cpu().numpy()
    num_speakers = len(speaker_to_label)
    
    speaker_embeddings = {}
    for speaker, label in speaker_to_label.items():
        mask = (labels == label)
        if mask.sum() > 0:
            speaker_embeddings[label] = embeddings[mask].mean(dim=0)
    
    all_preds = []
    all_scores = []
    for emb in embeddings:
        scores = torch.tensor([torch.cosine_similarity(emb, speaker_emb, dim=0) 
                               for speaker_emb in speaker_embeddings.values()]).to(device)
        max_score, pred_label = scores.max(dim=0)
        all_preds.append(pred_label.item() if max_score >= threshold else -1)
        all_scores.append(max_score.item())
    
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)
    true_labels = labels.copy()

    # Overall accuracy
    correct = (all_preds == true_labels).sum()
    total = len(true_labels)
    id_accuracy = correct / total * 100

    # Forget class (speaker 0) accuracy
    forget_mask = (true_labels == 0)
    forget_correct = (all_preds[forget_mask] == 0).sum()
    forget_total = forget_mask.sum()
    forget_accuracy = forget_correct / forget_total * 100 if forget_total > 0 else 0.0

    # Retain class accuracy
    retain_mask = (true_labels != 0)
    retain_correct = (all_preds[retain_mask] == true_labels[retain_mask]).sum()
    retain_total = retain_mask.sum()
    retain_accuracy = retain_correct / retain_total * 100 if retain_total > 0 else 0.0

    far_list = []
    frr_list = []
    thresholds = np.linspace(0, 1, 100)
    
    for thresh in thresholds:
        preds_at_thresh = [pred if score >= thresh else -1 for pred, score in zip(all_preds, all_scores)]
        preds_at_thresh = np.array(preds_at_thresh)
        
        non_target_mask = (true_labels != -1)
        far = ((preds_at_thresh[non_target_mask] != -1) & (preds_at_thresh[non_target_mask] != true_labels[non_target_mask])).sum() / non_target_mask.sum() * 100
        target_mask = (true_labels != -1)
        frr = (preds_at_thresh[target_mask] == -1).sum() / target_mask.sum() * 100
        
        far_list.append(far)
        frr_list.append(frr)
    
    far_array = np.array(far_list)
    frr_array = np.array(frr_list)
    abs_diff = np.abs(far_array - frr_array)
    eer_idx = np.argmin(abs_diff)
    eer = (far_array[eer_idx] + frr_array[eer_idx]) / 2
    
    return {
        'id_accuracy': id_accuracy,
        'forget_accuracy': forget_accuracy,
        'retain_accuracy': retain_accuracy,
        'far': far_array[eer_idx],
        'frr': frr_array[eer_idx],
        'eer': eer
    }

def compute_per(original_acc, unlearned_acc):
    if original_acc > 0:
        per = (original_acc - unlearned_acc) / original_acc * 100
    else:
        per = 0.0
    return max(0, min(per, 100))

def plot_metrics(models_metrics, metric_name, title, ylabel, save_path=None):
    models = list(models_metrics.keys())
    values = [models_metrics[model][metric_name] for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

os.makedirs('plots', exist_ok=True)

print("Loading and evaluating ResNet18 models (Classification)...")
resnet = models.resnet18(pretrained=False)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 100)
resnet = resnet.to(device)

resnet_metrics = {}
methods = ['Original', 'Custom', 'GradientAscent', 'SynapticDampening', 'FisherForgetting', 'NegativeGradient']
file_prefixes = ['ResNet18_speaker_classifier', 'unlearned_ResNet18', 'ga_unlearned_ResNet18', 'sd_unlearned_ResNet18', 'ff_unlearned_ResNet18', 'ng_unlearned_ResNet18']

for method, prefix in zip(methods, file_prefixes):
    checkpoint = torch.load(f"models/{prefix}_LibriSpeech.pth", map_location=device)
    resnet.load_state_dict(checkpoint['model_state_dict'])
    resnet_metrics[method] = compute_biometric_metrics(resnet, test_loader, device)
    if method != 'Original':
        resnet_metrics[method]['per'] = compute_per(resnet_metrics['Original']['forget_accuracy'], resnet_metrics[method]['forget_accuracy'])

print("\nResNet18 Classification Results:")
for method in methods:
    print(f"\n{method} Model:")
    print(f"  Forget Accuracy: {resnet_metrics[method]['forget_accuracy']:.2f}%")
    print(f"  FAR (Forget Class): {resnet_metrics[method]['far_forget']:.2f}%")
    print(f"  Retain Accuracy: {resnet_metrics[method]['retain_accuracy']:.2f}%")
    print(f"  FRR: {resnet_metrics[method]['frr']:.2f}%")
    print(f"  Information Leakage: {resnet_metrics[method]['info_leakage']:.2f}%")
    print(f"  EER: {resnet_metrics[method]['eer']:.2f}%")
    if method != 'Original':
        print(f"  Privacy Erasure Rate: {resnet_metrics[method]['per']:.2f}%")

plot_metrics({k: v for k, v in resnet_metrics.items() if k != 'Original'}, 'per', 'ResNet18 - Classification Privacy Erasure Rate', 'PER (%)', 'plots/resnet18_per.png')
plot_metrics(resnet_metrics, 'forget_accuracy', 'ResNet18 - Forget Accuracy', 'Accuracy (%)', 'plots/resnet18_forget_acc.png')
plot_metrics(resnet_metrics, 'retain_accuracy', 'ResNet18 - Retain Accuracy', 'Accuracy (%)', 'plots/resnet18_retain_acc.png')
plot_metrics(resnet_metrics, 'eer', 'ResNet18 - Classification EER', 'EER (%)', 'plots/resnet18_eer.png')

print("\nLoading and evaluating ViT-Tiny models (Classification)...")
vit = create_model('vit_tiny_patch16_224', pretrained=False, num_classes=100, in_chans=1)
vit = vit.to(device)

vit_metrics = {}
file_prefixes = ['ViT-Tiny_speaker_classifier', 'unlearned_ViT-Tiny', 'ga_unlearned_ViT-Tiny', 'sd_unlearned_ViT-Tiny', 'ff_unlearned_ViT-Tiny', 'ng_unlearned_ViT-Tiny']

for method, prefix in zip(methods, file_prefixes):
    checkpoint = torch.load(f"models/{prefix}_LibriSpeech.pth", map_location=device)
    vit.load_state_dict(checkpoint['model_state_dict'])
    vit_metrics[method] = compute_biometric_metrics(vit, test_loader, device)
    if method != 'Original':
        vit_metrics[method]['per'] = compute_per(vit_metrics['Original']['forget_accuracy'], vit_metrics[method]['forget_accuracy'])

print("\nViT-Tiny Classification Results:")
for method in methods:
    print(f"\n{method} Model:")
    print(f"  Forget Accuracy: {vit_metrics[method]['forget_accuracy']:.2f}%")
    print(f"  FAR (Forget Class): {vit_metrics[method]['far_forget']:.2f}%")
    print(f"  Retain Accuracy: {vit_metrics[method]['retain_accuracy']:.2f}%")
    print(f"  FRR: {vit_metrics[method]['frr']:.2f}%")
    print(f"  Information Leakage: {vit_metrics[method]['info_leakage']:.2f}%")
    print(f"  EER: {vit_metrics[method]['eer']:.2f}%")
    if method != 'Original':
        print(f"  Privacy Erasure Rate: {vit_metrics[method]['per']:.2f}%")

plot_metrics({k: v for k, v in vit_metrics.items() if k != 'Original'}, 'per', 'ViT-Tiny - Classification Privacy Erasure Rate', 'PER (%)', 'plots/vit_per.png')
plot_metrics(vit_metrics, 'forget_accuracy', 'ViT-Tiny - Forget Accuracy', 'Accuracy (%)', 'plots/vit_forget_acc.png')
plot_metrics(vit_metrics, 'retain_accuracy', 'ViT-Tiny - Retain Accuracy', 'Accuracy (%)', 'plots/vit_retain_acc.png')
plot_metrics(vit_metrics, 'eer', 'ViT-Tiny - Classification EER', 'EER (%)', 'plots/vit_eer.png')

print("\nEvaluating ResNet18 Identification...")
resnet_id_metrics = {}
file_prefixes = ['original_resnet', 'unlearned_resnet', 'ga_unlearned_resnet', 'sd_unlearned_resnet', 'ff_unlearned_resnet', 'ng_unlearned_resnet']

for method, prefix in zip(methods, file_prefixes):
    embeddings_dict = torch.load(f"models/{prefix}_embeddings.pth", map_location=device)
    resnet_id_metrics[method] = compute_identification_metrics(embeddings_dict, embeddings_dict['labels'], speaker_to_label, device)
    if method != 'Original':
        resnet_id_metrics[method]['per'] = compute_per(resnet_id_metrics['Original']['id_accuracy'], resnet_id_metrics[method]['id_accuracy'])

print("\nResNet18 Identification Results:")
for method in methods:
    print(f"\n{method} Model:")
    print(f"  Identification Accuracy (All): {resnet_id_metrics[method]['id_accuracy']:.2f}%")
    print(f"  Forget Class Accuracy: {resnet_id_metrics[method]['forget_accuracy']:.2f}%")
    print(f"  Retain Class Accuracy: {resnet_id_metrics[method]['retain_accuracy']:.2f}%")
    print(f"  FAR: {resnet_id_metrics[method]['far']:.2f}%")
    print(f"  FRR: {resnet_id_metrics[method]['frr']:.2f}%")
    print(f"  EER: {resnet_id_metrics[method]['eer']:.2f}%")
    if method != 'Original':
        print(f"  Privacy Erasure Rate (Overall): {resnet_id_metrics[method]['per']:.2f}%")


print("\nEvaluating ViT-Tiny Identification...")
vit_id_metrics = {}
file_prefixes = ['original_vit', 'unlearned_vit', 'ga_unlearned_vit', 'sd_unlearned_vit', 'ff_unlearned_vit', 'ng_unlearned_vit']

for method, prefix in zip(methods, file_prefixes):
    embeddings_dict = torch.load(f"models/{prefix}_embeddings.pth", map_location=device)
    vit_id_metrics[method] = compute_identification_metrics(embeddings_dict, embeddings_dict['labels'], speaker_to_label, device)
    if method != 'Original':
        vit_id_metrics[method]['per'] = compute_per(vit_id_metrics['Original']['id_accuracy'], vit_id_metrics[method]['id_accuracy'])

print("\nViT-Tiny Identification Results:")
for method in methods:
    print(f"\n{method} Model:")
    print(f"  Identification Accuracy (All): {vit_id_metrics[method]['id_accuracy']:.2f}%")
    print(f"  Forget Class Accuracy: {vit_id_metrics[method]['forget_accuracy']:.2f}%")
    print(f"  Retain Class Accuracy: {vit_id_metrics[method]['retain_accuracy']:.2f}%")
    print(f"  FAR: {vit_id_metrics[method]['far']:.2f}%")
    print(f"  FRR: {vit_id_metrics[method]['frr']:.2f}%")
    print(f"  EER: {vit_id_metrics[method]['eer']:.2f}%")
    if method != 'Original':
        print(f"  Privacy Erasure Rate (Overall): {vit_id_metrics[method]['per']:.2f}%")
