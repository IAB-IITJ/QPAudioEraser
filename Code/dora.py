

import time
import numpy as np
import pickle
import os
import random
import pandas as pd 
import librosa
import soundfile as sf
import copy
import glob # To find files/folders

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, classification_report

from sklearn import metrics
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchaudio
import torchaudio.transforms as T


from peft import get_peft_model, LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training


from modelss import FineTuneCNN14 

SEED = 42

ESC_DATA_ROOT = './data/' 

AUDIOMNIST_ROOT = './audio_mnist/' 
AUDIOMNIST_DATA_DIR = os.path.join(AUDIOMNIST_ROOT, 'data')

# Output paths
OUTPUT_DIR = './output_dora_audiomnist_cnn14/'
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
LOG_FILE = os.path.join(OUTPUT_DIR, 'training_results_audiomnist.log') # Changed log file name
ESC_EVAL_LOG_FILE = os.path.join(OUTPUT_DIR, 'esc50_evaluation_results.log')


TARGET_SR = 16000 
SEG_T_AUDIOMNIST = 1.5 


WINDOW_SIZE_SEC = 0.032
HOP_SIZE_SEC = 0.01
MEL_BINS = 64
FMIN = 50
FMAX = 8000 

N_FFT = int(WINDOW_SIZE_SEC * TARGET_SR)
HOP_LENGTH = int(HOP_SIZE_SEC * TARGET_SR)


MODEL_NAME = 'FineTuneCNN14'
AUDIOMNIST_CLASSES_NUM = 10 
ESC50_CLASSES_NUM = 50
BATCH_SIZE = 32 
LEARNING_RATE = 5e-5 
NUM_EPOCHS = 30 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EARLY_STOPPING_PATIENCE = 5 

DORA_R = 16
DORA_ALPHA = 32
DORA_DROPOUT = 0.1

DORA_TARGET_MODULES = ["attn.qkv", "attn.proj", "fc1", "fc2"] 


def check_dir(rootdir):
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AudioMNISTDataset(Dataset):
    def __init__(self, data_dir, target_sr, segment_len_samples,
                 n_fft, hop_length, n_mels, fmin, fmax,
                 speaker_ids_to_include, is_train=True,
                 transform=None, target_transform=None):

        self.data_dir = data_dir
        self.target_sr = target_sr
        self.segment_len_samples = segment_len_samples
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.is_train = is_train # Flag for random cropping vs center cropping

        self.transform = transform
        self.target_transform = target_transform

        self.file_list = []
        self.labels = []

        print(f"Loading AudioMNIST data for speakers: {speaker_ids_to_include}")
        for speaker_id in speaker_ids_to_include:
            speaker_path = os.path.join(data_dir, speaker_id)
            if not os.path.isdir(speaker_path):
                print(f"Warning: Speaker directory not found: {speaker_path}")
                continue
            # Find all .wav files for this speaker
            wav_files = glob.glob(os.path.join(speaker_path, '*.wav'))
            for fpath in wav_files:
                fname = os.path.basename(fpath)
                try:
                    # Extract label (digit) from filename like '0_01_0.wav'
                    label = int(fname.split('_')[0])
                    if 0 <= label <= 9:
                        self.file_list.append(fpath)
                        self.labels.append(label)
                    else:
                        print(f"Warning: Skipping file with invalid label in name: {fpath}")
                except (ValueError, IndexError):
                    print(f"Warning: Skipping file with unexpected name format: {fpath}")

        if not self.file_list:
             raise FileNotFoundError(f"No valid .wav files found in {data_dir} for speakers {speaker_ids_to_include}")

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)

        print(f"Loaded {len(self.file_list)} files for {'train' if is_train else 'test'}.")

    def __len__(self):
        return len(self.file_list)

    def _load_and_process(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

        if sr != self.target_sr:
            resampler = T.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

      
        current_len = waveform.shape[1]
        if current_len < self.segment_len_samples:
           
            ratio = self.segment_len_samples / current_len
            waveform = waveform.repeat(1, int(np.ceil(ratio)))
            waveform = waveform[:, :self.segment_len_samples]
           
        elif current_len > self.segment_len_samples:
            if self.is_train:
                 start = random.randint(0, current_len - self.segment_len_samples)
            else: 
                 start = (current_len - self.segment_len_samples) // 2
            waveform = waveform[:, start:start + self.segment_len_samples]

    
        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

  
        log_mel_spec = log_mel_spec.squeeze(0).transpose(0, 1)

        return log_mel_spec

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]

        log_mel_spec = self._load_and_process(audio_path)

        if log_mel_spec is None: # Handle loading errors
            print(f"Warning: Using data from first sample due to loading error for {audio_path}")
           
            audio_path0 = self.file_list[0]
            log_mel_spec = self._load_and_process(audio_path0)
            label = self.labels[0]
            if log_mel_spec is None:
                raise IOError(f"Failed to load even the first sample ({audio_path0})")

        if self.transform:
             log_mel_spec = self.transform(log_mel_spec)
        if self.target_transform:
            label = self.target_transform(label)

        return log_mel_spec.float(), torch.tensor(label, dtype=torch.long)


def evaluate_model(model, loader, criterion, device, num_classes, dataset_name="Test", class_names=None):
   
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    # 
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Evaluating on {dataset_name}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            valid_indices = labels >= 0 
            if not valid_indices.all():
                 inputs = inputs[valid_indices]
                 labels = labels[valid_indices]
                 if inputs.shape[0] == 0: continue

            outputs = model(inputs)
            logits = outputs['logits']
            
            if logits.shape[1] != num_classes:
                 print(f"Warning: Logits dimension ({logits.shape[1]}) doesn't match num_classes ({num_classes}). Skipping batch evaluation.")
        
                 continue
                 
            loss = criterion(logits, labels)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels:
        print(f"No valid labels found for evaluation on {dataset_name}. Cannot compute metrics.")

        return { 'loss': 0, 'accuracy': 0, 'precision_macro': 0, 'recall_macro': 0,
                 'f1_macro': 0, 'f1_weighted': 0, 'confusion_matrix': np.array([]),
                 'labels': [], 'preds': [] }

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0, labels=np.arange(num_classes) 
    )
    fscore_weighted, _, _, _ = metrics.precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0, labels=np.arange(num_classes)
    )

    print(f'{dataset_name} Results:')
    print(f'Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Macro Precision: {precision:.4f}')
    print(f'Macro Recall: {recall:.4f}')
    print(f'Macro F1-score: {fscore:.4f}')
    print(f'Weighted F1-score: {fscore_weighted:.4f}')

    if len(all_preds) > 0:

        target_names_print = [str(i) for i in range(num_classes)]
        if class_names and len(class_names) == num_classes:
            target_names_print = class_names

      
        unique_labels_present = np.unique(all_labels + all_preds)
        labels_for_report = [l for l in np.arange(num_classes) if l in unique_labels_present]
        target_names_for_report = [target_names_print[l] for l in labels_for_report]

        if labels_for_report: 
            print(classification_report(all_labels, all_preds, labels=labels_for_report, target_names=target_names_for_report, zero_division=0))
        else:
            print("Classification report skipped (no common labels found or labels outside expected range).")

       
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    else:
        conf_matrix = np.array([])


    results = {
        'loss': avg_loss, 'accuracy': accuracy, 'precision_macro': precision, 'recall_macro': recall,
        'f1_macro': fscore, 'f1_weighted': fscore_weighted, 'confusion_matrix': conf_matrix,
        'labels': all_labels, 'preds': all_preds,
    }
    return results


if __name__ == "__main__":
    seed_everything(SEED)
    check_dir(OUTPUT_DIR)
    check_dir(MODEL_SAVE_DIR)

 
    print("--- Preparing AudioMNIST Dataset ---")

   
    print(f"Checking AudioMNIST data dir: {os.path.abspath(AUDIOMNIST_DATA_DIR)}")
    if not os.path.isdir(AUDIOMNIST_DATA_DIR):
        print(f"ERROR: AudioMNIST data directory not found at '{AUDIOMNIST_DATA_DIR}'!");
        print(f"Ensure the dataset is in '{AUDIOMNIST_ROOT}' and contains the 'data' subdirectory.")
        import sys
        sys.exit(1)
   

    # Get speaker IDs (folder names in data_dir)
    all_speaker_ids = sorted([d for d in os.listdir(AUDIOMNIST_DATA_DIR) if os.path.isdir(os.path.join(AUDIOMNIST_DATA_DIR, d))])
    if not all_speaker_ids:
         print(f"ERROR: No speaker directories found in {AUDIOMNIST_DATA_DIR}")
         sys.exit(1)

    print(f"Found {len(all_speaker_ids)} speakers: {all_speaker_ids}")

   
    try:
        speaker_ints = [int(sid) for sid in all_speaker_ids]
        
        train_speaker_ints = [sid for sid in speaker_ints if sid <= 50]
        test_speaker_ints = [sid for sid in speaker_ints if sid > 50]
    
        train_speaker_ids = [f"{sid:02d}" for sid in train_speaker_ints]
        test_speaker_ids = [f"{sid:02d}" for sid in test_speaker_ints]

  
        if not train_speaker_ids or not test_speaker_ids:
             print("Warning: Standard speaker split (<=50/>50) resulted in empty set. Using random 80/20 split.")
             random.shuffle(all_speaker_ids)
             split_idx = int(0.8 * len(all_speaker_ids))
             train_speaker_ids = all_speaker_ids[:split_idx]
             test_speaker_ids = all_speaker_ids[split_idx:]

    except ValueError:
         print("Warning: Speaker IDs are not simple integers. Using random 80/20 split.")
         random.shuffle(all_speaker_ids)
         split_idx = int(0.8 * len(all_speaker_ids))
         train_speaker_ids = all_speaker_ids[:split_idx]
         test_speaker_ids = all_speaker_ids[split_idx:]

    if not train_speaker_ids or not test_speaker_ids:
         print("ERROR: Could not create non-empty train/test speaker split.")
         sys.exit(1)

    print(f"Using {len(train_speaker_ids)} speakers for training: {train_speaker_ids}")
    print(f"Using {len(test_speaker_ids)} speakers for testing: {test_speaker_ids}")


    segment_len_samples = int(SEG_T_AUDIOMNIST * TARGET_SR)

    train_dataset = AudioMNISTDataset(
        data_dir=AUDIOMNIST_DATA_DIR,
        target_sr=TARGET_SR,
        segment_len_samples=segment_len_samples,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=MEL_BINS, fmin=FMIN, fmax=FMAX,
        speaker_ids_to_include=train_speaker_ids, is_train=True
    )
    test_dataset = AudioMNISTDataset(
        data_dir=AUDIOMNIST_DATA_DIR,
        target_sr=TARGET_SR,
        segment_len_samples=segment_len_samples, # Use same length for test
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=MEL_BINS, fmin=FMIN, fmax=FMAX,
        speaker_ids_to_include=test_speaker_ids, is_train=False # Set is_train=False
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
        pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    print("--- AudioMNIST DataLoaders Ready ---")
    audiomnist_class_names = [str(i) for i in range(AUDIOMNIST_CLASSES_NUM)] # ['0', '1', ..., '9']


   
    print("--- Loading Base Model and Applying DoRA ---")
    # Load the base model - classes_num matches AudioMNIST
    base_model = FineTuneCNN14(
        sample_rate=TARGET_SR, window_size=N_FFT, hop_size=HOP_LENGTH, mel_bins=MEL_BINS,
        fmin=FMIN, fmax=FMAX, classes_num=AUDIOMNIST_CLASSES_NUM # Use 10 classes
    )

 
    print("Model Architecture (Top Level):")
   

    
    peft_config = LoraConfig(
     
        task_type=TaskType.SEQ_CLS,               
        r=DORA_R,
        lora_alpha=DORA_ALPHA,
        lora_dropout=DORA_DROPOUT,
        target_modules=DORA_TARGET_MODULES, 
        use_dora=True,
        bias="none",
    )

  
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.to(DEVICE)
    print("--- Model with DoRA Ready ---")

  
    print("--- Starting Training on AudioMNIST ---")
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True) # Shorter patience?

    best_f1 = -1.0
    epochs_no_improve = 0
    best_model_path = None 
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
      
        for inputs, labels in train_loop:
             inputs = inputs.to(DEVICE)
             labels = labels.to(DEVICE)
             optimizer.zero_grad()
             outputs = model(inputs)
             logits = outputs['logits']
             loss = criterion(logits, labels)
             loss.backward()
             optimizer.step()
             train_loss += loss.item() * inputs.size(0)
             train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        val_results = evaluate_model(model, test_loader, criterion, DEVICE, AUDIOMNIST_CLASSES_NUM,
                                     dataset_name="AudioMNIST Test", class_names=audiomnist_class_names)
        val_f1 = val_results['f1_macro']

       
        with open(LOG_FILE, 'a') as f:
             f.write(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_results['loss']:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_results['accuracy']:.4f}\n")

       
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            save_path = os.path.join(MODEL_SAVE_DIR, f"best_audiomnist_dora_f1_{best_f1:.4f}")
            model.save_pretrained(save_path)
            print(f"New best model saved to {save_path} with F1: {best_f1:.4f}")
            best_model_path = save_path # Store the path
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs. Best F1: {best_f1:.4f}")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

        scheduler.step(val_f1)

    print("--- Finished Training on AudioMNIST ---")

    
    print("--- Final Evaluation on AudioMNIST Test Set (Best Model) ---")
    if best_model_path and os.path.exists(best_model_path):
         print(f"Loading best PEFT model from {best_model_path}")
     
         base_model_for_eval = FineTuneCNN14(
             sample_rate=TARGET_SR, window_size=N_FFT, hop_size=HOP_LENGTH, mel_bins=MEL_BINS,
             fmin=FMIN, fmax=FMAX, classes_num=AUDIOMNIST_CLASSES_NUM
         )
       
         final_model = PeftModel.from_pretrained(base_model_for_eval, best_model_path)
        
         final_model.to(DEVICE)
         final_audiomnist_results = evaluate_model(final_model, test_loader, criterion, DEVICE, AUDIOMNIST_CLASSES_NUM,
                                                 dataset_name="AudioMNIST Final Test", class_names=audiomnist_class_names)
         with open(LOG_FILE, 'a') as f:
             f.write("\n--- Final AudioMNIST Test Results (Best Model) ---\n")
             for key, value in final_audiomnist_results.items():
                 if key not in ['confusion_matrix', 'labels', 'preds']:
                     f.write(f"{key}: {value}\n")
             f.write(f"Confusion Matrix:\n{final_audiomnist_results['confusion_matrix']}\n")
    else:
         print("No best model state was saved or found. Cannot perform final AudioMNIST evaluation.")
         best_model_path = None # Ensure defined for next step


  
    print("\n--- Evaluating DoRA-adapted Model on ESC-50 Test Sets ---")
    if best_model_path is None or not os.path.exists(best_model_path):
         print("Skipping ESC-50 evaluation as no best DoRA model was trained/saved from AudioMNIST.")
    else:
   
        print(f"Loading DoRA adapters from AudioMNIST training ({best_model_path}) onto ESC-50 configured base model...")

      
        esc_base_model = FineTuneCNN14(
            sample_rate=TARGET_SR, window_size=N_FFT, hop_size=HOP_LENGTH, mel_bins=MEL_BINS,
            fmin=FMIN, fmax=FMAX, classes_num=ESC50_CLASSES_NUM # Correct number of classes
        )

        try:
            
            esc_eval_model = PeftModel.from_pretrained(esc_base_model, best_model_path, is_trainable=False)
           
            esc_eval_model.to(DEVICE)
            esc_eval_model.eval()
            print("Successfully loaded DoRA adapters onto ESC-50 model.")

           
            esc_criterion = nn.CrossEntropyLoss()
            all_esc_f1_macros = []
           
            for fold in [1, 2, 3, 4, 5]:
                print(f"\n--- Evaluating on ESC-50 Fold {fold} ---")
                esc_test_pkl_path = os.path.join(ESC_DATA_ROOT, f'split_full_dataset/sr_{TARGET_SR}/fold{fold}/seg_{SEG_T_ESC}s_test_cnn14_inpt_resampled.pkl')
               
                if not os.path.exists(esc_test_pkl_path):
                    print(f"Warning: ESC-50 test file not found: {esc_test_pkl_path}. Skipping fold {fold}.")
                    continue

                with open(esc_test_pkl_path, 'rb') as f:
                    test_blob = pickle.load(f)
                X_test_esc = test_blob[:, :-1]; y_test_esc = test_blob[:, -1].astype('int64')

          
                esc_expected_time_steps = int(SEG_T_ESC * TARGET_SR) // HOP_LENGTH + 1
                print(f"ESC data shape raw: {X_test_esc.shape}, Expected time steps: {esc_expected_time_steps}, Mel bins: {MEL_BINS}")
                if len(X_test_esc.shape) == 2 and X_test_esc.shape[1] == esc_expected_time_steps * MEL_BINS:
                     print("Reshaping ESC data from flat to (Time, Freq)")
                     X_test_esc = X_test_esc.reshape(X_test_esc.shape[0], esc_expected_time_steps, MEL_BINS)
                elif len(X_test_esc.shape) != 3 or X_test_esc.shape[1] < esc_expected_time_steps - 5 or X_test_esc.shape[1] > esc_expected_time_steps + 5 or X_test_esc.shape[2] != MEL_BINS: # Allow slight time diff
                     print(f"Warning: ESC data shape {X_test_esc.shape} mismatch vs expected ({esc_expected_time_steps}, {MEL_BINS}). Check PKL format/padding.")
                 
                   
                     # if X_test_esc.shape[1] < esc_expected_time_steps:
                     #      pad_t = esc_expected_time_steps - X_test_esc.shape[1]
                     #      X_test_esc = np.pad(X_test_esc, ((0,0), (0,pad_t), (0,0)), mode='constant')
                     # elif X_test_esc.shape[1] > esc_expected_time_steps:
                     #      X_test_esc = X_test_esc[:, :esc_expected_time_steps, :]


                x_test_tensor = torch.from_numpy(X_test_esc).float()
                y_test_tensor = torch.from_numpy(y_test_esc).long()

                esc_test_data = TensorDataset(x_test_tensor, y_test_tensor)
                esc_test_loader = DataLoader(esc_test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

                esc_results = evaluate_model(esc_eval_model, esc_test_loader, esc_criterion, DEVICE, ESC50_CLASSES_NUM, dataset_name=f"ESC-50 Fold {fold}") # Add class_names=esc50_class_names if defined
                all_esc_f1_macros.append(esc_results['f1_macro'])

                # Log ESC-50 results (same as before)
                with open(ESC_EVAL_LOG_FILE, 'a') as f:
                    f.write(f"\n--- ESC-50 Fold {fold} Results ---\n")
                
                    for key, value in esc_results.items():
                        if key not in ['confusion_matrix', 'labels', 'preds']:
                            f.write(f"{key}: {value}\n")
                    f.write(f"Confusion Matrix:\n{esc_results['confusion_matrix']}\n")


            avg_esc_f1 = np.mean(all_esc_f1_macros) if all_esc_f1_macros else 0
            print(f"\nAverage Macro F1-score across ESC-50 folds: {avg_esc_f1:.4f}")
            with open(ESC_EVAL_LOG_FILE, 'a') as f:
                f.write(f"\nAverage Macro F1-score across ESC-50 folds: {avg_esc_f1:.4f}\n")

        except Exception as e:
            print(f"Error during ESC-50 evaluation setup or execution: {e}")
            import traceback
            traceback.print_exc()

    print("--- Script Finished ---")