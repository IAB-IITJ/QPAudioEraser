#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on Fri Jun 7 2024
"""

import time
import numpy as np
import pickle
import os
import random
from numba import cuda

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
from sklearn import metrics
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import TensorDataset
import os

def check_dir(rootdir):
    if not os.path.exists(rootdir):
        os.makedirs(rootdir)
from modelss import FineTuneCNN14, FineTuneMBNetV2, FineTuneResNet38

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

def seed_worker(worker_id):
    worker_seed = 0 
    np.random.seed(worker_seed)
    random.seed(worker_seed)

print(torch.cuda.is_available())
#%%
window_size = 0.032   # 0.032 for cnn14, 0.064 for others # in sec
hop_size = 0.01    # 0.01 for cnn14, 0.02 for others   # in sec
seg_t = 5   # in sec
batch_size = 64 
model_name = 'FineTuneCNN14'   # 'FineTuneCNN14' # 'FineTuneMBNetV2'   # 'FineTuneResNet38'
mel_bins = 64
fmin, fmax = 50, 8000
classes_num = 50   #50
learning_rate = 1e-3
num_epochs = 100
device = 'cuda'
    
global_acc, global_f1 = [], []


PATH_data = './data/'   # root dir of the resampled esc audio
#PATH_data = '/home/research-lowhigh/scripted_study_adl/'
for sr in [2000, 1000, 16000]:
    for fold in [1, 2 ,3, 4, 5]:
        X_train_feat, X_test_feat = [], []
        if os.path.exists(PATH_data + 'split_full_dataset/sr_%d/fold%s/seg_%ds_train_cnn14_inpt_resampled.pkl' % (sr, fold, seg_t)):
        #if os.path.exists(PATH_data + 'split/sr_%d/fold%s/seg_%ds_train_cnn14_inpt_resampled.pkl' % (sr, fold, seg_t)):
            with open(PATH_data + 'split_full_dataset/sr_%d/fold%s/seg_%ds_train_cnn14_inpt_resampled.pkl' % (sr, fold, seg_t), 'rb') as f:
                train_blob = pickle.load(f)
            with open(PATH_data + 'split_full_dataset/sr_%d/fold%s/seg_%ds_test_cnn14_inpt_resampled.pkl' % (sr, fold, seg_t), 'rb') as f:
                test_blob = pickle.load(f)
        else:
            raise RuntimeError("original training and test data not prepared. Please use esc_folds.py to generate them first.")
            
        X_train = train_blob[:, : -1]
        y_train = train_blob[:, -1]
        X_test = test_blob[:, : -1]
        y_test = test_blob[:, -1]
                            
        y_test = y_test.astype('int64')
        y_train = y_train.astype('int64')
            
        print('training and val size:', np.shape(X_train), np.shape(X_test), np.shape(y_train))
        assert len(np.unique(y_train)) == classes_num, "class # is not correct, please check!" 
        
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)
        np.random.seed(0)        
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        
        torch.cuda.empty_cache()
        
        save_model_path = './models/cnn14+fc/sr_%d_shuffle/seg_%ds/models_for_esc_50class/fold%s/' %(sr, seg_t, fold)
        #save_model_path = './models/cnn14+fc/sr_%d/seg_%ds/models_for_adl/fold%s/' %(sr, seg_t, fold)
        check_dir(save_model_path)
        
        Model = eval(model_name)
        
        ## CNN
        model = Model(sample_rate=16000, window_size=int(window_size*16000), hop_size=int(hop_size*16000), mel_bins=mel_bins, 
                      fmin=fmin, fmax=fmax, classes_num=classes_num)
        
        # Parallel
        # print('GPU number: {}'.format(torch.cuda.device_count()))
        # model = torch.nn.DataParallel(model)
        
        if 'cuda' in str(device):
            model.to(device)
        
        x_train_tensor = torch.from_numpy(np.array(X_train)).float()
        y_train_tensor = torch.from_numpy(np.array(y_train)).float()
        x_test_tensor = torch.from_numpy(np.array(X_test)).float()
        y_test_tensor = torch.from_numpy(np.array(y_test)).float()
        
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)
        
        
        train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                            batch_size=batch_size,
                            num_workers=0, pin_memory=True, shuffle = True, 
                            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0))
        test_loader = torch.utils.data.DataLoader(dataset=test_data, 
                                batch_size=batch_size,
                                num_workers=0, pin_memory=True, shuffle = False,
                                worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0))
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
        
        iteration = 0
        ### Training Loop ########
        f1_stop, recall_stop, stop_cnt = 0, 0, 0
        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            if stop_cnt == 10:
                break
            for i, d in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = d
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.int64)
        
                model.train()       
                outputs = model(inputs)['logits']
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
            print('[Epoch %d]' % (epoch))
            print('Train loss: {}'.format(loss))
            eval_output = []
            true_output = []
            test_output = []
            true_test_output = []
        
            with torch.no_grad():
                accuracy = metrics.balanced_accuracy_score(labels.cpu().numpy(), np.argmax(np.vstack(outputs.cpu().numpy()), axis = 1))
                print(accuracy)
                
                for x_val, y_val in test_loader:
                    x_val = torch.from_numpy(np.array(x_val)).float()
                    x_val = x_val.to(device)
                    y_val = y_val.to(device, dtype=torch.int64)
        
                    model.eval()
        
                    yhat = model(x_val)['logits']
                    test_loss = criterion(yhat, y_val)
        
                    test_output.append(yhat.data.cpu().numpy())
                    true_test_output.extend(y_val.data.cpu().numpy())
        
                test_oo = np.argmax(np.vstack(test_output), axis = 1)
                true_test_oo = np.asarray(true_test_output)
        
                accuracy = metrics.balanced_accuracy_score(true_test_oo, test_oo)
                precision, recall, fscore,_ = metrics.precision_recall_fscore_support(true_test_oo, test_oo, labels=np.unique(true_test_oo), average='macro')
                # early stopping
                #if f1_stop < fscore:
                if recall_stop < recall:
                    model_best = copy.deepcopy(model)
                    stop_cnt = 0
                    f1_stop = fscore
                    precision_stop, recall_stop, = precision, recall
                    confusion = confusion_matrix(true_test_oo, test_oo, normalize=None).reshape((1, -1))
                    iteration = epoch
                    try:
                        auc_test = metrics.roc_auc_score(np.vstack(true_test_output), np.vstack(test_output), average="macro")
                    except ValueError:
                        auc_test = None
                    print('Test loss: {}'.format(test_loss))
                    print('TEST average_precision: {}'.format(precision))
                    print('TEST average recall: {}'.format(recall))
                    print('TEST average f1: {}'.format(fscore))
            
                    trainLoss = {'Trainloss': loss}
                    testLoss = {'Testloss': test_loss}
                    test_f1 = {'test_f1':fscore}
                   
                else:
                    stop_cnt += 1
                    
        print('Finished Training')
        
        ### Save model ########
        save_model_path = save_model_path + '/%s_f1=%.4f_epoch%d.pth' % (model_name, f1_stop, iteration)
        torch.save(model_best.state_dict(), save_model_path)
        print('{}\t{}\t{}\t{}\t{}\t{}'.format('CNN14FC', fold, f1_stop, precision_stop, recall_stop, confusion), 
                  #file=open('./models/mobilenet+fc/sr_%d/seg_%ds/mobilenetFC_ESCdata_50class_result.logs' %(sr, seg_t),'a+')) 
                  file=open('./models/cnn14+fc/sr_%d_shuffle/seg_%ds/CNN14FC_ADLdata_5fold_result.logs' %(sr, seg_t),'a+'))

print('avg acc, f1:', np.mean(global_acc), np.mean(global_f1))
cuda.close()