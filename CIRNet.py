import os
import numpy as np
import pandas as pd
import itertools
import torch

from configs import Config
from dataloader.CIR_dataloader import dataloader
from models import modules
from CIR_trainer import CIRtrainer, evaluator

import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '128'

configs = Config()

weights = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
stds = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5] + [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# stds = [2.0]

results = pd.DataFrame()
print('Experiments with CIRNet begin...')
for task in configs.tasks:
    weight, std = weights[2], stds[0]
    for std, num in list(itertools.product(stds, range(configs.repeat_num))):  # configs.repeat_num
        num = num + 30
        # std = stds[8]
        # task, num = configs.tasks[0], f'CIRNet_{num}'
        print('Current task: ', task, 'Current weight: ', weight, '| Intervention std: ', std, '| Repeat number: ', num)
        train_dl, val_dl, test_dl = dataloader(configs, task[1], std=std)

        encoder = modules.ResNet1D(configs)
        classifier = modules.MLP2(32, 16, configs.classes)
        model_CIRNet = modules.CIRNet(encoder, classifier)

        optimizer = torch.optim.Adam(model_CIRNet.parameters(), 1e-2)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.5)

        CIRtrainer(model_CIRNet, train_dl, test_dl, optimizer, scheduler, configs, task[1], weight, num)

        print('Begin testing with CIRNet...')
        checkpoint_best_val = torch.load(os.path.join(configs.exp_log_dir, f'CIRNet_best_val_{task[1]}_{num}.pt'))
        print('The model with best ==val== is obtained at epoch: ', checkpoint_best_val['epoch'])
        encoder.load_state_dict(checkpoint_best_val['encoder_dict'])
        classifier.load_state_dict(checkpoint_best_val['classifier_dict'])
        model_best_val = modules.TSINet(encoder, classifier)
        accuracy, recall, precision, f1 = evaluator(model_best_val, test_dl, configs, False)
        result = {'Source domains': task[0], 'Target domain': task[1], 'Number': num, 'Weight': weight, 'Std': std,
                  'Accuracy': np.round(accuracy, 4), 'Precision': np.round(precision, 4), 'Recall': np.round(recall, 4),
                  'F1 Score': np.round(f1, 4), 'flag': 'best val', 'epoch': checkpoint_best_val['epoch']}
        print(result)
        results = pd.concat([results, pd.DataFrame([result])], axis=0)

        checkpoint_best_loss = torch.load(os.path.join(configs.exp_log_dir, f'CIRNet_best_loss_{task[1]}_{num}.pt'))
        print('The model with best ==loss== is obtained at epoch: ', checkpoint_best_loss['epoch'])
        encoder.load_state_dict(checkpoint_best_loss['encoder_dict'])
        classifier.load_state_dict(checkpoint_best_loss['classifier_dict'])
        model_best_loss = modules.TSINet(encoder, classifier)
        accuracy, recall, precision, f1 = evaluator(model_best_loss, test_dl, configs, False)
        result = {'Source domains': task[0], 'Target domain': task[1], 'Number': num, 'Weight': weight, 'Std': std,
                  'Accuracy': np.round(accuracy, 4), 'Precision': np.round(precision, 4), 'Recall': np.round(recall, 4),
                  'F1 Score': np.round(f1, 4), 'flag': 'best loss', 'epoch': checkpoint_best_loss['epoch']}
        print(result)
        results = pd.concat([results, pd.DataFrame([result])], axis=0)

        checkpoint_last = torch.load(os.path.join(configs.exp_log_dir, f'CIRNet_last_{task[1]}_{num}.pt'))
        print('The model obtained at the last epoch: ', checkpoint_last['epoch'])
        encoder.load_state_dict(checkpoint_last['encoder_dict'])
        classifier.load_state_dict(checkpoint_last['classifier_dict'])
        model_best_loss = modules.TSINet(encoder, classifier)
        accuracy, recall, precision, f1 = evaluator(model_best_loss, test_dl, configs, False)
        result = {'Source domains': task[0], 'Target domain': task[1], 'Number': num, 'Weight': weight, 'Std': std,
                  'Accuracy': np.round(accuracy, 4), 'Precision': np.round(precision, 4), 'Recall': np.round(recall, 4),
                  'F1 Score': np.round(f1, 4), 'flag': 'last', 'epoch': checkpoint_last['epoch']}
        print(result)
        results = pd.concat([results, pd.DataFrame([result])], axis=0)

        # Save results once per experiment to prevent run interruptions
        results.to_excel(f'results_{configs.aim}_CIRNet_1002.xlsx', index=False)

