import os
import pandas as pd
import sklearn.metrics as metrics
import torch

from configs import Config
from dataloader.CIR_dataloader import raw_dataloader, entropy_dataloader
from models import modules, CIR_losses
from CIR_trainer import trainer

import warnings

warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '128'

configs = Config()
configs.train_epoch = 20  # 20, SIDGFD-HVAC: 5

print('Experiments with ResNet begin (subpopulation identification)...')
results = pd.DataFrame()
for task in configs.tasks:
    num = 'prob'
    print('Current task: ', task)

    train_dl, val_dl, test_dl = raw_dataloader(configs, task[0], task[1])
    all_data_dl = entropy_dataloader(configs, task[0])

    encoder = modules.ResNet1D(configs)
    classifier = modules.MLP2(32, 16, configs.classes)
    model_ResNet = modules.TSINet(encoder, classifier)

    optimizer = torch.optim.Adam(model_ResNet.parameters(), 1e-2)  # 1e-1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.5)
    loss_fn = CIR_losses.GeneralizedCrossEntropyLoss(0.7)  # 0.1, 0.3, 0.5, 0.7, 0.9
    # loss_fn = torch.nn.CrossEntropyLoss()

    trainer(model_ResNet, optimizer, scheduler, loss_fn, train_dl, train_dl, configs, task[1], num)

    checkpoint_last = torch.load(os.path.join(configs.exp_log_dir, f'ResNet_last_{task[1]}_{num}.pt'))
    print('The model obtained at the last epoch: ', checkpoint_last['epoch'])
    model_ResNet.load_state_dict(checkpoint_last['model_dict'])

    raw_data, raw_labels = torch.tensor([]), torch.tensor([])
    pred_labels, predicted_probs = torch.tensor([]), torch.tensor([])
    for batch_idx, (inputs, labels) in enumerate(all_data_dl):
        inputs, labels = inputs.float().to(configs.eval_device), labels.long().to(configs.eval_device)
        outputs = model_ResNet(inputs)

        _, predicted = torch.max(outputs.data, dim=1)

        raw_data = torch.cat((raw_data, inputs), dim=0)
        raw_labels = torch.cat((raw_labels, labels), dim=0)
        pred_labels = torch.cat((pred_labels, predicted), dim=0)

        predicted_probs = torch.cat((predicted_probs, torch.softmax(outputs, dim=1)), dim=0)

    print(metrics.confusion_matrix(raw_labels.cpu().numpy(), pred_labels.cpu().numpy()))

    data_dict = {
        'samples': raw_data,
        'labels': raw_labels.detach(),
        'probability': predicted_probs.detach()
    }
    torch.save(data_dict, os.path.join(configs.data_path, f"{configs.aim}_for_{task[1]}_DGFD.pt"))

    data_dict_test = {
            'samples': raw_data[::5],
            'labels': raw_labels.detach()[::5],
            'probability': predicted_probs.detach()[::5]
        }
    torch.save(data_dict_test, os.path.join(configs.data_path, f"{configs.aim}_for_{task[1]}_test_DGFD.pt"))

