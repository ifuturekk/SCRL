import os
import time
from time import strftime, gmtime
from collections import Counter
from tqdm import tqdm
import sklearn.metrics as metrics
import torch

from models import CIR_losses
from models.losses import FocalLoss
from utils import domain_label_reset, feature_masking


def CIRtrainer(model, train_dl, val_dl, optimizer, scheduler, configs, target, weight, num):
    cf_loss_fn = torch.nn.CrossEntropyLoss()
    ct_loss_fn = torch.nn.TripletMarginLoss(margin=1e-10)  # margin=0.01, 1e-3, 1e-10
    start = time.time()
    best_f1, best_loss = 0, 1e2
    for epoch in tqdm(range(configs.train_epoch)):
        model.to(configs.train_device)
        model.train()
        total_loss = []
        total_cf_loss, total_fac_loss, total_ct_loss = [], [], []
        for batch_idx, (inputs, aug_inputs, pos, neg, labels) in enumerate(train_dl):
            inputs, labels = inputs.float().to(configs.train_device), labels.long().to(configs.train_device)
            aug_inputs = aug_inputs.float().to(configs.train_device)
            pos, neg = pos.float().to(configs.train_device), neg.float().to(configs.train_device)

            optimizer.zero_grad()
            classes, features = model(inputs)
            aug_classes, aug_features = model(aug_inputs)
            _, pos_features = model(pos)
            _, neg_features = model(neg)

            cf_loss = cf_loss_fn(classes, labels) + cf_loss_fn(aug_classes, labels)

            masked_features, masked_aug_features = feature_masking(features, aug_features)
            fac_loss = CIR_losses.factorization_loss(masked_features, masked_aug_features)

            triplets = [features, pos_features, neg_features]
            ct_loss = ct_loss_fn(*triplets)

            loss = cf_loss + weight[0] * fac_loss + weight[1] * ct_loss  # fac_loss  # + ct_loss

            loss.backward()
            optimizer.step()

            total_cf_loss.append(cf_loss.item())
            total_fac_loss.append(fac_loss.item())
            total_ct_loss.append(ct_loss.item())
            total_loss.append(loss.item())

        scheduler.step()
        ave_cf_loss = torch.tensor(total_cf_loss).mean()
        ave_fac_loss = torch.tensor(total_fac_loss).mean()
        ave_ct_loss = torch.tensor(total_ct_loss).mean()
        ave_loss = torch.tensor(total_loss).mean()
        accuracy, recall, precision, f1 = evaluator(model, val_dl, configs, True)
        if f1 > best_f1:
            print('F1 score has been updated!')
            print(f'Accuracy: {accuracy: 2.5f}\t', f'Recall: {recall: 2.5f}\t',
                  f'Precision: {precision: 2.5f}\t', f'F1 score: {f1: 2.5f}\t')
            best_f1 = f1
            checkpoint_val = {'epoch': epoch,
                              'encoder_dict': model.encoder.state_dict(),
                              'classifier_dict': model.classifier.state_dict()}
        if ave_cf_loss < best_loss:
            print('Best loss has been updated!')
            print(f'Accuracy: {accuracy: 2.5f}\t', f'Recall: {recall: 2.5f}\t',
                  f'Precision: {precision: 2.5f}\t', f'F1 score: {f1: 2.5f}\t')
            best_loss = ave_cf_loss
            checkpoint_loss = {'epoch': epoch,
                               'encoder_dict': model.encoder.state_dict(),
                               'classifier_dict': model.classifier.state_dict()}

        if epoch % 1 == 0:
            print(f'Epoch: {epoch}\t|\tTotal: {ave_loss: 2.6f}\t|\tcf: {ave_cf_loss: 2.6f}'
                  f'\t|\tfac: {ave_fac_loss: 2.6f}\t|\tct: {ave_ct_loss: 2.6f}')

    checkpoint_last = {'epoch': epoch,
                       'encoder_dict': model.encoder.state_dict(),
                       'classifier_dict': model.classifier.state_dict()}

    os.makedirs(os.path.join(configs.exp_log_dir), exist_ok=True)
    torch.save(checkpoint_val, os.path.join(configs.exp_log_dir, f'CIRNet_best_val_{target}_{num}.pt'))
    torch.save(checkpoint_loss, os.path.join(configs.exp_log_dir, f'CIRNet_best_loss_{target}_{num}.pt'))
    torch.save(checkpoint_last, os.path.join(configs.exp_log_dir, f'CIRNet_last_{target}_{num}.pt'))

    end = time.time()
    total_time = end - start
    print('Total training time: {}'.format(strftime('%H:%M:%S', gmtime(total_time))))


def evaluator(model, eval_dl, configs, mo=False):
    model.to(configs.eval_device)
    model.eval()
    with torch.no_grad():
        true_labels, pred_labels = torch.tensor([]), torch.tensor([])
        for batch_idx, (inputs, labels) in enumerate(eval_dl):
            inputs, labels = inputs.float().to(configs.eval_device), labels.long().to(configs.eval_device)
            if mo:
                outputs = model(inputs)[0]
            else:
                outputs = model(inputs)

            _, predicted = torch.max(outputs.data, dim=1)

            true_labels = torch.cat((true_labels, labels), dim=0)
            pred_labels = torch.cat((pred_labels, predicted), dim=0)

        true_labels, pred_labels = true_labels.cpu().numpy(), pred_labels.cpu().numpy()
        cm = metrics.confusion_matrix(true_labels, pred_labels)
        print(cm)
        val_accuracy = metrics.accuracy_score(true_labels, pred_labels)
        val_recall = metrics.recall_score(true_labels, pred_labels, average='macro')
        val_precision = metrics.precision_score(true_labels, pred_labels, average='macro')
        val_f1 = metrics.f1_score(true_labels, pred_labels, average='macro')

    return val_accuracy, val_recall, val_precision, val_f1




