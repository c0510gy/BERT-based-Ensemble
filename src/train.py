import os
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from transformers import get_linear_schedule_with_warmup
from datasets import load_metric

from model import HateSpeechBERTModel, HateSpeechELECTRAModel
from utils import freeze_bert, load_config, load_dataset, empty_mem

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")


def compute_metrics(model, device, testdata_loader, criterion):
    all_labels, all_predictions = [], []

    model.eval()

    running_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, data in enumerate(testdata_loader):
            n_batches += 1

            empty_mem()

            inputs, masks, labels = data
            inputs, masks, labels = inputs.to(
                device), masks.to(device), labels.to(device)

            outputs = model(inputs, attention_mask=masks)

            loss = criterion(outputs, torch.max(labels, 1)[1])
            _, predicted = torch.max(outputs, dim=-1)
            _, target = torch.max(labels, dim=1)

            running_loss += loss.item()
            all_labels.extend(predicted.tolist())
            all_predictions.extend(target.tolist())

    scores = {
        'loss': running_loss / n_batches,
        'accuracy': accuracy_metric.compute(predictions=all_predictions, references=all_labels)['accuracy'],
        'f1': f1_metric.compute(predictions=all_predictions, references=all_labels, average='macro')['f1'],
        'precision': precision_metric.compute(predictions=all_predictions, references=all_labels, average='macro')['precision'],
        'recall': recall_metric.compute(predictions=all_predictions, references=all_labels, average='macro')['recall'],
    }

    return scores


def train(model, device, epoch, train_loader, valid_loader, criterion, best_model, checkpoint_path, compute_metrics_on_steps=False):

    running_loss = 0.0
    correct, tot = 0, 0
    n_batches = 0
    steps = 0

    all_labels, all_predictions = [], []

    for i, data in enumerate(train_loader):
        model.train()
        n_batches += 1

        inputs, masks, labels = data
        inputs, masks, labels = inputs.to(
            device), masks.to(device), labels.to(device)

        optimizer.zero_grad()
        empty_mem()

        outputs = model(inputs, attention_mask=masks)
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()
        scheduler.step()

        _, predicted = torch.max(outputs, dim=-1)
        _, target = torch.max(labels, dim=1)

        all_labels.extend(predicted.tolist())
        all_predictions.extend(target.tolist())

        running_loss += loss.item()
        correct += (predicted == target).sum().item()
        tot += target.size(0)

        steps += 1
        if compute_metrics_on_steps and steps % 100 == 0:
            valid_scores = compute_metrics(
                model, device, valid_loader, criterion)
            curr = {
                'steps': steps,
                'train_loss': running_loss / n_batches,
                'train_accuracy': correct / tot,
                'train_f1': f1_metric.compute(predictions=all_predictions, references=all_labels, average='macro')['f1'],
                'train_precision': precision_metric.compute(predictions=all_predictions, references=all_labels, average='macro')['precision'],
                'train_recall': recall_metric.compute(predictions=all_predictions, references=all_labels, average='macro')['recall'],
                'valid_loss': valid_scores['loss'],
                'valid_accuracy': valid_scores['accuracy'],
                'valid_f1': valid_scores['f1'],
                'valid_precision': valid_scores['precision'],
                'valid_recall': valid_scores['recall'],
            }

            print(epoch, i, '/', len(train_loader))
            print(curr)

            if best_model['metrics'] is None or best_model['metrics']['valid_f1'] < curr['valid_f1']:
                best_model['metrics'] = curr

                with open(os.path.join(checkpoint_path, f'best_metrics.json'), 'w') as f:
                    json.dump(curr, f)
                    f.close()

                torch.save(
                    model, os.path.join(checkpoint_path, f'best_model.pt'))
                torch.save(
                    model.state_dict(), os.path.join(checkpoint_path, f'best_state_dict.pt'))

            print('best_model', best_model)

    return {
        'train_loss': running_loss / n_batches,
        'train_accuracy': correct / tot,
        'train_f1': f1_metric.compute(predictions=all_predictions, references=all_labels, average='macro')['f1'],
        'train_precision': precision_metric.compute(predictions=all_predictions, references=all_labels, average='macro')['precision'],
        'train_recall': recall_metric.compute(predictions=all_predictions, references=all_labels, average='macro')['recall'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        required=True, help="yaml file path")
    parser.add_argument("--gpu", type=str, default="", help="gpu number")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config, default_config_dict={
        'epochs': 1,
        'batch_size': 32,
        'lr': 2e-5,
        'bert_model': 'electra',  # 'electra' | 'roberta'
        'pretrain_model_name': 'beomi/beep-KcELECTRA-base-hate',
        'freeze_layers': 0,
        'task': 'gender_bias',  # 'gender_bias' | 'bias' | 'hate_speech'
        'save_dir': './model/',
        'train_path': './train.gender_bias.binary.csv',
        'valid_path': './dev.gender_bias.binary.csv',
    })
    print(args, config)

    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = float(config['lr'])
    bert_model = config['bert_model']
    pretrain_model_name = config['pretrain_model_name']
    freeze_layers = config['freeze_layers']
    task = config['task']
    save_dir = config['save_dir']
    num_classes = 2 if task == 'gender_bias' else 3
    train_path = config['train_path']
    valid_path = config['valid_path']

    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_model = AutoModel.from_pretrained(pretrain_model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_name)

    if bert_model == 'electra':
        model = HateSpeechELECTRAModel(base_model, num_classes)
    else:
        model = HateSpeechBERTModel(base_model, num_classes)
    model.to(device)

    if freeze_layers > 0:
        freeze_bert(model, freeze_layers)

    train_dl = load_dataset(
        task, train_path, num_classes, tokenizer, batch_size)
    valid_dl = load_dataset(
        task, valid_path, num_classes, tokenizer, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_dl) * epochs)

    best_model = {
        'metrics': None,
    }

    compute_metrics_on_steps = True

    for epoch in range(epochs):
        train_scores = train(model, device, epoch, train_dl, valid_dl, criterion, best_model,
                             save_dir, compute_metrics_on_steps=compute_metrics_on_steps)

        if not compute_metrics_on_steps:
            valid_scores = compute_metrics(
                model, device, valid_dl, criterion)
            curr = {
                'epoch': epoch + 1,
                'train_loss': train_scores['train_loss'],
                'train_accuracy': train_scores['train_accuracy'],
                'train_f1': train_scores['train_f1'],
                'train_precision': train_scores['train_precision'],
                'train_recall': train_scores['train_recall'],
                'valid_loss': valid_scores['loss'],
                'valid_accuracy': valid_scores['accuracy'],
                'valid_f1': valid_scores['f1'],
                'valid_precision': valid_scores['precision'],
                'valid_recall': valid_scores['recall'],
            }
            print(curr)
