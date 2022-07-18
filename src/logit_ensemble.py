import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_metric

from model import HateSpeechBERTModel, HateSpeechELECTRAModel
from utils import load_config, load_dataset

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")


def get_outputs(models, valid_dl):

    outputs = [[] for _ in range(len(models))]
    all_labels = []

    for data in valid_dl:

        inputs_list, masks_list, labels = data
        inputs_list = [inputs.to(device) for inputs in inputs_list]
        masks_list = [masks.to(device) for masks in masks_list]

        with torch.no_grad():
            for i, (model, inputs, masks) in enumerate(zip(models, inputs_list, masks_list)):
                output = model(inputs, masks)
                outputs[i].append(output)

        all_labels.append(labels)

    outputs = [torch.cat(outputs_, dim=0) for outputs_ in outputs]
    all_labels = torch.cat(all_labels, dim=0)

    return outputs, all_labels


def get_ensemble(outputs, labels, weights):
    final_outputs = torch.zeros_like(outputs[0])

    for i in range(len(outputs)):
        final_outputs += outputs[i] * weights[i]

    _, predictions = torch.max(final_outputs, dim=-1)
    _, targets = torch.max(labels, dim=-1)

    accuracy = accuracy_metric.compute(
        predictions=predictions, references=targets)['accuracy']
    f1 = f1_metric.compute(predictions=predictions,
                           references=targets, average='macro')['f1']
    precision = precision_metric.compute(
        predictions=predictions, references=targets, average='macro')['precision']
    recall = recall_metric.compute(
        predictions=predictions, references=targets, average='macro')['recall']

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def dfs_ensemble_weights(weights, outputs, labels, idx, num_models, max_scores):

    if idx + 1 == num_models:
        weights[idx] = 100 - sum(weights[:idx])
        lw = list(map(lambda x: x / 100, weights))
        scores = get_ensemble(outputs, all_labels, lw)
        if not max_scores or max_scores[0]['f1'] < scores['f1']:
            print(lw, scores)
            if not max_scores:
                max_scores.append(scores)
            else:
                max_scores[0] = scores
        return

    for r in range(0, 101 - sum(weights[:idx])):
        weights[idx] = r
        dfs_ensemble_weights(weights, outputs, labels,
                             idx + 1, num_models, max_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        required=True, help="yaml file path")
    parser.add_argument("--gpu", type=str, default="", help="gpu number")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config, default_config_dict={
        'batch_size': 32,
        'task': 'gender_bias',  # 'gender_bias' | 'bias' | 'hate_speech'
        'ensemble_models': [
            {'bert_model': 'electra', 'pretrain_model_name': 'beomi/beep-KcELECTRA-base-hate',
                'path': './model/best_model.pt'},
            {'bert_model': 'roberta', 'pretrain_model_name': 'klue/roberta-base',
                'path': './model/best_model2.pt'}
        ],
        'valid_path': './dev.gender_bias.binary.csv',
    })
    print(args, config)

    batch_size = config['batch_size']
    task = config['task']
    ensemble_models = config['ensemble_models']
    num_classes = 2 if task == 'gender_bias' else 3
    valid_path = config['valid_path']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizers = []
    models = []

    for ensemble_model in ensemble_models:
        bert_model = ensemble_model['bert_model']
        pretrain_model_name = ensemble_model['pretrain_model_name']
        model_path = ensemble_model['path']

        base_model = AutoModel.from_pretrained(pretrain_model_name)
        if bert_model == 'electra':
            model = HateSpeechELECTRAModel(base_model, num_classes)
        else:
            model = HateSpeechBERTModel(base_model, num_classes)
        model.to(device)

        tokenizers.append(AutoTokenizer.from_pretrained(pretrain_model_name))
        models.append(model)

    valid_dl = load_dataset(
        task, valid_path, num_classes, tokenizers, batch_size, aug=False, multi_tokenizer=True)

    outputs, all_labels = get_outputs(models, valid_dl)

    dfs_ensemble_weights([0 for _ in range(len(models))],
                         outputs, all_labels, 0, len(models), [])
