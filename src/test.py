import os
import argparse

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from datasets import load_metric

from model import HateSpeechBERTModel, HateSpeechELECTRAModel
from utils import load_config, encode

accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")
precision_metric = load_metric("precision")
recall_metric = load_metric("recall")


def final_model(models, tokenizers, ensemble_weights, test_X, device, distribution_base_rules=None):

    test_predicted = []

    for i, x in enumerate(test_X):

        outputs = []

        for model, tokenizer in zip(models, tokenizers):
            encoded = encode(tokenizer, x)
            test_input, test_mask = torch.tensor([encoded['input_ids'].tolist()]), torch.tensor([
                encoded['attention_mask'].tolist()])
            with torch.no_grad():
                output = model(test_input.to(device), test_mask.to(device))
                outputs.append(output)

        outputs = sum(
            [output * r for output, r in zip(outputs, ensemble_weights)])
        _, predicted = torch.max(outputs, dim=-1)

        if distribution_base_rules is not None:
            for db_rul in distribution_base_rules:
                prediction_df, rules = db_rul
                ot_pred = prediction_df['label'][i].item()
                for rule in rules:
                    r_pred = rule['pred']
                    r_weight = torch.tensor(rule['weight']).to(device)

                    if r_pred[0] == predicted.item() and r_pred[1] == ot_pred:
                        outputs *= r_weight
                        _, predicted = torch.max(outputs, dim=-1)

        test_predicted.append(predicted.item())

    return test_predicted


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        required=True, help="yaml file path")
    parser.add_argument("--gpu", type=str, default="", help="gpu number")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_config(args.config, default_config_dict={
        'task': 'gender_bias',  # 'gender_bias' | 'bias' | 'hate_speech'
        'ensemble_models': [
            {'bert_model': 'electra', 'pretrain_model_name': 'beomi/beep-KcELECTRA-base-hate',
                'path': './model/best_model.pt'},
            {'bert_model': 'roberta', 'pretrain_model_name': 'klue/roberta-base',
                'path': './model/best_model2.pt'}
        ],
        'ensemble_weights': [0.5, 0.5],
        'distribution_base_ensemble': False,
        'distribution_base_rules': [{'prediction_path': './pred.csv', 'rules': [{'pred': [1, 0], 'weight': [0.3, 0.7]}]}],
        'test_path': './test.gender_bias.no_label.csv',
    })
    print(args, config)

    task = config['task']
    ensemble_models = config['ensemble_models']
    ensemble_weights = config['ensemble_weights']
    distribution_base_ensemble = config['distribution_base_ensemble']
    distribution_base_rules = config['distribution_base_rules']
    num_classes = 2 if task == 'gender_bias' else 3
    test_path = config['test_path']

    if distribution_base_ensemble:
        distribution_base_rules_ = []
        for db_rule in distribution_base_rules:
            prediction_path = db_rule['prediction_path']
            rules = db_rule['rules']

            prediction_df = pd.read_csv(prediction_path)
            distribution_base_rules_.append((prediction_df, rules))
        distribution_base_rules = distribution_base_rules_
    else:
        distribution_base_rules = None

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

    test_df = pd.read_csv(test_path)
    test_X = test_df['comments'].to_list()
    test_predicted = final_model(
        models, tokenizers, ensemble_weights, test_X, device, distribution_base_rules=distribution_base_rules)

    test_predicted_df = pd.DataFrame(
        data={'comments': test_X, 'label': test_predicted})

    test_predicted_df.to_csv('./predicted.csv', index=False)
