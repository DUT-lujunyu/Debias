import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import time
import json
import copy
from dataset.dataset import to_tensor, convert_onehot
from model.CF import BERT_CFModel
from utils import * 

def train(config, train_iter, dev_iter, test_iter, test_iter_noi, test_iter_oi, test_iter_oni):

    model = BERT_CFModel(config).to(config.device)

    emb_params = [
        {"params": [param for name, param in model.named_parameters() if "entity_mlp" in name]}
    ]
    other_params = [
        {"params": [param for name, param in model.named_parameters() if "entity_mlp" not in name]}
    ]

    model_name = '{}_MLP_emb_back_att_hm_sep_ML-{}_D-{}_B-{}_E-{}_Lr-{}'.format(config.model_name, config.pad_size, config.dropout, 
                                            config.batch_size, config.num_epochs, config.learning_rate)
    emb_optimizer = optim.AdamW(emb_params, weight_decay=1e-2, lr=config.learning_rate)
    model_optimizer = optim.AdamW(other_params, lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    max_score = 0

    for epoch in range(config.num_epochs):
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))
        loss_all = 0.
        all_preds = []
        k_preds = []
        w_preds = []
        s_preds = []
        cf_preds = []
        labels = []
        step = 0

        for batch in tqdm(train_iter, desc='Training', colour = 'MAGENTA'):
            step += 1

            args = to_tensor(batch)
            out = model(**args)
            pred_logit = out["logits_all"].cpu()
            ensemble_logit = out["logits_k"].cpu()
            entity_pred_logit = out["logits_w"].cpu()
            text_pred_logit = out["logits_s"].cpu()
            cf_pred_logit = out["logits_cf"].cpu()
            label = args["label"]

            pred = get_preds(config, pred_logit)  
            all_preds.extend(pred) 
            k_pred = get_preds(config, ensemble_logit)  
            k_preds.extend(k_pred) 
            w_pred = get_preds(config, entity_pred_logit)  
            w_preds.extend(w_pred) 
            s_pred = get_preds(config, text_pred_logit)  
            s_preds.extend(s_pred) 
            cf_pred = get_preds(config, cf_pred_logit)  
            cf_preds.extend(cf_pred) 

            one_hot_label = args["label"]
            labels.extend(one_hot_label.detach().numpy())

            emb_loss = loss_fn(entity_pred_logit, label).cpu()
            loss = loss_fn(pred_logit, label).cpu() + loss_fn(text_pred_logit, label).cpu()
            loss_all += loss.item()

            emb_optimizer.zero_grad()
            emb_loss.backward(retain_graph=True)
            emb_optimizer.step()

            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            if step % 1000 == 0:
                f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
                f.write(' ==================================================  Epoch: {} Step: {}  ==================================================\n'.format(epoch, step))
                dev_scores, _ = eval(config, model, model_name, dev_iter, data_name='DEV')
                max_score = save_best(config, epoch, model_name, model, dev_scores, max_score)
                test(config, model, model_name, test_iter, test_iter_noi, test_iter_oi, test_iter_oni)

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time)/60.))
        print("TRAINED for {} epochs".format(epoch))

        end_time = time.time()
        print(" took: {:.1f} min".format((end_time - start_time)/60.))
        print("TRAINED for {} epochs".format(epoch))

        # 验证
        trn_scores = cf_get_scores(all_preds, k_preds, w_preds, s_preds, cf_preds, labels, loss_all, len(train_iter), data_name="TRAIN")
        
        f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
        f.write(' ==================================================  Epoch: {}  ==================================================\n'.format(epoch))
        f.write('TrainScore: \n{}\n'.format(json.dumps(trn_scores))) 
        dev_scores, _ = eval(config, model, model_name, dev_iter, data_name='DEV')
        max_score = save_best(config, epoch, model_name, model, dev_scores, max_score)
        print("ALLTRAINED for {} epochs".format(epoch))

    test(config, model, model_name, test_iter, test_iter_noi, test_iter_oi, test_iter_oni)


def test(config, model, model_name, test_iter, test_iter_noi, test_iter_oi, test_iter_oni):
    path = '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_scores, _ = eval(config, model, model_name, test_iter, data_name='TEST')  # 所有包含 dirty word 的样本
    test_scores_noi, _ = eval(config, model, model_name, test_iter_noi, data_name='NOI_TEST')
    test_scores_oi, _ = eval(config, model, model_name, test_iter_oi, data_name='OI_TEST')
    test_scores_oni, _ = eval(config, model, model_name, test_iter_oni, data_name='ONI_TEST')

    f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
    f.write('Test: \n{}\n'.format(json.dumps(test_scores)))
    f.write('Test_NOI: \n{}\n'.format(json.dumps(test_scores_noi)))
    f.write('Test_OI: \n{}\n'.format(json.dumps(test_scores_oi)))
    f.write('Test_ONI: \n{}\n'.format(json.dumps(test_scores_oni)))


def eval(config, model, model_name, dev_iter, data_name='DEV', if_adv=False):
    loss_fn = nn.CrossEntropyLoss()
    loss_all = 0.
    all_preds = []
    k_preds = []
    w_preds = []
    s_preds = []
    cf_preds = []

    labels = []
    model.eval()
    for batch in tqdm(dev_iter, desc='Evaling', colour = 'CYAN'):   
        with torch.no_grad():
            args = to_tensor(batch, if_adv)
            out = model(**args)

            cf_pred_logit = out["logits_cf"].cpu()
            label = args["label"]
            loss = loss_fn(cf_pred_logit, label).cpu()

            pred = get_preds(config, out["logits_all"].cpu())  
            all_preds.extend(pred) 
            k_pred = get_preds(config, out["logits_k"].cpu())  
            k_preds.extend(k_pred) 
            w_pred = get_preds(config, out["logits_w"].cpu())  
            w_preds.extend(w_pred) 
            s_pred = get_preds(config, out["logits_s"].cpu())  
            s_preds.extend(s_pred) 
            cf_pred = get_preds(config, out["logits_cf"].cpu())  
            cf_preds.extend(cf_pred) 

            labels.extend(label.detach().numpy())

            loss_all += loss.item()
    
    scores = cf_get_scores(all_preds, k_preds, w_preds, s_preds, cf_preds, labels, loss_all, len(dev_iter), data_name=data_name)
    f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
    f.write('{}: \n{}\n'.format(data_name, json.dumps(scores)))

    return scores, cf_preds



