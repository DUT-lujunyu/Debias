import torch
import torch.optim as optim
from tqdm import tqdm

import time
import json
from dataset.dataset import to_tensor, convert_onehot
from model.Debias_Roberta import *
from utils import *

def train(config, train_iter, dev_iter, test_iter, test_iter_noi, test_iter_oi, test_iter_oni):

    model = RobertaForDebiasSequenceClassification.from_pretrained(config.model_name).to(config.device)
    model_name = '{}-NN_ML-{}_D-{}_B-{}_E-{}_Lr-{}'.format(config.model_name, config.pad_size, config.dropout, 
                                            config.batch_size, config.num_epochs, config.learning_rate)
    model_optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    max_score = 0

    for epoch in range(config.num_epochs):
        model.train()
        start_time = time.time()
        print("Model is training in epoch {}".format(epoch))
        loss_all = 0.
        preds = []
        labels = []
        step = 0

        for batch in tqdm(train_iter, desc='Training', colour = 'MAGENTA'):
            step += 1

            args = to_tensor(batch)
            outputs = model(config, if_bias=False, **args)
            loss, logits = outputs[:2]

            loss = loss.cpu()
            logits = logits.cpu()

            pred = get_preds(config, logits)  
            preds.extend(pred)
            label = args["label"]
            labels.extend(label.detach().numpy())

            loss_all += loss.item()
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

        # 验证
        trn_scores = get_scores(preds, labels, loss_all, len(train_iter), data_name="TRAIN")
        f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
        f.write(' ==================================================  Epoch: {}  ==================================================\n'.format(epoch))
        f.write('TrainScore: \n{}\n'.format(json.dumps(trn_scores))) 
        dev_scores, _ = eval(config, model, model_name, dev_iter, data_name='DEV')
        max_score = save_best(config, epoch, model_name, model, dev_scores, max_score)
        print("ALLTRAINED for {} epochs".format(epoch))


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
    loss_all = 0.
    preds = []
    labels = []
    model.eval()
    for batch in tqdm(dev_iter, desc='Evaling', colour = 'CYAN'):    
        with torch.no_grad():
            args = to_tensor(batch, if_adv)
            outputs = model(config, if_bias=False, **args)
            loss, logits = outputs[:2]

            loss = loss.cpu()
            logits = logits.cpu()
            
            pred = get_preds(config, logits)  
            preds.extend(pred)
            label = args['label']
            labels.extend(label.detach().numpy())

            loss_all += loss.item()
            
    scores = get_scores(preds, labels, loss_all, len(dev_iter), data_name=data_name)

    f = open('{}/{}.all_scores.txt'.format(config.result_path, model_name), 'a')
    f.write('{}: \n{}\n'.format(data_name, json.dumps(scores)))

    return scores, preds