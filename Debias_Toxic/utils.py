import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, confusion_matrix

# For Multi Classfication
def get_preds(config, logit):
    results = torch.max(logit.data, 1)[1].cpu().numpy()
    # new_results = []
    # for result in results:
    #     result = convert_onehot(config, result)
    #     new_results.append(result)
    # return new_results
    return results

def get_fpr(labels, preds):
    matrix = confusion_matrix(labels, preds)
    print(matrix)
    _tp = matrix[1, 1]
    _fn = matrix[1, 0]
    _fp = matrix[0, 1]
    _tn = matrix[0, 0]
    _tpr = _tp / (_tp + _fn)
    _fpr = _fp / (_tn + _fp)
    return _fpr

def get_scores(all_preds, all_lebels, loss_all, len, data_name):

    score_dict = dict()
    # fpr, _, _ = roc_curve(all_lebels, all_preds, pos_label=0)
    # fpr = get_fpr(all_lebels, all_preds)

    score_dict['F1'] = f1_score(all_lebels, all_preds)
    score_dict['weighted_F1'] = f1_score(all_lebels, all_preds, average="weighted")
    score_dict['accuracy'] = accuracy_score(all_lebels, all_preds)
    score_dict["FPR"] = 1 - recall_score(all_lebels, all_preds, pos_label=0)
    score_dict['all_loss'] = loss_all/len
    # print("Evaling on \"{}\" data".format(data_name))
    # for s_name, s_val in score_dict.items(): 
    #     print("{}: {}".format(s_name, s_val)) 
    return score_dict


def cf_get_scores(all_preds, k_preds, w_preds, s_preds, cf_preds, all_lebels, loss_all, len, data_name):
    
    score_dict = dict()

    score_dict['cf_pred_F1'] = f1_score(all_lebels, cf_preds)
    score_dict['cf_pred_acc'] = accuracy_score(all_lebels, cf_preds)
    score_dict["FPR"] = 1 - recall_score(all_lebels, cf_preds, pos_label=0)    

    score_dict['pred_F1'] = f1_score(all_lebels, all_preds)
    # score_dict['pred_acc'] = accuracy_score(all_lebels, all_preds)
    score_dict["pred_FPR"] = 1 - recall_score(all_lebels, all_preds, pos_label=0)    

    score_dict['k_pred_F1'] = f1_score(all_lebels, k_preds)
    # score_dict['k_pred_acc'] = accuracy_score(all_lebels, k_preds)
    score_dict["k_pred_FPR"] = 1 - recall_score(all_lebels, k_preds, pos_label=0)

    score_dict['w_pred_F1'] = f1_score(all_lebels, w_preds)
    # score_dict['w_pred_acc'] = accuracy_score(all_lebels, w_preds)
    score_dict["w_pred_FPR"] = 1 - recall_score(all_lebels, w_preds, pos_label=0)    

    score_dict['s_pred_F1'] = f1_score(all_lebels, s_preds)
    # score_dict['s_pred_acc'] = accuracy_score(all_lebels, s_preds)
    score_dict["s_pred_FPR"] = 1 - recall_score(all_lebels, s_preds, pos_label=0)    

    # score_dict['all_loss'] = loss_all/len
    # print("Evaling on \"{}\" data".format(data_name))
    # for s_name, s_val in score_dict.items(): 
    #     print("{}: {}".format(s_name, s_val)) 
    return score_dict


# def test_get_scores(all_preds, all_lebels, loss_all, len, data_name):
#     score_dict = dict()
#     f1 = f1_score(all_lebels, all_preds, average='weighted')
#     acc = accuracy_score(all_lebels, all_preds)
#     recall_neg = recall_score(all_lebels, all_preds, pos_label=0)

#     score_dict['F1'] = f1
#     score_dict['accuracy'] = acc
#     score_dict["FPR"] = 1 - recall_neg

#     score_dict['all_loss'] = loss_all/len
#     print("Evaling on \"{}\" data".format(data_name))
#     for s_name, s_val in score_dict.items(): 
#         print("{}: {}".format(s_name, s_val)) 
#     return score_dict

def save_best(config, epoch, model_name, model, score, max_score):
    score_key = config.score_key
    curr_score = score[score_key]

    if curr_score >= max_score:
        max_score = curr_score
        torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        }, '{}/ckp-{}-{}.tar'.format(config.checkpoint_path, model_name, 'BEST'))

    print('The epoch_{} {}: {}\nCurrent max {}: {}'.format(epoch, score_key, curr_score, score_key, max_score))
    return max_score
