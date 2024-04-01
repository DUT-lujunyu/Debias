import os
import numpy as np

from dataset.dataset import *
from config.config import *

# from train_eval_debias_roberta import *
from train_eval_cf import *


if __name__ == "__main__":

    model_name = "CF"
    dataset = "toxic_lang_data_pub"

    config = Config_base(model_name, dataset)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    
    if not os.path.exists(config.data_path): 
        trn_data = Datasets(config, config.train_path)
        dev_data = Datasets(config, config.dev_path)
        test_data = Datasets(config, config.test_path)
        test_data_noi = Datasets(config, config.test_path_noi)
        test_data_oi = Datasets(config, config.test_path_oi)
        test_data_oni = Datasets(config, config.test_path_oni)

        torch.save({
            'trn_data' : trn_data,
            'dev_data' : dev_data,
            'test_data' : test_data,
            'test_data_oni' : test_data_oni,
            'test_data_oi' : test_data_oi,
            'test_data_noi' : test_data_noi,
            }, config.data_path)
    else:
        checkpoint = torch.load(config.data_path)
        trn_data = checkpoint['trn_data']
        dev_data = checkpoint['dev_data']
        test_data = checkpoint['test_data']
        test_data_oni = checkpoint['test_data_oni']
        test_data_oi = checkpoint['test_data_oi']
        test_data_noi = checkpoint['test_data_noi']
        print("Have loaded!")

    train_iter = Dataloader(trn_data,  batch_size=int(config.batch_size), shuffle=False)
    dev_iter = Dataloader(dev_data,  batch_size=int(config.batch_size), shuffle=False)
    test_iter = Dataloader(test_data,  batch_size=int(config.batch_size), shuffle=False)
    test_iter_noi = Dataloader(test_data_noi,  batch_size=int(config.batch_size), shuffle=False)
    test_iter_oi = Dataloader(test_data_oi,  batch_size=int(config.batch_size), shuffle=False)
    test_iter_oni = Dataloader(test_data_oni,  batch_size=int(config.batch_size), shuffle=False)

    train(config, train_iter, dev_iter, test_iter, test_iter_noi, test_iter_oi, test_iter_oni)