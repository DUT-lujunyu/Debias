import torch
from os import path

class Config_base(object):

    """配置参数"""
    def __init__(self, model_name, dataset):
        # path
        self.model_name = model_name
        self.train_path = path.dirname(path.dirname(__file__))+ '/'+ 'resource/'+ dataset + '/data/preprocess_train.json'                                # 训练集 one_hot 形式
        self.dev_path = path.dirname(path.dirname(__file__))+ '/'+ 'resource/'+ dataset + '/data/preprocess_test.json'                                    # 验证集 one_hot 形式
        self.test_path = path.dirname(path.dirname(__file__))+ '/'+ 'resource/'+ dataset + '/data/containing_dirty_word_test.json'                       # 验证集 one_hot 形式
        self.test_path_noi = path.dirname(path.dirname(__file__))+ '/'+ 'resource/'+ dataset + '/data/NOI_test.json'                       # 验证集 one_hot 形式
        self.test_path_oi = path.dirname(path.dirname(__file__))+ '/'+ 'resource/'+ dataset + '/data/OI_test.json'                       # 验证集 one_hot 形式
        self.test_path_oni = path.dirname(path.dirname(__file__))+ '/'+ 'resource/'+ dataset + '/data/ONI_test.json'                       # 验证集 one_hot 形式
        self.test_path_adv = path.dirname(path.dirname(__file__))+ '/'+ 'resource/'+ "adversial_data" + '/original_data/all_adv_data.json'                       # 验证集 one_hot 形式 
        self.result_path = path.dirname(path.dirname(__file__))+ '/'+ 'resource/'+ dataset + '/result'
        self.checkpoint_path = path.dirname(path.dirname(__file__))+ '/'+ 'resource/'+ dataset + '/saved_dict'        # 数据集、模型训练结果
        self.data_path = self.checkpoint_path + '/aflite_data.tar'
        self.word_list_path = path.dirname(path.dirname(__file__))+ '/'+ "resource/" + dataset + "/data/word_based_bias_list.csv"

        # dataset
        self.seed = 1        
        self.num_classes = 2                                             # 类别数
        self.pad_size = 50                                              # 每句话处理成的长度(短填长切)
        self.emb_pad_size = 10      

        # model
        self.dropout = 0.1                                              # 随机失活
        self.vocab_dim = 1024 if self.model_name == "roberta-large" else 768
        self.mlp_dim = [256]
        self.hidden_size = 768

        # train
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')   # 设备
        self.learning_rate = 1e-5                                       # 学习率  transformer:5e-4 
        self.scheduler = False                                          # 是否学习率衰减
        self.adversarial = False  # 是否对抗训练
        self.num_warm = 0                                               # 开始验证的epoch数
        self.num_epochs = 3                                            # epoch数 
        self.batch_size = 8                                            # mini-batch大小

        # evaluate
        self.threshold = 0.5                                            # 二分类阈值
        # 
        if self.model_name == "CF":
            self.score_key = "cf_pred_F1"
        else:
            self.score_key = "F1"

        # CF
        self.eps = 1e-12
        self.mode = "counter_factual"  # original, counter_factual
        self.fusion_mode = "sum"   # ['rubi', 'hm', 'sum']
        self.if_z_s = True
 

if __name__ == "__main__":
    config = Config_base("bert-base-cased", "toxic_lang_data_pub")  # 引入Config参数，包括Config_base和各私有Config
    print(config.checkpoint_path)
