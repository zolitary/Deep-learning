# 导入常用的包
import random,time
import os, gc
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

pd.set_option('max_colwidth',None)
warnings.filterwarnings('ignore')

# torch相关包
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as Data

from precess_utils import *
from transfomer import *
from train_utils import *

from sklearn.model_selection import train_test_split


# 固定随机种子减少实验波动
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 实现My_Dataset
class My_Dataset(object):

    def __init__(self, dataset, max_ratio, pad=0):
        self.dataset = dataset.reset_index(drop=True)
        self.image_data = self.dataset['image_data'].values
        self.text_id = self.dataset['text_id'].values
        self.pad = pad   # padding标识符的id，默认0
        self.max_ratio = max_ratio * 3      # 将宽拉长3倍

        # 定义 Normalize
        self.trans_Normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])

    def __getitem__(self, index):
        # ----------------
        # 图片预处理
        # ----------------
        # load image
        img =self.image_data[index]

        # 对图片进行大致等比例的缩放
        # 将高缩放到32，宽大致等比例缩放
        w, h = img.size
        ratio = round((w / h) * 3)   # 将宽拉长3倍，然后四舍五入
        if ratio == 0:
            ratio = 1
        if ratio > self.max_ratio:
            ratio = self.max_ratio
        h_new = 32
        w_new = h_new * ratio
        img_resize = img.resize((w_new, h_new), Image.BILINEAR)

        # 对图片右半边进行padding，使得宽/高比例固定=self.max_ratio
        img_padd = Image.new('RGB', (32*self.max_ratio, 32), (0,0,0))
        img_padd.paste(img_resize, (0, 0))

        # Normalize
        img_input = self.trans_Normalize(img_padd)

        # ----------------
        # label处理
        # ----------------

        # 构造encoder的mask
        encode_mask = [1] * ratio + [0] * (self.max_ratio - ratio)
        encode_mask = torch.tensor(encode_mask)
        encode_mask = (encode_mask != 0).unsqueeze(0)

        # 构造ground truth label
        gt = self.text_id[index]

        # decoder的输入
        decode_in = gt[:-1]
        decode_in = torch.tensor(decode_in)

        # decoder的输出
        decode_out = gt[1:]
        decode_out = torch.tensor(decode_out)

        # decoder的mask
        decode_mask = self.make_std_mask(decode_in, self.pad)
        
        # 有效tokens数
        ntokens = (decode_out != self.pad).data.sum()

        return img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens


    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        tgt_mask = tgt_mask.squeeze(0)   # subsequent返回值的shape是(1, N, N)
        return tgt_mask

    def __len__(self):
        return len(self.text_id)

# 模型结构
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, src_position, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed    # input embedding module
        self.src_position = src_position
        self.tgt_embed = tgt_embed    # ouput embedding module
        self.generator = generator    # output generation module

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src --> [bs, 3, 32, 768]  [bs, c, h, w]
        # src_mask --> [bs, 1, 24]  [bs, h/32, w/32]
        memory = self.encode(src, src_mask)
        # memory --> [bs, 24, 512]
        # tgt --> decode_in [bs, 20]  [bs, sequence_len-1]
        # tgt_mask --> decode_mask [bs, 20]  [bs, sequence_len-1]
        res = self.decode(memory, src_mask, tgt, tgt_mask)  # [bs, 20, 512]
        return res

    def encode(self, src, src_mask):
        # feature extract
        # src --> [bs, 3, 32, 768]
        src_embedds = self.src_embed(src)
        src_embedds = src_embedds.squeeze(-2)
        src_embedds = src_embedds.permute(0, 2, 1)

        # position encode
        src_embedds = self.src_position(src_embedds)  # [bs, 24, 512]

        return self.encoder(src_embedds, src_mask)  # [bs, 24, 512]

    def decode(self, memory, src_mask, tgt, tgt_mask):
        target_embedds = self.tgt_embed(tgt)  # [bs, 20, 512]
        return self.decoder(target_embedds, memory, src_mask, tgt_mask)


def make_model(tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy


    backbone = models.resnet18(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])    # 去掉最后两个层 (global average pooling and fc layer)

    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # 构建模型
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        backbone,
        c(position),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # Initialize parameters with Glorot / fan_avg.
    for child in model.children():
        if child is backbone:
            for param in child.parameters():
                param.requires_grad = True

            continue
        for p in child.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol):
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).long()
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        next_word = torch.ones(1, 1).type_as(src.data).fill_(next_word).long()
        ys = torch.cat([ys, next_word], dim=1)

        next_word = int(next_word)
        if next_word == end_symbol:
            break

    ys = ys[0, 1:]

    return ys


def EditDistance(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    if len_str1 > len_str2:
        max_len = len_str1
    else: max_len = len_str2

    matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if(str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1
            
            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)
    
    value = 1.0 - float(matrix[len(str1)][len(str2)])/float(max_len)

    return value


def ExtarctMatch(str1, str2):
    str2 = str2[:len(str1)]
    is_correct = 1 if str1 == str2  else 0

    return int(is_correct)

def Evaluation_statistics(model,data_loader,sequence_len,id2lbl_map,device):
    edit_distance_sum = 0.0
    extarct_match_sum = 0.0
    total_sum = 0.0
    record_str1 = []
    record_str2 = []
    for batch_idx, batch in enumerate(data_loader):
        img_input, encode_mask, decode_in, decode_out, decode_mask, ntokens = batch
        img_input = img_input.to(device)
        encode_mask = encode_mask.to(device)
 
        # 获取单张图像信息
        print('Batch [{}] '.format(batch_idx+1),end='')
        bs = img_input.shape[0]
        for i in range(bs):
            cur_img_input = img_input[i].unsqueeze(0)
            cur_encode_mask = encode_mask[i].unsqueeze(0)
            cur_decode_out = decode_out[i].cpu()

            # 贪心解码
            pred_result = greedy_decode(model, cur_img_input, cur_encode_mask, max_len=sequence_len, start_symbol=1, end_symbol=2)
            pred_result = pred_result.cpu()

            # 转换成字符串
            str1 = Sequence_to_string(cur_decode_out.data.numpy(),id2lbl_map)
            str2 = Sequence_to_string(pred_result.data.numpy(),id2lbl_map)
            record_str1.append(str1)
            record_str2.append(str2)

            # 记录
            total_sum += 1
            edit_distance_sum += EditDistance(str1, str2)
            extarct_match_sum += ExtarctMatch(str1, str2)
            res_list = pd.DataFrame({'str1':record_str1,'str2':record_str2})
            if i%50 == 0:
                print('#',end='')

        print()
    # 计算指标(以百分数表示)
    edit_distance_score = edit_distance_sum / total_sum*100
    extarct_match_score = extarct_match_sum / total_sum*100
    

    return edit_distance_score, extarct_match_score, res_list

class EnjoyTime:
    def __init__(self,input_path,config):

        # 加载词典映射文件
        lbl2id_map, id2lbl_map = load_lbl2id_map(input_path['lbl2id_map_path'])

        # 建立模型
        tgt_vocab = len(lbl2id_map.keys()) 
        model = make_model(tgt_vocab, N=4, d_model=512, d_ff=2048, h=8, dropout=0.12).to(config['device'])

        # 加载预训练权重
        '''
        model.load_state_dict(torch.load(input_path['pretrain_weights'],map_location=torch.device('cpu')))
        self.model = model.eval()
        self.id2lbl_map = id2lbl_map
        self.config = config
        self.input_path = input_path
'''

    def Create_Input(self, img, max_ratio=3):
        # 定义 Normalize
        trans_Normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
        # ----------------
        # 图片预处理
        # ----------------
      
        # 对图片进行大致等比例的缩放
        # 将高缩放到32，宽大致等比例缩放
        w, h = img.size
        ratio = round((w / h) * 3)   # 将宽拉长3倍，然后四舍五入
        if ratio == 0:
            ratio = 1
        if ratio > max_ratio:
            ratio = max_ratio
        h_new = 32
        w_new = h_new * ratio
        img_resize = img.resize((w_new, h_new), Image.BILINEAR)

        # 对图片右半边进行padding，使得宽/高比例固定=self.max_ratio
        img_padd = Image.new('RGB', (32*max_ratio, 32), (0,0,0))
        img_padd.paste(img_resize, (0, 0))

        # Normalize
        img_input = trans_Normalize(img_padd).unsqueeze(dim=0)

        # 构造encoder的mask
        encode_mask = [1] * ratio + [0] * (max_ratio - ratio)
        encode_mask = torch.tensor(encode_mask)
        encode_mask = (encode_mask != 0).unsqueeze(0)

        return img_padd, img_input, encode_mask

    def Recognize_single_image(self,image_path,max_ratio=3,require_show=True):

        img_padd, cur_img_input, cur_encode_mask = self.Create_Input(image_path,max_ratio)

        pred = greedy_decode(self.model, cur_img_input, cur_encode_mask, 
                        max_len=self.config['sequence_len'], 
                        start_symbol=1, end_symbol=2)

        result = Sequence_to_string(pred.cpu().data.numpy(),id2lbl_map=self.id2lbl_map)

        if require_show == True:
            plt.figure(figsize=(4,3),dpi=100)
            plt.title('Rec: '+result,c='red')
            plt.imshow(img_padd)
        
        return img_padd, result

    def Multi_impressions(self,image_paths):
        nums = len(image_paths)

        fig, axes = plt.subplots(3, nums//3, figsize=(7,4))
        fig.set_facecolor(color='white')
        CNT = 0
        for i, row in enumerate(axes):
            for j, col in enumerate(row):
                img_padd, result = self.Recognize_single_image(image_paths[CNT],require_show=False)
                col.imshow(img_padd)
                col.set_title('Rec: '+result,c='red')
                col.set_xticks([])
                col.set_yticks([])
                CNT += 1

        plt.tight_layout()
        plt.savefig(self.input_path['save_figure_path']+'/show_results.png')
        print('输出图像保存在-->'+self.input_path['save_figure_path']+'/show_results.png')


if __name__ == '__main__':

    input_path = {'labels_path':'data/math_formula_images_grey_labels',
              'images_path':'data/math_formula_images_grey',
              'vocab_path':'vocab.txt',
              'lbl2id_map_path':'lbl2id_map.txt',
              'data_path':'pure_data.pickle',
              'train_data_path':'train_data.pickle',
              'valid_data_path':'valid_data.pickle',
              'test_data_path':'test_data.pickle',
              'pretrain_weights':'model_weights/model_weights_100.pickle',
              'weights_save_path':'model_weights/model_weights_new.pickle',
              'save_figure_path':'figures'}

    config = {'reprocess_data':True,
          'retrain':True,
          'sequence_len':10,
          'sample_frac':1,
          'test_frac':0.08,
          'device':'cpu',
          'use_sample_num':1,
          'epochs':100,
          'verbose':1,
          'seed':520,
          'num_workers':0}

    print('---------------程序开始------------------',end='\n\n')

    # 固定随机种子
    set_seed(config['seed'])


    # 数据预处理


    if config['reprocess_data'] == True:
        print('---------------数据预处理-----------------')
        t1 = time.time()
        data = data_processor(input_path, config)
        print(f'数据总量: {data.shape[0]}')
        t2 = time.time()
        process_data_time = (t2-t1)/60.0
        print('Time to process data: %.3f min'%process_data_time)
        print('-----------------------------------------',end='\n\n')
    
        # 垃圾回收
        gc.collect()
    else : 
        print('-------您选择了直接加载处理后的数据------',end='\n\n')
        
        # 加载数据
        print('---------------加载数据------------------')
        t1 = time.time()
        data = pd.read_pickle(input_path['data_path'])
        t2 = time.time()
        load_data_time = (t2-t1)/60.0
        print(f'数据总量: {data.shape[0]}')
        print('Time to load data: %.3f min'%load_data_time)
        print('-----------------------------------------',end='\n\n')


    # 划分数据集

    train_valid_data, test_data = train_test_split(data,test_size=config['test_frac'],random_state=config['seed'])
    train_data, valid_data = train_test_split(train_valid_data.sample(frac=config['use_sample_num']),
                                              test_size=0.2,random_state=config['seed'])
    '''
    print('-----------------------------------------')
    print(f'训练样本数: {len(train_data)}')
    print(f'验证样本数: {len(valid_data)}')
    print(f'测试样本数: {len(test_data)}')
    '''
    train_data.to_pickle(input_path['train_data_path'], protocol=3)
    valid_data.to_pickle(input_path['valid_data_path'], protocol=3)
    test_data.to_pickle(input_path['test_data_path'], protocol=3)
    print('-----------------------------------------',end='\n\n')



    train_data = pd.read_pickle(input_path['train_data_path'])
    valid_data = pd.read_pickle(input_path['valid_data_path'])
    test_data = pd.read_pickle(input_path['test_data_path'])

    # 构建数据加载器
    train_loader = Data.DataLoader(My_Dataset(train_data, max_ratio=1, pad=0),
                                batch_size=512,
                                shuffle=True,
                                num_workers=config['num_workers']) # 在CPU环境中好像只能设置为0，不然直接卡住

    valid_loader = Data.DataLoader(My_Dataset(valid_data, max_ratio=1, pad=0),
                                batch_size=1024,
                                shuffle=False,
                                num_workers=config['num_workers'])

    test_loader = Data.DataLoader(My_Dataset(test_data, max_ratio=1, pad=0),
                                batch_size=1024,
                                shuffle=False,
                                num_workers=config['num_workers'])
    # 加载词典映射文件
    lbl2id_map, id2lbl_map = load_lbl2id_map(input_path['lbl2id_map_path'])

    # 建立模型
    tgt_vocab = len(lbl2id_map.keys()) 
    model = make_model(tgt_vocab, N=4, d_model=512, d_ff=2048, h=8, dropout=0.12).to(config['device'])

    # 加载预训练权重
    #model.load_state_dict(torch.load(input_path['pretrain_weights'],map_location=torch.device('cpu')))

    # 定义训练相关的配置
    criterion = LabelSmoothing(size=tgt_vocab, padding_idx=0, smoothing=0.03)  # label smoothing
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    model_opt = NoamOpt(512, 1, 400, optimizer)  # warmup

    if config['retrain'] == True:
        print('----------------训练模型-----------------')
        t1 = time.time()
        model,history = train(model,train_loader,valid_loader,criterion,model_opt,input_path,config)
        np.save('history.npy',history)
        t2 = time.time()
        train_time = (t2-t1)/60.0
        print('Time to train the model: %.3f min'%train_time)
        print('-----------------------------------------',end='\n\n')
    else: 
        print('---------您选择了不用再次训练----------',end='\n\n')
        model.eval()


    # 评估
    print('----------------评估模型-----------------')
   
    t1 = time.time()
    res_data = Evaluation_statistics(model,test_loader,config['sequence_len'],id2lbl_map,config['device'])
    edit_distance_score, extarct_match_score, result_file = res_data

    print('\n输出评估结果-->')
    print(f'edit_distance_score: {edit_distance_score}%')
    print(f'extarct_match_score: {extarct_match_score}%')

    # 保存验证结果
    result_file.to_csv('result_list.csv',index=False)

    gc.collect()

    t2 = time.time()
    evaluate_time = (t2-t1)/60.0
    print('Time to evaluate the model: %.3f min'%evaluate_time)
    print('-----------------------------------------',end='\n\n')
    

    print('--------------识别样例展示---------------')
    num = 4
    sample_result = result_file.sample(num).values
    print('------gt------'+'\t<---->\t'+'-----ocr-----')
    for i in range(num):
        gt, ocr = sample_result[i][0],sample_result[i][1]
        print(gt+'\t<---->\t'+ocr)
    print('-----------------------------------------',end='\n\n')
    

    enjoy = EnjoyTime(input_path,config)
    image_datas = test_data['image_data'].sample(9).values
    enjoy.Multi_impressions(image_datas)

