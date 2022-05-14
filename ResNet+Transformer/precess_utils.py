# 导入常用的包
import os, glob
from typing import Protocol
import warnings
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# 忽略警告
warnings.filterwarnings('ignore')


def read_txt(file_dir,filenane):
    """
    函数功能：读取单个标签文件（.txt）里面的内容
    参数说明：
    file_dir 是存放标签文件的文件夹路径，采用相对路径即是文件夹名
    filename 是需要读取的标签文件名
    """
    filepath = os.path.join(file_dir,filenane)
    with open(filepath,"r",encoding="utf-8") as f:
        text = f.read()
        
        return text

def name_to_number(x):
    """
    函数功能：提取文件名并将其转化成数值，服务于后续的文件排序
    参数说明：
    x 是传入的文件名
    返回值：由文件名生成的唯一数值
    """
    t = x.split('.')[0].split('_')

    return int(t[0]) + int(t[1])/1000.0


def FMM_func(user_dict, sentence):
    """
    正向最大匹配（FMM）
    :param user_dict: 词典
    :param sentence: 句子
    """
    # 词典中最长词长度
    max_len = max([len(item) for item in user_dict])
    start = 0
    token_list = []
    while start != len(sentence):
        index = start+max_len
        if index>len(sentence):
            index = len(sentence)
        for i in range(max_len):
            if (sentence[start:index] in user_dict) or (len(sentence[start:index])==1):
                token_list.append(sentence[start:index])
                start = index
                break
            index += -1

    return token_list

def is_contain_chinese(check_str):
    """
    判断字符串中是否包含中文
    :param check_str: {str} 需要检测的字符串
    :return: {bool} 包含返回True， 不包含返回False
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def is_error_mathpix(check_str) :
    """
    判断字符串中是否有换行以及标签错误的情况
    """
    if '\n' in check_str:
        return True

    if 'error mathpix' in check_str:
        return True
    else: return False

def statistics_label_cnt(df, lbl_cnt_map):
    """
    统计标签文件中label都包含哪些字符以及各自出现的次数
    lbl_cnt_map : 记录标签中字符出现次数的字典
    """
    for d in df:
        for lbl in d:
                if lbl not in lbl_cnt_map.keys():
                    lbl_cnt_map[lbl] = 1
                else:
                    lbl_cnt_map[lbl] += 1

def Filter_low_frequency_char(lbl_cnt_map,f=10):
    """
    函数功能：去除字典中低频字符，根据分析大多数异常字符都是低频词，所以采用这种方式进行过滤
    参数说明：lbl_cnt_map 是记录标签中字符出现次数的字典
              f 是词频
    返回：new_map 是去除低频词后的词典，low_frequency_char 是低频词列表
    """
    new_map = lbl_cnt_map.copy()
    dict_item = lbl_cnt_map.items()
    low_frequency_char = []
    for t in dict_item:
        if t[1] < f:
            new_map.pop(t[0])
            low_frequency_char.append(t[0])

    return new_map,low_frequency_char


def is_contain_low_char(text, low_char):
    """
    判断字符串中是否有低频词
    """
    for c in text:
        if c in low_char:
            return True

    return False

def Generate_character_id(lbl2id_map_path,lbl_cnt_map):
    """
    函数功能：为每个词生成对应的数值id并保存在（.txt）文件中
    参数说明：lbl2id_map_path 是文件保存路径
              lbl_cnt_map 是记录标签中字符出现次数的字典
    """
    # 构造label中 字符--id 之间的映射
    lbl2id_map = dict()

    # 初始化三个特殊字符
    lbl2id_map['☯'] = 0    # padding标识符
    lbl2id_map['■'] = 1    # 句子起始符
    lbl2id_map['□'] = 2    # 句子结束符

    # 生成其余字符的id映射关系
    cur_id = 3
    for lbl in lbl_cnt_map.keys():
        lbl2id_map[lbl] = cur_id
        cur_id += 1

    # 保存 字符--id 之间的映射 到txt文件
    with open(lbl2id_map_path, 'w', encoding='utf-8') as writer:  # 参数encoding是可选项，部分设备并未默认为utf-8
        for lbl in lbl2id_map.keys():
            cur_id = lbl2id_map[lbl]
            line = lbl + '\t' + str(cur_id) + '\n'
            writer.write(line)

def load_lbl2id_map(lbl2id_map_path):
    """
    读取 字符-id 映射关系记录的txt文件，并返回 lbl->id 和 id->lbl 映射字典
    lbl2id_map_path : 字符-id 映射关系记录的txt文件路径
    """
    lbl2id_map = dict()
    id2lbl_map = dict()
    with open(lbl2id_map_path, 'r',encoding='utf-8') as reader:
        for line in reader:
            items = line.rstrip().split('\t')
            label = items[0]
            cur_id = items[1]
            lbl2id_map[label] = int(cur_id)
            id2lbl_map[cur_id] = label
            
    return lbl2id_map, id2lbl_map


def read_image(path,name):
    """
    功能：读取文件夹中的图片数据
    path: 图片文件夹路径
    name：图片文件名
    """
    img = Image.open(os.path.join(path,name)).convert('RGB')
    
    return img

def text_to_id(lbl2id_map,lbl_str,sequence_len):
    """
    函数功能：将list形式的字符转换成数值序列
    参数说明：lbl2id_map 词--> id 
              sequence_len 序列长度
    """
    gt = []
    gt.append(1)    # 先添加句子起始符
    for lbl in lbl_str:
        gt.append(int(lbl2id_map[lbl]))
    gt.append(2)

    for i in range(len(lbl_str), sequence_len):   # 除去起始符终止符，lbl长度为sequence_len，剩下的padding
        gt.append(0)

    # 截断为预设的最大序列长度
    gt = gt[:sequence_len]

    return gt

def data_processor(input_path,config):
    """
    输入参数示例:
    input_path = {'labels_path':'data\chemistry_formula_images_grey_labels',
              'images_path':'data\chemistry_formula_images_grey',
              'vocab_path':'vocab.txt',
              'lbl2id_map_path':'lbl2id_map.txt'}

    config = {'sequence_len':10,
          'sample_frac':0.01}
    """
    print('读取标签-->',end='')
    image_labels = pd.DataFrame({'filename':os.listdir(input_path['labels_path'])}).sample(frac=config['sample_frac'])
    image_labels['text'] = image_labels['filename'].apply(lambda x: read_txt(input_path['labels_path'],x))

    print('读取词表-->',end='')
    with open(input_path['vocab_path'], 'r', encoding='utf-8') as f:
        user_dict = f.read().split()

    print('清洗数据-->',end='')
    image_labels['number'] = image_labels['filename'].apply(name_to_number)
    image_labels['text'] = image_labels['text'].apply(lambda sentence: FMM_func(user_dict, sentence))
    image_labels['text_len'] = image_labels['text'].apply(lambda x: len(x))
    image_labels['image_name'] = image_labels['filename'].apply(lambda x: x.split('.')[0]+'.png')
    image_labels = image_labels.sort_values(by='number')
    image_labels = image_labels.reset_index(drop=True)

    print('过滤中文字符、多行、错误标签-->',end='')
    image_labels['is_contain_chinese'] = image_labels['text'].map(is_contain_chinese)
    image_labels['is_error_mathpix'] = image_labels['text'].map(is_error_mathpix)
    data = image_labels[(image_labels['is_contain_chinese']==False) & (image_labels['is_error_mathpix']==False)]

    print('过滤长标签数据-->',end='')
    data = data[data['text_len'] < config['sequence_len']]
    data.drop(['number'],axis=1,inplace=True)
    data = data.reset_index(drop=True)

    print('词典生成-->',end='')
    lbl_cnt_map = dict()  # 用于存储字符出现次数的字典
    statistics_label_cnt(data['text'], lbl_cnt_map)  # 训练集中字符出现次数统计
    new_lbl_cnt_map, low_char = Filter_low_frequency_char(lbl_cnt_map)
    data['is_contain_low_char'] = data['text'].apply(lambda text: is_contain_low_char(text,low_char))
    new_data = data[data['is_contain_low_char']==False]
    lbl_cnt_map = dict() 
    statistics_label_cnt(new_data['text'], lbl_cnt_map) 

    # 生成词典及其对应的数值id
    Generate_character_id(input_path['lbl2id_map_path'], lbl_cnt_map)

    # 加载词典映射关系
    lbl2id_map, id2lbl_map = load_lbl2id_map(input_path['lbl2id_map_path'])

    # 提取干净的的数据
    pure_data = new_data[['image_name','text']]
    pure_data = pure_data.reset_index(drop=True)

    print('提取图片数据-->',end='')
    pure_data['image_path'] = pure_data['image_name'].map(lambda name: input_path['images_path']+'/'+name)
    pure_data['image_data'] = pure_data['image_path'].apply(lambda p: Image.open(p).convert('RGB'))

    print('字符序列转数值序列-->',end='')
    pure_data['text_id'] = pure_data['text'].apply(lambda x: text_to_id(lbl2id_map,x,config['sequence_len']))

    res_data = pure_data[['image_path','text','image_data','text_id']]
    
    print('保存数据',end='\n\n')
    res_data.to_pickle(input_path['data_path'], protocol=3)

    return res_data

def Sequence_to_string(seq,id2lbl_map):
    """
    函数功能：将数值序列转换成字符串
    参数说明:
            seq： 数值序列
            id2lbl_map：数值向字符映射的词典
    返回结果: 字符串
    """
    temp_str = ''
    for i in seq:
        if i in [0,1,2]:
            continue
        temp_str += id2lbl_map[str(i)]

    return temp_str