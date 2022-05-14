import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import json
import pickle
from tqdm import tqdm
np.set_printoptions(threshold=np.inf)


vocab_dir = "vocab.txt"
with open(vocab_dir, 'r', encoding='utf-8') as f:
    vocab = f.read().split()

#加载原始图片和描述内容
def load_raw_image_path_and_caption_content(save_dir_name="raw_image_path_and_caption_content"):
    if not os.path.exists(save_dir_name):
        raise ValueError(f"Not found {save_dir_name}")
    img_name_train = np.load(save_dir_name + '/img_name_train.npy')
    img_name_val = np.load(save_dir_name + '/img_name_val.npy')
    cap_train = np.load(save_dir_name + '/cap_train.npy')
    cap_val = np.load(save_dir_name + '/cap_val.npy')
    return img_name_train, img_name_val, cap_train, cap_val

#InceptionV3特征提取模型
def initialize_InceptionV3_image_features_extract_model():
    """
    :return: InceptionV3 model instance
    """
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output  # shape [batch_size, 8, 8, 2048]

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    return image_features_extract_model

#载入图片并转为299*299*3大小
def load_image(image_path):
    #print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

#最长匹配
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
                # print(sentence[start:index], end='/')
                start = index
                break
            index += -1
    return token_list

#读取图片名及标签
def read_raw_image_and_caption_file(annotation_file, img_file_dir, num_examples=None):
    all_captions = []
    all_img_name_vector = []
    # Read the file
    for i in os.listdir(annotation_file):
        path = annotation_file + i
        f = open(path, 'r')
        annotations = f.read()
        caption = '<start> ' + annotations+ ' <end>'
        all_captions.append(caption)

    # Store captions and image names in vectors
    for i in os.listdir(img_file_dir):
        all_img_name_vector.append(img_file_dir+i)

    # Shuffle captions and image_names together
    # Set a random state
    train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)

    # Select the first num_examples captions from the shuffled set, None for all data
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    print(f"train_captions numbers {len(train_captions)}\t img_name_vector numbers {len(img_name_vector)}")
    return train_captions, img_name_vector

#提取图片特征载入cache_image中
def caching_image_features_extracted_from_InceptionV3(img_name_vector, cache_image_dir="cache_image"):
    """
    :param img_name_vector: image file path list
    :param cache_image_dir: folder of store image features extracted from Inception V3
    :return:
    """
    if not os.path.exists(cache_image_dir):
        os.mkdir(cache_image_dir)

    # Get unique images
    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    image_features_extract_model = initialize_InceptionV3_image_features_extract_model()

    for img, path in tqdm(image_dataset):
        # shape [batch_size, 8, 8 ,2048]
        batch_features = image_features_extract_model(img)
        # shape [batch_size, 64, 2048]
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

        for batch_features, img_path in zip(batch_features, path):
            path_of_feature = img_path.numpy().decode("utf-8")
            path_of_feature_file_name = os.path.basename(path_of_feature)
            save_file_path_of_feature = os.path.join(cache_image_dir, path_of_feature_file_name)
            np.save(save_file_path_of_feature, batch_features.numpy())

#将描述文字转为向量
def word2vector(train_caption_list):
    train_caption = []
    index = 0
    for content in train_caption_list:
        # print(content)
        # y_train.append(content.replace('\n','').replace(' ',''))

        max_length = 0
        content = content.replace('\n', '')
        content = content.replace('\n','')
        token_list = FMM_func(vocab, content)

        token_list = [token_list[i] for i in range(len(token_list)) if token_list[i] != ' '] # 去除空格

        token_list = ' '.join(token_list)

        train_caption.append(token_list)
    print(train_caption)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='\n')

    tokenizer.fit_on_texts(train_caption)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    print('tokenizer:', tokenizer.word_counts)
    with open("caption_tokenizer_c", "wb") as f:
        pickle.dump(tokenizer, f)
    train_seqs = tokenizer.texts_to_sequences(train_caption)

    caption_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # max_length = max(len(t) for t in caption_vector)

    return caption_vector


#拆分保存原始数据路径及描述内容
def split_and_save_raw_image_path_and_caption_content(img_name_vector, caption_vector, test_size,raw_image_path_and_caption_content):
    save_dir_name = raw_image_path_and_caption_content
    if not os.path.exists(save_dir_name):
        os.mkdir(save_dir_name)
    # Create training and validation sets using an 80-20 split
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, caption_vector,
                                                                        test_size=test_size, random_state=0)

    print(f"img_name_train numbers:\t{len(img_name_train)}\tcap_train numbers:\t{len(cap_train)}\n"
          f"img_name_val numbers:\t {len(img_name_val)}\tcap_val numbers:\t{len(cap_val)}")

    np.save(save_dir_name + '/img_name_train.npy', img_name_train)
    np.save(save_dir_name + '/img_name_val.npy', img_name_val)
    np.save(save_dir_name + '/cap_train.npy', cap_train)
    np.save(save_dir_name + '/cap_val.npy', cap_val)
    print(f"Prepared file save to {save_dir_name}")

def get_caption_tokenizer(caption_tokenizer_path="caption_tokenizer_c"):
    with open(caption_tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        return tokenizer
if __name__ == '__main__':
    annotation_file = './data_c/formula_label_end/'
    img_file_dir = './data_c/formula_image_end/'
    cache_image_dir = 'cache_image_c'
    train_captions, img_name_vector = read_raw_image_and_caption_file(annotation_file, img_file_dir)

    caption_vector = word2vector(train_captions)

    caching_image_features_extracted_from_InceptionV3(img_name_vector, cache_image_dir=cache_image_dir)

    test_size = 0.2

    raw_image_path_and_caption_content = "raw_image_path_and_caption_content_c"

    split_and_save_raw_image_path_and_caption_content(img_name_vector, caption_vector, test_size=test_size,
                                                      raw_image_path_and_caption_content=raw_image_path_and_caption_content)
    #print(get_caption_tokenizer().word_index)