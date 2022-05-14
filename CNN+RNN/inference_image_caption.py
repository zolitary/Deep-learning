import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from data_utils import load_image, initialize_InceptionV3_image_features_extract_model, \
    get_caption_tokenizer, load_raw_image_path_and_caption_content
from model import CNN_Encoder, RNN_Decoder


# 将图片预测依据和预测结果对应的图片保存
def plot_attention(idx, image, result, attention_plot, plt_show=False):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)

    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result // 4 + 1, 4 + 1, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.savefig(f"{idx}.png")
    if plt_show:
        plt.show()


# 将模型训练的存档载入
def restore_model(checkpoint_path, vocab_size):
    image_features_extract_model = initialize_InceptionV3_image_features_extract_model()
    encoder = CNN_Encoder()
    decoder = RNN_Decoder(vocab_size)
    optimizer = tf.keras.optimizers.Adam()

    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!')

    return image_features_extract_model, encoder, decoder,


def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    str1_list = str1.split()
    str2_list = str2.split()
    matrix = [[i + j for j in range(len(str2_list) + 1)] for i in range(len(str1_list) + 1)]

    for i in range(1, len(str1_list) + 1):
        for j in range(1, len(str2_list) + 1):
            if (str1_list[i - 1] == str2_list[j - 1]):
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1_list)][len(str2_list)]


def main(model_predicte_number, checkpoint_path, raw_image_path_and_caption_content_dir,
         caption_max_length, attention_features_shape, plot_image_attention):
    def evaluate():
        attention_plot = np.zeros((caption_max_length, attention_features_shape))

        hidden = decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        image_features_encoder = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []
        dicts = {}
        for i in range(caption_max_length):
            predictions, hidden, attention_weights = decoder(dec_input, image_features_encoder, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            if tokenizer.index_word[predicted_id] not in dicts:
                dicts[tokenizer.index_word[predicted_id]] = 1
            else:
                dicts[tokenizer.index_word[predicted_id]] += 1

            if dicts[tokenizer.index_word[predicted_id]] <= 2:
                result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot


            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]

        return result, attention_plot

    # restore tokenizer
    tokenizer = get_caption_tokenizer(caption_tokenizer_path="caption_tokenizer")
    vocab_size = len(tokenizer.word_index) + 1

    # Preparing validation set data
    _, img_name_val, _, cap_val = load_raw_image_path_and_caption_content(
        save_dir_name=raw_image_path_and_caption_content_dir)
    # img_name_val, _, cap_val, _ = load_raw_image_path_and_caption_content(
    #     save_dir_name=raw_image_path_and_caption_content_dir)
    # restore image caption model
    image_features_extract_model, encoder, decoder = restore_model(checkpoint_path, vocab_size)

    # model prediction
    correct = 0
    L_distance = 0
    real = []
    predict = []
    for idx in range(model_predicte_number):
        # captions on the validation set
        rid = np.random.randint(0, len(img_name_val))
        image = img_name_val[rid]
        real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
        result, attention_plot = evaluate()
        # if plot_image_attention:
        #     Image.open(image)
        # plot_attention(idx, image, result, attention_plot, plt_show=False) #Todo:fix Bug
        real_caption = real_caption.replace('<start> ', '')
        real_caption = real_caption.replace(' <end>', '')
        results = ' '.join(result)
        results = results.replace('<end>', '')
        real.append(real_caption)
        predict.append(results)
        res_list = pd.DataFrame({'Real Formula': real, 'Prediction Formula': predict})

        print('Real Formula:', real_caption)
        print('Prediction Formula:', ' '.join(result).replace(' <end>', ''))
        print("Levenshtein_Distance_Recursive: ", 1-(Levenshtein_Distance(real_caption,results)/caption_max_length))
        L_distance += 1-(Levenshtein_Distance(real_caption,results)/caption_max_length)
        if 1-(Levenshtein_Distance(real_caption,results)/caption_max_length)==1:
            correct += 1
    print('测试样本数量：16000')
    print('平均Levenshtein距离: ', L_distance/16000)
    print('ExactMatch: ', correct/16000)
    res_list.to_csv('formula_predict.csv',index=False)

    # for rid in range(16000):
    #     # captions on the validation set
    #     image = img_name_val[rid]
    #     real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
    #     result, attention_plot = evaluate()
    #
    #     print('Real Caption:', real_caption)
    #     print('Prediction Caption:', ' '.join(result))
    #     # if plot_image_attention:
    #     #     Image.open(image)
    #     # plot_attention(idx, image, result, attention_plot, plt_show=False) #Todo:fix Bug
    #     real_caption = real_caption.replace('<start>', '')
    #     real_caption = real_caption.replace('<end>', '')
    #     results = ' '.join(result)
    #     results = results.replace('<start>', '')
    #     results = results.replace('<end>', '')
    #     print("Levenshtein_Distance_Recursive: ", 1-(Levenshtein_Distance(real_caption,results)/caption_max_length))
    #     L_distance += 1-(Levenshtein_Distance(real_caption,results)/caption_max_length)
    #     if 1-(Levenshtein_Distance(real_caption,results)/caption_max_length)==1:
    #         correct += 1
    # print('平均Levenshtein距离',L_distance/16000)
    # print(correct/16011)
if __name__ == "__main__":
    model_predicte_number = 16000
    caption_max_length = 30
    checkpoint_path = "checkpoints1/train"
    raw_image_path_and_caption_content_dir = "raw_image_path_and_caption_content"
    attention_features_shape = 64
    plot_image_attention = False

    main(model_predicte_number, checkpoint_path, raw_image_path_and_caption_content_dir,
         caption_max_length, attention_features_shape, plot_image_attention)