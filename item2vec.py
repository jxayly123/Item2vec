import tensorflow as tf
import numpy as np
import math
import collections
import random
from tensorflow.contrib.tensorboard.plugins import projector
import gc
import csv
import os
# import pandas as pd
# from six.moves import xrange


local_file = 'D:/pycharm/item2vec/ml-20m/train_data.txt'
log_dir = 'D:/pycharm/item2vec/ml-20m/train_log'


def read_data(filename):
    corpus = list()
    vocabulary = set()
    words = list()
    max_len = 0
    with open(filename, 'r') as f:
        for line in f:
            sentence = line.strip().split('|')
            corpus.append(sentence)
            max_len = max(max_len, len(sentence))
            words += sentence
    print('corpus size: {} sentence {} words'.format(len(corpus), len(words)))
    print('sentence max len: {}'.format(max_len))
    return corpus, words

def read_txt(filename):
    corpus = list()
    words = list()
    max_len = 0
    with open(filename) as f:
        for line in f:
            sentences = line.strip().split(' ')
            corpus.append(sentences)
            max_len = max(max_len, len(sentences))
            words += sentences
    f.close()
    print('corpus size: {} sentence {} words'.format(len(corpus), len(words)))
    print('sentence max len: {}'.format(max_len))
    return corpus, words, set(words)
            

def generate_batch_from_sentence(sentence, num_skips, skip_window):
    batch_inputs = []
    batch_labels = []
    for i in range(len(sentence)):
        window = list(range(len(sentence)))
        window.remove(i)
        # 句子内除该元素以外的所有元素
        sample_index = random.sample(window, min(num_skips, len(window)))
        input_id = word2id.get(sentence[i])
        for index in sample_index:
            label_id = word2id.get(sentence[index])
            batch_inputs.append(input_id)
            batch_labels.append(label_id)

    batch_inputs = np.array(batch_inputs, dtype=np.int32)
    batch_labels = np.array(batch_labels, dtype=np.int32)
    batch_labels = np.reshape(batch_labels, [-1, 1])
    # print(len(batch_inputs), len(batch_labels))
    return batch_inputs, batch_labels


# 建立item到id以及id到item的映射
corpus, words, vocabulary = read_txt(local_file)
# vocabulary = collections.Counter(words)
count = []
count.extend(collections.Counter(words).most_common(len(words) - 1))
word2id = dict()
for word, _ in count:
    word2id[word] = len(word2id)

id2word = dict(zip(word2id.values(), word2id.keys()))
word2name = dict()
movie_file = "D:/pycharm/item2vec/ml-20m/movies1.csv"
with open(movie_file, 'r',encoding='utf-8') as f:
    file = csv.reader(f)
    linenum = 0
    for line in file:
        if linenum == 0:
            linenum += 1
            continue
        # items = line.strip().split(',')
        # print(line)
        movieid, movie_name = line[0], line[1]
        word2name[movieid] = movie_name
f.close()

#模型设置
embedding_size = 128    # embeding 维度
skip_window = 2        # 单边窗口长度
num_skips = 4          # 从整个窗口中选取多少个不同的item作为我们的output item
num_samples = 5        # 负采样样本数

valid_size = 16         # 选取验证的样本数
valid_window = 500      # 在前500个挑选验证样本
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

vocabulary_size = len(vocabulary)
print("vocabulary size:", vocabulary_size)
batch_size = 128


graph = tf.Graph()
with graph.as_default():
    # 输入数据
    with tf.name_scope("inputs"):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 查找输入的embedding
    with tf.name_scope("embeddings"):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    # 模型内部参数矩阵初始化
    with tf.name_scope("weights"):
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                      stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # 得到nce损失
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_samples,
            num_classes=vocabulary_size))

    tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # 计算与指定若干item的相似度
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embedding = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)

    similarity = tf.matmul(valid_embeddings, normalized_embedding, transpose_b=True)

    merged = tf.summary.merge_all()

    # 变量初始化
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()




# 训练部分

num_steps = 1
earily_stop = 10
count = 0
flag = 1
with tf.Session(graph=graph) as sess:
    train_loss_records = collections.deque(maxlen=10)       #保存最近10次的误差
    train_loss_k10 = 0
    train_sents_num = 0


    # 写入summary
    writer = tf.summary.FileWriter(log_dir, sess.graph)


    # 初始化全局参数
    init.run()
    aver_loss = 0
    block = 10
    while flag:
        for step in range(len(corpus) // block):
            batch_inputs = list()
            batch_labels = list()
            s_count = 0
            while s_count<= block:
                # print("structuring inputs and outputs.")
                sentences = corpus[(step*block+s_count) % len(corpus)] #逐句训练
                if len(sentences) < 2*skip_window+1:
                    s_count += 1
                    continue
                batch_input, batch_label = generate_batch_from_sentence(sentences, num_skips, skip_window)
                batch_inputs.extend(batch_input)
                batch_labels.extend(batch_label)
                s_count += 1
                # print("There are {} sentences have been concatenated".format(s_count))
            # print("batch_inputs:",batch_inputs)
            # print("batch_labels:", batch_labels)
            for x in range(len(batch_inputs) // batch_size):
                feed_dict = {train_inputs: batch_inputs[x*batch_size:x*batch_size+batch_size], train_labels: batch_labels[x*batch_size:x*batch_size+batch_size]}
                      # Define metadata variable.
                run_metadata = tf.RunMetadata()
                _, summary, loss_val = sess.run([optimizer, merged, loss],
                                                feed_dict=feed_dict)
            del batch_labels
            del batch_inputs
        # gc.collect()
                # aver_loss += loss_val
            train_sents_num += 10
            train_loss_records.append(loss_val)
            train_loss_k10 = np.mean(train_loss_records)
            # 每一步将summary写入Writter
            writer.add_summary(summary, step)
            if train_loss_k10 <= 2.2:
                count += 1
                if count >= earily_stop:
                    flag = 0
                    # Add metadata to visualize the graph for the last run.
                    writer.add_run_metadata(run_metadata, 'step{}'.format(step))
                    break
            if train_sents_num % 100 == 0:
                print('{a} sentences dealed, loss: {b}'.format(a=train_sents_num, b=train_loss_k10))

            if train_sents_num % 5000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = id2word[valid_examples[i]]
                    valid_name = word2name[valid_word]
                    top_k = 8
                    nearest = (-sim[i,:]).argsort()[1: top_k + 1]
                    log_str = 'Nearest to {}:'.format(valid_name)

                    for k in range(top_k):
                        close_word = id2word[nearest[k]]
                        close_name = word2name[close_word]
                        log_str = "{} {},".format(log_str, close_name)

                    print(log_str)

    final_embeddings = normalized_embedding.eval()

    # Write corresponding labels for the embeddings.
    with open(log_dir + '/metadata.tsv', 'w') as f:
        for i in (vocabulary):
            f.write(word2name[i] + '\n')

    saver.save(sess, os.path.join(log_dir, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in
    # TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

    writer.close()
