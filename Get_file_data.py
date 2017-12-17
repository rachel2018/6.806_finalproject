import os
import numpy as np
from os.path import dirname, realpath
import gzip

HOME_PATH = dirname(dirname(realpath(__file__)))
DATA_PATH = os.path.join(HOME_PATH,'ubuntu')
VECTORS_FILE = os.path.join(DATA_PATH,"vector","vectors_pruned.200.txt.gz")
DATA_FILE = os.path.join(DATA_PATH,"text_tokenized.txt.gz")
TRAIN_FILE = os.path.join(DATA_PATH,"train_random.txt")
DEV_SET_FILE = os.path.join(DATA_PATH,"dev.txt")
TEST_SET_FILE = os.path.join(DATA_PATH,"test.txt")

def get_embeddings_tensor():
    with gzip.open(VECTORS_FILE) as f:
        data = f.readlines()
    embedding_tensor = []
    word_to_index = {}
    for index, l in enumerate(data):
        word, vector= l.strip().split(" ", 1)
        vector = map(float, vector.split(" "))

        if index == 0:
            embedding_tensor.append(np.zeros(len(vector)))
            embedding_tensor.append(np.zeros(len(vector)))
        embedding_tensor.append(vector)
        word_to_index[word] = index+2

    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word_to_indx

def helper(data_file):
    examples = []
    with open(data_file,"rb") as f:
        data = f.readlines()
        for l in data:
            qid, similar_qids, cand, scores = line.strip().split("\t",3)
            if similar_qids == "":
                similar_qids = [] 
            else:
                similar_qids = map(int, similar_qids.split(' '))
            cand = map(int, cand.split(' '))
            scores = map(float, scores.split(' '))
            examples.append((int(qid), similar_qids, cand, scores))
    return examples


def get_data():
    dic = {}
    with gzip.open(DATA_FILE) as f:
        data = f.readlines()
        for l in data:
            qid, q = l.strip().split("\t",1)
            q = q.split("\t")
            if len(q) == 1: 
                dic[int(qid)] = (q[0], "")
            else: 
                dic[int(qid)] = (q[0], q[1])
    return dic

def get_train():
    examples = []
    with open(TRAIN_FILE,"rb") as f:
        data = f.readlines()
        for l in data:
            qid, similar_qids, random_qids = l.strip().split("\t",2)
            similar_qids = map(int, similar_qids.split(' '))
            random_qids = map(int, random_qids.split(' '))
            examples.append((int(qid), similar_qids, random_qids))
    return train_examples

def get_test():
    return helper(TEST_SET_FILE)

def get_dev():
    return helper(DEV_SET_FILE)
