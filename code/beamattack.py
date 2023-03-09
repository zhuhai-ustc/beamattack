from nltk.corpus import stopwords
import pickle
import copy
from tqdm import *
import math
import argparse
import os
import numpy as np
import dataloader
from train_classifier import Model
import criteria
import random
import tensorflow as tf
import tensorflow_hub as hub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from train_classifier import Model
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig
from generate_embedding_mat import generate_embedding_mat
import generateCosSimMat
from scipy.spatial.distance import pdist
from transformers import pipeline
import datetime# 攻击开始
import tqdm
import argparse
parser = argparse.ArgumentParser()
unmasker = pipeline('fill-mask', model='bert-base-uncased')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves',',','.','?',"!","n't"]
filter_words = list(set(filter_words))

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_masked(words,text):
    len_text = len(words)
    masked_words = []
    for i in text:
        a = words[0:i] + ['[MASK]'] + words[i +1:]
        #print(len(a),a)
        a = " ".join(a)
        masked_words.append([a,i])
    # list of words
    #print(len_text,len(masked_words[0][0]))
    return masked_words

def bert_synonym_list(seq_mask,k,token,text_ls,stop,sim_score_threshold):
    aa = seq_mask.split(" ")
    if token>220:
        if token+110>len(aa):
            seq_mask =" ".join(aa[token-110:])
        else:
            seq_mask = " ".join(aa[token-110:token+110])
    else:
        if token-110<0:
            seq_mask = " ".join(aa[:220])
        else:
            seq_mask = " ".join(aa[token-110:token+110])
    a = unmasker(seq_mask, top_k=k)
    
    word = []
    for i in a:
        tmp_text = text_ls.copy()
        tmp_text[token] = i['token_str']
        if semantic_sim(" ".join(text_ls)," ".join(tmp_text))>sim_score_threshold:
            word.append(i['token_str'])

    return word

def bert_get_synonym_list(text,k,text_ls,stop,sim_score_threshold):
    #result = copy.deepcopy(tmp)
    result =[]
    mask = get_masked(text_ls,text)
    #print(mask[0][0])
    #print(mask)
    for a in mask:
        tmp = []
        tmp = bert_synonym_list(a[0],k,a[1],text_ls,stop,sim_score_threshold)
        result.append(tmp)
   
    return result
        

#USE = hub.load('USE')
USE = hub.load('https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder/3.tar.gz')
def semantic_sim(text1, text2):
    a = USE([text1,text2])
    simi = 1.0 - pdist([a['outputs'][0], a['outputs'][1]], 'cosine') # 余弦相似度
    return simi.item()


# loading model
class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 nclasses,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=nclasses)
        self.model = self.model.to(device)

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data, batch_size=32):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data, batch_size=batch_size)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, text_a) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data, batch_size=32):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(data,
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        return eval_dataloader
    
def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def csim_matrix(lst_revs ,embeddings, word2idx_vocab):
    ''' Create a cosine similarity matrix only for words in reviews

    Identify all the words in the reviews that are not stop words
    Use embedding matrix to create a cosine similarity submatrix just for these words
    Columns of this cosine similarity matrix correspond to the original words in the embedding
    rows of the matrix correspond to the words in the reviews
    '''
    reviews = [d[0] for d in lst_revs]
    all_words = set()

    for r in reviews:
        all_words = all_words.union(set([str(word) for word in r if str(word) in word2idx_vocab and str(word) not in stopwords.words('english')]))

    word2idx_rev={}
    idx2word_rev={}
    p=0
    embeddings_rev_words=[]
    for word in all_words:
        word2idx_rev[str(word)] = p
        idx2word_rev[p]=str(word)
        p+=1
        embeddings_rev_words.append(embeddings[word2idx_vocab[str(word)]])

    embeddings_rev_words=np.array(embeddings_rev_words)
    cos_sim = np.dot(embeddings_rev_words, embeddings.T)

    return cos_sim, word2idx_rev,idx2word_rev

def get_synonym_list(ori_text, stop_words_set, cos_sim, idx2word_vocab, word2idx_rev, synonym_num=50, synonym_sim=0.5):
    text_len = len(ori_text)
    synonym_list = [None]*text_len
    pos_ls = criteria.get_pos(ori_text)
    for wd_i in range(len(ori_text)):
        _text = ori_text.copy()
        wd = ori_text[wd_i]
        if wd in stop_words_set or wd not in word2idx_rev:
            synonym_list[wd_i] = []
        else:
            synonym_words, _ = pick_most_similar_words_batch([word2idx_rev[wd]], cos_sim, idx2word_vocab, synonym_num, synonym_sim)
            new_texts = [_text[:wd_i]+[syn]+_text[1+wd_i:] for syn in synonym_words[0]]
            synonyms_pos_ls = [criteria.get_pos(tmp_text[max(wd_i - 4, 0):wd_i + 5])[min(4, wd_i)] \
                                                   if text_len > 10 else criteria.get_pos(tmp_text)[wd_i] for tmp_text in new_texts]
            pos_mask = np.array(criteria.pos_filter(pos_ls[wd_i], synonyms_pos_ls))
            yes_syn = [synonym_words[0][i] for i in range(len(synonym_words[0])) if pos_mask[i]]
            synonym_list[wd_i] = yes_syn
    synonym_len = [len(i) for i in synonym_list]
    w_select_probs = np.array(synonym_len)/sum(synonym_len)
    return synonym_list, w_select_probs

def random_select(aa,k):
    a = aa.copy()
    b = []
    for i in range(k):
        tmp = torch.rand((1,1)).item()
    #---
        if tmp<0.5:
            b.append(a[0])
            a.pop(0)
        else:
            tmp = torch.randint(0,len(a),(1,1)).item()
            b.append(a[tmp])
            a.pop(tmp)
    #---
    return b

def issuccess(predictor,batch_size,all_text_ls,text_ls,true_label,num_queries,k=2):
    tmp_prob = []
    success_text_ls = []
    #batch 优化
    _probs = predictor(all_text_ls, batch_size=batch_size)
    _probs_argmax = torch.argmax(_probs, dim=-1).cpu().numpy()
    num_queries += len(all_text_ls)
    np_true_label = np.array([true_label]*len(all_text_ls))
    success_text_ls = list(np.array(all_text_ls)[np_true_label!=_probs_argmax])

    
    if len(success_text_ls) == 0:
        tmp_prob,_ = _probs.max(dim=-1)
        a,idx1 = torch.sort(torch.tensor(tmp_prob))
        k_text_ls = []
        tmp_k = []
        for i in idx1:
            k_text_ls.append(all_text_ls[i])
        if len(k_text_ls)<2000:
            tmp_k = k_text_ls
        else:
            #tmp_k = random_select(k_text_ls,2000)
            tmp_k = k_text_ls

        if len(idx1)>k:
            k_text_ls =  k_text_ls[:k]
        return [0,k_text_ls,tmp_k],num_queries

    semantic = []
    for i in success_text_ls:
        semantic.append(semantic_sim(" ".join(i)," ".join(text_ls)))
    return [1,text_ls,num_queries,success_text_ls[semantic.index(max(semantic))]],num_queries


def auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,words_perturb_idx,start,synonym_words,k=2):
    new_tmp = []
    tmp = []
    for i in range(k):
        if i==0:
            tmp = k_text_ls
        else:
            tmp = new_tmp
            new_tmp = []
        for text in tmp:
            #new_tmp = []
            new_tmp.append(text)

            for j in synonym_words[start+i]:
                aa = text.copy()
                aa[words_perturb_idx[start+i]]  = j
                new_tmp.append(aa)

        suc,num_queries = issuccess(predictor,batch_size,new_tmp,text_ls,true_label,num_queries,k)
        if suc[0] == 1:
            return suc,num_queries
        new_tmp = suc[1]

    return suc,num_queries
#===================================attack=============================================
def attack(text_id,fail,text_ls,true_label,predictor,stop_word_set,word2idx, idx2word, cos_sim,
synonyms_num=50,sim_score_threshold=0, batch_size=32,import_score_threshold=0.005,pos=1,k=2):
    less = 0
    len_text = len(text_ls)

    num_queries = 0
    orig_probs = predictor([text_ls]).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()
    if true_label != orig_label:
        fail[text_id] = 0
        return  [2, orig_label, orig_label, 0]
    else:

        #print(len_text)
        #==============================word importance rank
        #采用MASK
        #leave_1_texts = [text_ls[:ii] + ['[MASK]'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        #采用oov
        leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        #delete
        #leave_1_texts = [text_ls[:ii]  + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
        num_queries += len(leave_1_texts)
        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                    leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                      leave_1_probs_argmax))).data.cpu().numpy()
        
        #print(import_scores)
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):

            if score > import_score_threshold and text_ls[idx] not in stop_word_set:
                words_perturb.append((idx, text_ls[idx]))

        #words_perturb_idx = [idx for idx, word in words_perturb if word in word2idx]
        word_perturb_text_idx = [idx for idx, word in words_perturb]
        word_list = [word for idx, word in words_perturb]

        synonym_words = []
        #print(word_list)
        for i in range(len(word_list)):
            #if word_list[i] in word2idx:
            #    t , _ = pick_most_similar_words_batch( [word2idx[word_list[i]]], cos_sim, idx2word, synonyms_num,sim_score_threshold)
            #    synonym_words.append(t[0])
            #else:
                #t = bert_get_synonym_list([word_perturb_text_idx[i]],synonyms_num,text_ls,filter_words,sim_score_threshold)
                #synonym_words.append(t[0])
            t = bert_get_synonym_list([word_perturb_text_idx[i]],synonyms_num,text_ls,filter_words,sim_score_threshold)
            synonym_words.append(t[0])

       
        tmp_syn = []
        rank_synonyms_word = []
        new_word_1 =[]
        new_word_2 = []
        for i in range(len(word_perturb_text_idx)):
            if synonym_words[i]!=[]:
                #new_word_1.append(words_perturb_idx[i])
                new_word_2.append(word_perturb_text_idx[i])
                new_word_1.append(word_list[i])
                tmp_syn.append(synonym_words[i])
                
        #words_perturb_idx = new_word_1
        word_perturb_text_idx = new_word_2
        word_list = new_word_1
        synonym_words = tmp_syn
        
        if len(word_perturb_text_idx)==0 or len(synonym_words) == 0:
            #print("4")
            fail[text_id] = 4
            return [0]
        if pos == 1:
        ###pos check
            pos_synonym_words = []
            pos_ls = criteria.get_pos(word_list)
            #print("before:{}".format(synonym_words))
            for i in range(len(synonym_words)):
                ttmp = []
                tmp = criteria.get_pos(synonym_words[i])
                result = criteria.pos_filter(pos_ls[i],tmp)
                for j in range(len(result)):
                    if result[j]==True:
                        ttmp.append(synonym_words[i][j])
                pos_synonym_words.append(ttmp)
            synonym_words = pos_synonym_words

            tmp_syn = []
            rank_synonyms_word = []
            new_word_1 =[]
            new_word_2 = []
            for i in range(len(word_perturb_text_idx)):
                if synonym_words[i]!=[]:
                    #new_word_1.append(words_perturb_idx[i])
                    new_word_2.append(word_perturb_text_idx[i])
                    
                    tmp_syn.append(synonym_words[i])
                    
            #words_perturb_idx = new_word_1
            word_perturb_text_idx = new_word_2
            synonym_words = tmp_syn

        #======================beam search
        if len(word_perturb_text_idx)==0 or len(synonym_words) == 0:
            #print("4")
            fail[text_id] = 4
            return [0]
            
        if len(word_perturb_text_idx)<k:
            suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,[text_ls],word_perturb_text_idx,0,synonym_words,k=len(word_perturb_text_idx))
            if suc[0] == 0:
                #print("1")
                fail[text_id] = 1
            return suc

        suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,[text_ls],word_perturb_text_idx,0,synonym_words,k=k)
        if suc[0] == 1:
            return suc
        k_text_ls = suc[1]
        for start in range(k,len(word_perturb_text_idx),k):
            if  len(word_perturb_text_idx)<start+k:#需要根据k来调整
                #for _ in k_text_ls:
                suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,word_perturb_text_idx,start,synonym_words,k=len(word_perturb_text_idx)-start)
                if suc[0] == 1:
                    return suc
                #print("2")
                fail[text_id] = 2
                return [0]

            suc,num_queries = auto_beamsearch(predictor,batch_size,text_ls,true_label,num_queries,k_text_ls,word_perturb_text_idx,start,synonym_words,k=k)
            if suc[0] == 1:
                return suc
            k_text_ls = suc[1]
        #print("3")
        fail[text_id] = 3
        return [0]

def test_beam(a):
    sim = [0]
    change_rates = [0]
    count = 0
    change_num = 0
    change_all = 0
    query = [0]
    for i in a:
        if i[0]!=0:
            count+=1
        if i[0] ==1:
            sim.append(semantic_sim(" ".join(i[-1])," ".join(i[1])))
            change_num += np.sum(np.array(i[-1]) != np.array(i[1]))
            change_all+=len(i[1])
            change_rates.append(np.sum(np.array(i[-1]) != np.array(i[1])) / len(i[1]))
            query.append(i[2])
    print("acc:{},sim:{},change_rate:{},new_change_rate:{},query:{}".format(count/len(a),np.mean(sim),np.mean(change_rates),change_num/change_all,np.mean(query)))


def run_attack():
    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        default='data/mr',
                        # required=True,
                        help="Which dataset to attack.")  # TODO

    parser.add_argument("--target_model",
                        type=str,
                        default='bert',
                        # required=True,
                        choices=['wordLSTM', 'bert', 'wordCNN'],
                        help="Target models for text classification: wordcnn, word Lstm ")   # TODO

    parser.add_argument("--nclasses",
                        type=int,
                        default=2,
                        help="How many classes for classification.")

    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='word_embeddings_path/glove.6B.200d.txt',
                        help="path to the word embeddings for the target model, CNN and LSTM is need")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        required=False,
                        default="counter_fitting_embedding/counter-fitted-vectors.txt",
                        help="path to the counter-fitting embeddings used to find synonyms")

    parser.add_argument("--sim",
                        type=int,
                        default=0.5,
                        help="sim threshold")


    parser.add_argument("--syn_num",
                        type=int,
                        default=50,
                        help="syn num")

    parser.add_argument("--k",
                        type=int,
                        default=10,
                        help="beamsearch")

    parser.add_argument("--pos",
                            type=int,
                            default=1,
                            help="beamsearch")

    args = parser.parse_args(args=[])
    #print(args)
    texts, labels = dataloader.read_corpus(args.dataset_path)

    data = list(zip(texts, labels))

    embeddings, word2idx_vocab, idx2word_vocab = generate_embedding_mat(args.counter_fitting_embeddings_path)

    cos_sim, word2idx_rev, idx2word_rev = csim_matrix(data, embeddings, word2idx_vocab)

    # Load the saved model using state dic
    if args.target_model == "wordCNN":
        default_model_path = "saved_models/cnn/"
        if 'imdb' in args.dataset_path:
            default_model_path += 'imdb'
        elif 'mr' in args.dataset_path:
            default_model_path += 'mr'
        elif 'yelp' in args.dataset_path:
            default_model_path += 'yelp'

        model = Model(args.word_embeddings_path, nclasses=args.nclasses, hidden_size=100, cnn=True).to(device)
        checkpoint = torch.load(default_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)

    elif args.target_model == "wordLSTM":
        default_model_path = "saved_models/lstm/"
        if 'imdb' in args.dataset_path:
            default_model_path += 'imdb'
        elif 'mr' in args.dataset_path:
            default_model_path += 'mr'
        elif 'yelp' in args.dataset_path:
            default_model_path += 'yelp'

        model = Model(args.word_embeddings_path, nclasses=args.nclasses).cuda()
        checkpoint = torch.load(default_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint)

    elif args.target_model == "bert":
        default_model_path = "saved_models/bert/"
        if 'imdb' in args.dataset_path:
            default_model_path += 'imdb'
            max_seq_length = 256
        elif 'mr' in args.dataset_path:
            default_model_path += 'mr'
            max_seq_length = 128
        elif 'yelp' in args.dataset_path:
            default_model_path += 'yelp'
            max_seq_length = 256
        model = NLI_infer_BERT(default_model_path, nclasses=args.nclasses, max_seq_length=max_seq_length)
    predictor = model.text_pred
    # predictor([data[0][0]]).squeeze() #　tensor([0.0181, 0.9819], device='cuda:0')
    print('Start attacking...')

    starttime = datetime.datetime.now()
    orig_texts = []
    #second_table = load_obj('yelp_add_exam/lstm_2000')
    AAAI_result = list() # AAAI 结果--
    ours_result = list()
    fail  = {}
    print("model:{},data:{}".format(args.target_model,args.dataset_path))
    print("para->sim:{},synonyms_num:{},k:{}".format(args.sim,args.syn_num,args.k))
    for idx, (text, true_label) in enumerate(tqdm.tqdm(data[:1000])):
        if idx % 5 == 0:
            cc = 1
            #print('*', end = ' ')
        if idx % 100 == 0 and idx != 0:
            test_beam(AAAI_result)
    #     print(idx, text)
        orig_texts.append(text)
        
        re = attack(idx,fail,text, true_label, predictor, filter_words, word2idx_rev, idx2word_vocab, cos_sim,
                    import_score_threshold=-1, sim_score_threshold=args.sim, synonyms_num=args.syn_num,batch_size=64,pos = args.pos, k=args.k)
                    
        AAAI_result.append(re)

    endtime = datetime.datetime.now()
    print("time:{}".format((endtime - starttime).seconds / 60))
    test_beam(AAAI_result)
    AAAI_result.append([(endtime - starttime).seconds / 60])
    AAAI_result.append(fail)
    datasetname = args.dataset_path.split("/")[-1]
    if args.pos  == 1:
        save_name = './bs/pos/' + args.target_model+"-"+datasetname+"-"+str(args.sim)+"-"+str(args.syn_num)+"-"+str(args.k)
    else:
        save_name = './bs/' + args.target_model+"-"+datasetname+"-"+str(args.sim)+"-"+str(args.syn_num)+"-"+str(args.k)
    save_obj(AAAI_result,save_name)

if __name__ == "__main__":
    run_attack()