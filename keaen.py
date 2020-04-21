# -*- coding: utf-8 -*-
# file: aen.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
import torch
import torch.nn as nn
import torch.nn.functional as F


# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class AEN_GloVe(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AEN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding()

        self.attn_k = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim * 3, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, target_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_indices != 0, dim=-1)
        target_len = torch.sum(target_indices != 0, dim=-1)
        context = self.embed(text_raw_indices)
        context = self.squeeze_embedding(context, context_len)
        target = self.embed(target_indices)
        target = self.squeeze_embedding(target, target_len)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        return out


class GKEAEN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(GKEAEN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q2 = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                 dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_k = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_s2 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim * 4, opt.polarities_dim)

    def forward(self, inputs):
        context, target, knowledge = inputs[0], inputs[1], inputs[2]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        knowledge_len = torch.sum(knowledge != 0, dim=-1)

        context = self.squeeze_embedding(context, context_len)
        context, _ = self.bert(context)
        context = self.dropout(context)

        target = self.squeeze_embedding(target, target_len)
        target, _ = self.bert(target)
        target = self.dropout(target)

        knowledge = self.squeeze_embedding(knowledge, knowledge_len)
        knowledge, _ = self.bert(knowledge)
        knowledge = self.dropout(knowledge)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)
        hk, _ = self.attn_q2(context, knowledge)
        hk = self.ffn_k(hk)

        s1, _ = self.attn_s1(hc, ht)
        s2, _ = self.attn_s2(hc, hk)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)
        knowledge_len = torch.tensor(knowledge_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))
        s2_mean = torch.div(torch.sum(s2, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean, s2_mean), dim=-1)
        out = self.dense(x)
        return out

class NKEAEN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(NKEAEN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q2 = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                 dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_k = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_s2 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim * 4, opt.polarities_dim)

    def forward(self, inputs):
        context, target, knowledge = inputs[0], inputs[1], inputs[2]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        knowledge_len = torch.sum(knowledge != 0, dim=-1)

        context = self.squeeze_embedding(context, context_len)
        context, _ = self.bert(context)
        context = self.dropout(context)

        target = self.squeeze_embedding(target, target_len)
        target, _ = self.bert(target)
        target = self.dropout(target)

        knowledge = self.squeeze_embedding(knowledge, knowledge_len)
        knowledge, _ = self.bert(knowledge)
        knowledge = self.dropout(knowledge)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)
        hk, _ = self.attn_q2(context, knowledge)
        hk = self.ffn_k(hk)

        s1, _ = self.attn_s1(hc, ht)
        s2, _ = self.attn_s2(hc, hk)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)
        knowledge_len = torch.tensor(knowledge_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))
        s2_mean = torch.div(torch.sum(s2, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean, s2_mean), dim=-1)
        out = self.dense(x)
        return out

class GNKEAEN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(GNKEAEN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)

        self.attn_k = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q2 = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                 dropout=opt.dropout)
        self.attn_q3 = Attention(opt.bert_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                 dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_k = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_s2 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)
        self.attn_s3 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim * 5, opt.polarities_dim)

    def forward(self, inputs):
        context, target, g_knowledge, n_knowledge = inputs[0], inputs[1], inputs[2], inputs[3]
        context_len = torch.sum(context != 0, dim=-1)
        target_len = torch.sum(target != 0, dim=-1)
        g_knowledge_len = torch.sum(g_knowledge != 0, dim=-1)
        n_knowledge_len = torch.sum(n_knowledge != 0, dim=-1)

        context = self.squeeze_embedding(context, context_len)
        context, _ = self.bert(context)
        context = self.dropout(context)

        target = self.squeeze_embedding(target, target_len)
        target, _ = self.bert(target)
        target = self.dropout(target)

        g_knowledge = self.squeeze_embedding(g_knowledge, g_knowledge_len)
        g_knowledge, _ = self.bert(g_knowledge)
        g_knowledge = self.dropout(g_knowledge)

        n_knowledge = self.squeeze_embedding(n_knowledge, n_knowledge_len)
        n_knowledge, _ = self.bert(n_knowledge)
        n_knowledge = self.dropout(n_knowledge)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)
        hg_k, _ = self.attn_q2(context, g_knowledge)
        hg_k = self.ffn_k(hg_k)
        hn_k, _ = self.attn_q3(context, n_knowledge)
        hn_k = self.ffn_k(hn_k)

        s1, _ = self.attn_s1(hc, ht)
        s2, _ = self.attn_s2(hc, hg_k)
        s3, _ = self.attn_s3(hc, hn_k)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)
        g_knowledge_len = torch.tensor(g_knowledge_len, dtype=torch.float).to(self.opt.device)
        n_knowledge_len = torch.tensor(n_knowledge_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))
        s2_mean = torch.div(torch.sum(s2, dim=1), context_len.view(context_len.size(0), 1))
        s3_mean = torch.div(torch.sum(s3, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean, s2_mean, s3_mean), dim=-1)
        out = self.dense(x)
        return out