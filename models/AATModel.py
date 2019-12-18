# Implementation for paper 'Adaptively Aligned Image Captioning via Adaptive Attention Time'
# https://arxiv.org/abs/1909.09060

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import misc.utils as utils

from .AttModel import AttModel, Attention
from .TransformerModel import LayerNorm
from .AoAModel import MultiHeadedDotAttention

class AATCore(nn.Module):
    def __init__(self, opt):
        super(AATCore, self).__init__()
        
        self.drop_prob_lm = opt.drop_prob_lm
        self.rnn_size = opt.rnn_size
        self.epsilon = opt.epsilon
        self.max_att_steps = opt.max_att_steps
        self.use_multi_head = opt.use_multi_head

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size)
        
        self.confidence = nn.Sequential(nn.Linear(opt.rnn_size, opt.rnn_size),
                                    nn.ReLU(),
                                    nn.Linear(opt.rnn_size, 1),
                                    nn.Sigmoid())

        self.h2query = nn.Sequential(nn.Linear(opt.rnn_size * 2, opt.rnn_size),
                            nn.ReLU())

        # if opt.use_multi_head == 1: # TODO, not implemented for now           
        #     self.attention = MultiHeadedAddAttention(opt.num_heads, opt.d_model, scale=opt.multi_head_scale)
        if opt.use_multi_head == 2:
            self.attention = MultiHeadedDotAttention(opt.num_heads, opt.rnn_size, project_k_v=0, scale=opt.multi_head_scale, use_output_layer=0, do_aoa=0, norm_q=1)
        else:            
            self.attention = Attention(opt)

        self.lang_lstm = nn.LSTMCell(opt.rnn_size + opt.rnn_size, opt.rnn_size)

        self.norm_h = LayerNorm(opt.rnn_size)
        self.norm_c = LayerNorm(opt.rnn_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        batch_size = fc_feats.size()[0]
        accum_conf = Variable(fc_feats.data.new(batch_size, 1).zero_())
        self.att_step = Variable(fc_feats.data.new(batch_size).zero_())
        self.att_cost = Variable(fc_feats.data.new(batch_size).zero_())

        h_lang = Variable(fc_feats.data.new(batch_size, self.rnn_size).zero_())
        c_lang = Variable(fc_feats.data.new(batch_size, self.rnn_size).zero_())

        att_lstm_input = torch.cat([fc_feats + state[0][-1], xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))
        h_att = self.norm_h(h_att)

        p = self.confidence(h_att)

        self.att_cost += (1+(1-p)).squeeze(1)
        
        selector = (p < 1 - self.epsilon).data

        if selector.any():
            accum_conf += p
            
            h_lang += p * h_att
            c_lang += p * state[1][1]

            h_lang_, c_lang_ = (state[0][1], state[1][1])

            for i in range(self.max_att_steps):
                att_query = self.h2query(torch.cat([h_lang_, h_att], 1))
                if self.use_multi_head == 2:
                    att_ = self.attention(att_query, p_att_feats.narrow(2, 0, self.rnn_size), p_att_feats.narrow(2, self.rnn_size, self.rnn_size), att_masks)
                else:
                    att_ = self.attention(att_query, att_feats, p_att_feats, att_masks)

                lang_lstm_input_ = torch.cat([att_, att_query], 1)
                h_lang_, c_lang_ = self.lang_lstm(lang_lstm_input_, (h_lang_, c_lang_))
                h_lang_ = self.norm_h(h_lang_)
                c_lang_ = self.norm_c(c_lang_)

                self.att_step += selector.squeeze(1).float()
                p_ = self.confidence(h_lang_)
          
                beta = p_ * (1 - accum_conf)
                accum_conf += beta * selector.float()
                h_lang += beta * h_lang_ * selector.float()
                c_lang += beta * c_lang_ * selector.float()

                self.att_cost += ((1+(i+2)*(1-p_)) * selector.float()).squeeze(1)
                selector = (accum_conf < 1 - self.epsilon).data * selector

                if not selector.any():
                    break

            h_lang /= accum_conf
            c_lang /= accum_conf

        else:
            h_lang += h_att
            c_lang += state[1][1]

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state

class AATModel(AttModel):
    def __init__(self, opt):
        super(AATModel, self).__init__(opt)
        self.num_layers = 2
        if opt.use_multi_head == 2:
            del self.ctx2att
            self.ctx2att = nn.Linear(opt.rnn_size, 2 * opt.rnn_size)
        self.core = AATCore(opt)
        
    def forward(self, *args, **kwargs):
        self.all_att_step = []
        self.all_att_cost = []
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_'+mode)(*args, **kwargs)
    
    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        
        self.all_att_step.append(self.core.att_step.cpu().numpy())
        self.all_att_cost.append(self.core.att_cost)

        return logprobs, state
