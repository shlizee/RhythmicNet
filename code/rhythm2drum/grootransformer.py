"""
RhyhmicNet stage 2 Rhythm2Drum (generating drum hits) code of "How does it sound?: Generation of Rhythmic
Soundtracks for Human Movement Videos" 
Copyright (c) 2021-2022 University of Washington. Developed in UW NeuroAI Lab by Xiulong Liu.
"""


import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from third_party.transformer.modules import Encoder, Decoder, get_subsequent_mask
import math


class GrooTransformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab=1, n_trg_vocab=152,
            d_word_vec_enc=64, d_word_vec_dec=256,
            n_layers=3, n_head=4, dropout=0.1, n_position_enc=100, n_position_dec=100, mask_zero=False):

        super().__init__()
        self.word_emb_sz = d_word_vec_dec
        self.tap_encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position_enc,
            d_word_vec=d_word_vec_enc, d_model=d_word_vec_enc, d_inner=d_word_vec_enc * 2,
            n_layers=n_layers, n_head=n_head, dropout=dropout)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position_dec,
            d_word_vec=d_word_vec_dec, d_model=d_word_vec_dec, d_inner=d_word_vec_dec * 2,
            n_layers=n_layers, n_head=n_head, dropout=dropout)

        self.trg_word_prj = nn.Linear(d_word_vec_dec, n_trg_vocab, bias=False)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.mask_zero = mask_zero

        # self.x_logit_scale = 1.
        # if trg_emb_prj_weight_sharing:
        #     # Share the weight between target word embedding & last dense layer
        #     self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
        #     self.x_logit_scale = (d_word_vec_dec ** -0.5)


    def forward(self, tap, trg_seq, targets=None, return_attn=False):

        trg_mask = get_subsequent_mask(trg_seq)

        enc_output, *_ = self.tap_encoder(tap)
        if return_attn:
            dec_output, dec_slf_attn_list, dec_enc_attn_list = self.decoder(trg_seq, trg_mask, enc_output, return_attns=return_attn)
        else:
            dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output)
        seq_logit = self.trg_word_prj(dec_output)
        seq_logit = seq_logit.view(-1, seq_logit.size(-1))
        # if we are given some desired targets also calculate the loss
        if targets is not None:
            if self.mask_zero:
                loss = F.cross_entropy(seq_logit, targets.view(-1), ignore_index=1) # 1 corresponds to '000000000'
            else:
                loss = F.cross_entropy(seq_logit, targets.view(-1))
            if return_attn:
                return seq_logit, loss, dec_slf_attn_list, dec_enc_attn_list
            else:
                return seq_logit, loss
        if return_attn:
            return seq_logit, dec_slf_attn_list, dec_enc_attn_list
        return seq_logit

    def sample(self, tap, max_steps, sample=True):
        enc_output, *_ = self.tap_encoder(tap)
        token_embed_cat = torch.zeros(tap.shape[0], 1, self.word_emb_sz).cuda()
        indices = []

        for i in range(max_steps):
            dec_out = self.decoder.sample(enc_output=enc_output, token_embed_cat=token_embed_cat)
            logits = self.trg_word_prj(dec_out)[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # if self.mask_zero:
            #     for m in range(ix.shape[0]):
            #         if tap[m, i] == 0:
            #             ix[m] = 1

            emb_ix = self.decoder.trg_word_emb(ix)
            if self.mask_zero:
                for m in range(ix.shape[0]):
                    if tap[m, i] == 0:
                        ix[m] = 1
            indices.append(ix)
            token_embed_cat = torch.cat((token_embed_cat, emb_ix), dim=1)
        indices = torch.cat(indices, dim=1)
        return indices


if __name__ == "__main__":
    model = GrooTransformer(n_src_vocab=2, n_trg_vocab=152, d_word_vec_enc=128, d_word_vec_dec=128)
    model.cuda()
    batch_size = 64
    tap_inp = torch.LongTensor(batch_size, 32).random_() % 2
    dec_inp = torch.LongTensor(batch_size, 31).random_() % 152
    logits = model(tap_inp.cuda(), dec_inp.cuda())
    print(logits, logits.shape)