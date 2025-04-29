"""
Drum2Music model definition of "How does it sound?: Generation of Rhythmic
Soundtracks for Human Movement Videos" 
Copyright (c) 2021-2022 University of Washington. Developed in UW NeuroAI Lab by Kun Su and Xiulong Liu.
"""

import torch.nn as nn
import drum2music.hypaprams as hyp
import drum2music.initialization as init
from third_party.transformer_xl.modules import MemTransformerLM, CrossAttentionMemTransformerLM, NormalEmbedding
from third_party.transformer_xl.proj_adaptive_softmax import NormalSoftmax


class Encoder_Decoder_TM_XL(nn.Module):
    def __init__(self, initialize = True):
        super(Encoder_Decoder_TM_XL, self).__init__()
        self.word_emb_1 = NormalEmbedding(n_token = hyp.vocab_size + 2, d_embed = hyp.enc_d_embed, d_proj = hyp.enc_d_model)
        self.word_emb_2 = NormalEmbedding(n_token = hyp.vocab_size + 2, d_embed = hyp.enc_d_embed, d_proj = hyp.enc_d_model)
        self.pos_emb = NormalEmbedding(n_token = hyp.pos_size, d_embed = hyp.enc_d_embed, d_proj = hyp.enc_d_model)
        # self.indep_emb = NormalEmbedding(n_token = hyp.indep_size, d_embed = hyp.enc_d_embed, d_proj = hyp.enc_d_model)
        self.crit = NormalSoftmax(n_token= hyp.vocab_size + 2, d_embed = hyp.enc_d_embed, d_proj=hyp.enc_d_model)
        # if hyp.tie_weight:
        #     for i in range(len(self.crit.out_layers)):
        #         self.crit.out_layers[i].weight = self.word_emb_2.emb_layers[i].weight

        self.encoder = MemTransformerLM(
            n_token=hyp.vocab_size + 2,
            n_layer=hyp.enc_n_layer,
            n_head=hyp.enc_n_head,
            d_model=hyp.enc_d_model,
            d_head=hyp.enc_d_head,
            d_inner=hyp.enc_d_ff,
            dropout=hyp.enc_dropout,
            dropatt=hyp.enc_dropout,
            d_embed=hyp.enc_d_embed,
            pre_lnorm=True,
            tgt_len=hyp.drums_len,
            ext_len=0,
            mem_len= hyp.enc_mem_len,
            attn_type=0)
        self.decoder = CrossAttentionMemTransformerLM(
            n_token=hyp.vocab_size + 2,
            n_layer=hyp.dec_n_layer,
            n_head=hyp.dec_n_head,
            d_model=hyp.dec_d_model,
            d_head=hyp.dec_d_head,
            d_inner=hyp.dec_d_ff,
            dropout=hyp.dec_dropout,
            dropatt=hyp.dec_dropout,
            d_embed=hyp.dec_d_embed,
            pre_lnorm=True,
            tgt_len=hyp.bass_len, #128,
            ext_len=0,
            mem_len= hyp.dec_mem_len, #128,
            attn_type=4)
        if initialize:
            self.word_emb_1.apply(init.weights_init)
            self.word_emb_2.apply(init.weights_init)
            self.pos_emb.apply(init.weights_init)
            # self.indep_emb.apply(init.weights_init)
            self.encoder.apply(init.weights_init)
            # self.encoder.word_emb.apply(init.weights_init)
            self.decoder.apply(init.weights_init)
            # self.decoder.word_emb.apply(init.weights_init)

    def forward(self, drums, bass_input, bass_target,
                drums_pos, bass_pos, #drums_indep, bass_indep,
                total_drums_mask, current_drums_mask, total_bass_mask, current_bass_mask,
                enc_mem, dec_mem):

        drums_word_emb = self.word_emb_1(drums)

        drums_pos_emb = self.pos_emb(drums_pos)

        # drums_indep_emb = self.indep_emb(drums_indep)
        # print("start encoder!")
        enc_ret = self.encoder(drums_word_emb, drums_pos_emb, total_drums_mask, enc_mem)
        # print()
        enc_out, enc_mems = enc_ret[0], enc_ret[1:]
        # print(enc_out.shape)
        #enc_out shape: T, B, N
        # print("start decoder!")
        bass_word_emb = self.word_emb_2(bass_input)
        bass_pos_emb = self.pos_emb(bass_pos)
        # bass_indep_emb = self.indep_emb(bass_indep)
        dec_ret = self.decoder(bass_word_emb, bass_pos_emb, enc_out, total_bass_mask, current_drums_mask, dec_mem)
        # print("done with decoder!")
        # print()
        hidden, dec_mems = dec_ret[0], dec_ret[1:]
        tgt_len = bass_target.size(0)
        pred_hid = hidden[-tgt_len:]
        loss = self.crit(pred_hid.reshape(-1, pred_hid.size(-1)), bass_target.reshape(-1))
        loss = loss.view(-1)

        old_drums_mask = current_drums_mask
        old_bass_mask = current_bass_mask
        # print(loss)
        return [loss, enc_mems, dec_mems, old_drums_mask, old_bass_mask]

    def encode(self, drums, drums_pos, total_drums_mask, enc_mem):
        drums_word_emb = self.word_emb_1(drums)
        drums_pos_emb = self.pos_emb(drums_pos)
        # drums_indep_emb = self.indep_emb(drums_indep)
        enc_ret = self.encoder(drums_word_emb, drums_pos_emb, total_drums_mask, enc_mem)

        return enc_ret

    def forward_generate(self, bass_input, bass_pos,  enc_out, current_drums_mask, dec_mem):
        tgt_len = bass_input.size(0)
        batch_size = bass_input.size(1)
        bass_word_emb = self.word_emb_2(bass_input)
        bass_pos_emb = self.pos_emb(bass_pos)
        # bass_indep_emb = self.indep_emb(bass_indep)
        dec_ret = self.decoder.forward_generate(bass_word_emb, bass_pos_emb, enc_out, current_drums_mask, dec_mem)
        hidden, dec_mems = dec_ret[0], dec_ret[1:]
        pred_hid = hidden[-tgt_len:]

        logits = self.crit._compute_logits(
            pred_hid.view(-1, pred_hid.size(-1)),
            self.crit.out_layers[0].weight,
            self.crit.out_layers[0].bias,
            None)
        logits = logits.view(tgt_len, batch_size, -1)

        old_drums_mask = current_drums_mask
        return [logits, dec_mems, old_drums_mask]




