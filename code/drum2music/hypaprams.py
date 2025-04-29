"""
RhyhmicNet Drum2Music stage hyperparameter code of "How does it sound?: Generation of Rhythmic
Soundtracks for Human Movement Videos" 
Copyright (c) 2021-2022 University of Washington. Developed in UW NeuroAI Lab by Xiulong Liu and Kun Su.
"""


pickle_dir = '/home/neuroai/encoder_decoder_transformer_xl/remi_data/new_emb_256_256_drums_piano_data.pickle' # bass
vocab_size = 308

END_TOKEN = 308
PAD_TOKEN = 308 + 1
drums_len = 256  # segment max length

bass_len = 256 # piano
enc_mem_len = 256  # memeory length
enc_n_layer = 4  # number of layers
enc_d_embed = 512  # number of embedding size
enc_d_model = 512  # number of hidden
enc_dropout = 0.1
enc_n_head = 8  # number of head
enc_d_head = enc_d_model // enc_n_head
enc_d_ff = 2048  # positionwise inner linear number
dec_mem_len = 256  # memeory length, piano
dec_n_layer = 8  # number of layers
dec_d_embed = 512  # number of embedding size
dec_d_model = 512  # number of hidden
dec_dropout = 0.1
dec_n_head = 8  # number of head
dec_d_head = enc_d_model // enc_n_head
dec_d_ff = 2048  # positionwise inner linear number

learning_rate = 0.0002
epochs = 1000
batch_size = 16 # piano
print_interval = 10
eval_interval = 30
max_step = 400000
eval_max_steps = 10
