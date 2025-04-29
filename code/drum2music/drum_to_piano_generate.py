"""
RhyhmicNet Drum2Music stage chord module of "How does it sound?: Generation of Rhythmic
Soundtracks for Human Movement Videos" 
Copyright (c) 2021-2022 University of Washington. Developed in UW NeuroAI Lab by Kun Su and Xiulong Liu.
"""


import third_party.remi.original_utils as utils
import numpy as np
import pickle
import drum2music.hypaprams as hyp
import torch
import os
import pretty_midi
from drum2music.encoder_decoder_trasnformer_model import Encoder_Decoder_TM_XL


if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')

def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)

###############################################################################
# Load data
###############################################################################
event2word, word2event = pickle.load(open('dictionary.pkl', 'rb'))
independent = []
pos = []
for key, value in event2word.items():
    print(key, value)
    event_name, event_value = word2event.get(value).split('_')
    if event_name == 'Position':
        pos.append(value)
    if event_name == 'Note On':
        independent.append(value)

END_TOKEN = hyp.vocab_size
PAD_TOKEN = hyp.vocab_size + 1

def extract_events(input_path):
    note_items, tempo_items = utils.read_items(input_path)
    note_items = utils.quantize_items(note_items)
    max_time = note_items[-1].end
    items = tempo_items + note_items
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)
    words = []
    for bar in events:
        bar_ = []
        for event in bar:
            e = '{}_{}'.format(event.name, event.value)
            if e in event2word:
                bar_.append(event2word[e])
            else:
                # OOV
                if event.name == 'Note Velocity':
                    # replace with max velocity based on our training data
                    bar_.append(event2word['Note Velocity_21'])
                else:
                    continue
        words.append(bar_)
    return words

def events2data(words):
    input_data = []
    for bar in words:
        drums = bar
        drums_pos = []
        for word in bar:
            # print(word)
            # print(word2event.get(word))
            if word in pos:
                idx = pos.index(word)
                cur_pos = idx
                drums_pos.append(idx)
            elif word in independent:
                drums_pos.append(cur_pos)
            else:
                event_name, event_value = word2event.get(word).split('_')
                if event_name == 'Note Velocity' or event_name == 'Note Duration':
                    drums_pos.append(cur_pos)
                else:
                    drums_pos.append(len(pos)) # if not position or notes related

        if len(drums) < 256:
            drums.append(END_TOKEN)
            drums_pos.append(len(pos))
            while len(drums) < 256:
                drums.append(PAD_TOKEN)
                drums_pos.append(len(pos) + 1) # if padding
        new_pair = [drums, drums_pos]
        input_data.append(new_pair)
    return np.array(input_data)

###############################################################################
# Build the model
###############################################################################
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    model.eval()
    return model


########################################
# temperature sampling
########################################
def temperature_sampling(logits, temperature, topk):
    probs = np.exp((logits - np.max(logits)) / temperature) / np.sum(np.exp((logits - np.max(logits))/ temperature))
    if topk == 1:
        prediction = np.argmax(probs)
    else:
        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i] for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return prediction

def create_masks(drums_input):
    drums_mask = (drums_input == hyp.PAD_TOKEN)
    return drums_mask

########################################
# generate
########################################
def generate(model, temperature, topk, output_path, drums_event):
    event2word, word2event = pickle.load(open('dictionary.pkl', 'rb'))
    segments = drums_event
    words = []

    poss = []
    for i in range(len(segments)):
        ws = [event2word['Bar_None']]
        if i == 0:
            chords = [v for k, v in event2word.items() if 'Chord' in k]
            ws.append(event2word['Position_1/16'])
            ws.append(np.random.choice(chords))
        ws.append(event2word['Position_1/16'])
        words.append(ws)

        pos_ = [16, pos.index(event2word['Position_1/16']), 16, pos.index(event2word['Position_1/16'])]
        poss.append(pos_)
    original_length = len(words[0])

    enc_mems = tuple()
    dec_mems = tuple()


    for j in range(len(segments)):
        print("generate bar:", j)
        pairs = segments[j]
        drums = np.expand_dims(pairs[0], 1)
        drums_pos = np.expand_dims(pairs[1], 1)

        drums = torch.from_numpy(drums).to(device)
        drums_pos = torch.from_numpy(drums_pos).to(device)
        drums_mask = create_masks(drums)
        if j != 0:
            total_drums_mask = torch.cat([old_drums_mask, drums_mask], dim=0)
            current_drums_mask = drums_mask
        else:
            total_drums_mask = drums_mask
            current_drums_mask = drums_mask

        with torch.no_grad():
            enc_ret = model.encode(drums, drums_pos, total_drums_mask, enc_mems)
            enc_out, enc_mems, = enc_ret[0], enc_ret[1:]

            temp_x = np.zeros((1, original_length), dtype=np.int)  # 1, 6
            temp_bass_pos = np.zeros((1, original_length), dtype=np.int)
            for b in range(1):
                for z, t in enumerate(words[j]):
                    temp_x[b][z] = t
                for z, t in enumerate(poss[j]):
                    temp_bass_pos[b][z] = t
            initial = True
            while len(words[j]) < 256: # 128
                if not initial:
                    # one word one word
                    temp_x = np.zeros((1, 1), dtype=np.int)  # 1,1
                    temp_x[b][0] = words[j][-1]  # 1,1 = 1,-1

                    old_bass_pos = temp_bass_pos[0][-1]
                    temp_bass_pos = np.zeros((1, 1), dtype=np.int)

                    if words[j][-1] in pos:
                        temp_bass_pos[0][0] = pos.index(words[j][-1])
                    elif words[j][-1] in independent:
                        temp_bass_pos[0][0] = old_bass_pos
                    else:
                        event_name, event_value = word2event.get(words[j][-1]).split('_')
                        if event_name == 'Note Velocity' or event_name == 'Note Duration':
                            temp_bass_pos[0][0] = old_bass_pos
                        else:
                            temp_bass_pos[0][0] = 16
                else:
                    initial = False
                bass_input = temp_x  # np.vstack(pairs[:, 1])
                bass_input = torch.transpose(torch.from_numpy(bass_input).to(device), 0, 1)
                bass_pos = torch.transpose(torch.from_numpy(temp_bass_pos).to(device), 0, 1)

                ret = model.forward_generate(bass_input, bass_pos, enc_out,
                                 current_drums_mask,
                                 dec_mems)
                logits, dec_mems, old_drums_mask = ret[0], ret[1], ret[2]
                word = temperature_sampling(logits=logits[-1, 0, :-1].cpu().numpy(), temperature=temperature, topk=topk)
                if word == event2word['Bar_None'] or word == END_TOKEN:
                    break
                words[j].append(word)  # 1,7
            print("bar {0} has {1} word".format(j, len(words[j])))

    final_words = []
    for bar in words:
        final_words.extend(bar)
    utils.write_midi(
        words=final_words,
        word2event=word2event,
        output_path=output_path, instrument=0)


def merge_midi_files(file_1_path, file_2_path, save_path):
    midi_1 = pretty_midi.PrettyMIDI(file_1_path)
    # print(midi_1.instruments)
    # print(midi_1.time_signature_changes)
    # print(midi_1.get_tempo_changes())
    midi_2 = pretty_midi.PrettyMIDI(file_2_path)
    # print(midi_2.instruments)
    # print(midi_2.time_signature_changes)
    # print(midi_2.get_tempo_changes())
    midi_1.instruments.append(midi_2.instruments[0])
    midi_1.write(save_path)



if __name__ == '__main__':
    # input drum midir dir
    midi_dir = "/data/neurips21/other_long_videos_samples/drums_alternative_midi/"


    # list of midi files
    midi_paths = list(find_files_by_extensions(midi_dir, ['.mid', '.midi']))
    # load model
    model_path = 'models/model-piano-new-emb-noshare-rel.pt'
    model = load_model(model_path)
    temperature = 1.2
    top_k = 5
    # generate
    for midi_path in midi_paths:

        words = extract_events(midi_path)
        drums_input = events2data(words)
        output_name = midi_path.split('/')[-1].split('.')[0]
        # if asit
        # output_path = 'outputs/drum_to_piano/aist/{}.mid'.format(output_name)
        # if in the wild
        output_path = "/data/neurips21/other_long_videos_samples/drums_piano_alternative_midi/piano/{}.mid".format(output_name)
        #'/data/neurips21/AIST_new_samples/drums_piano_midi/piano/{}.mid'.format(output_name)
        #'outputs/drum_to_piano/in_the_wild/{}.mid'.format(output_name)
        generate(model, temperature, top_k, output_path, drums_input)
        merge_midi_files(midi_path, output_path, "/data/neurips21/other_long_videos_samples/drums_piano_alternative_midi/merge/{}.mid".format(output_name))
        #'/data/neurips21/AIST_new_samples/drums_piano_midi/merge/{}.mid'.format(output_name))
        #'outputs/drum_to_piano/in_the_wild/merge_{}.mid'.format(output_name)

