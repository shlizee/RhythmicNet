"""
Inference pipeline code of "How does it sound?: Generation of Rhythmic
Soundtracks for Human Movement Videos" 
Copyright (c) 2021-2022 University of Washington. Developed in UW NeuroAI Lab by Xiulong Liu.
"""

import sys
sys.path.insert(0, "./drum2music")
sys.path.insert(0, "./checkpoints")
sys.path.insert(0, "./third_party/remi")
sys.path.insert(0, "./third_party/transformer_xl")
from video2rhythm.graph_transformer import SkelGraphTransformer
from video2rhythm.utils import extract_kinematic_offsets
import numpy as np
import torch
from madmom.features import DBNBeatTrackingProcessor
from third_party.magenta.data.converters.groove import GrooveConverter
from third_party.magenta.data.constants import DRUM_PITCH_CLASSES
from rhythm2drum.grootransformer import GrooTransformer
from rhythm2drum.onset2vo import Onset2VO
import note_seq
from drum2music.drum_to_guitar_generate import extract_events, events2data, load_model, generate, merge_midi_files

import math, json, os
import matplotlib.pyplot as plt

from moviepy.editor import *
import pretty_midi
import soundfile as sf


# Stage 1: beat estimation + extract `style' pattern
# Stage 2: style -> polyphonic drum
# Stage 3: polyphonic drum -> full music Midi


# Interpolate to 60fps
def interpolate_skel(skel, frame_rate=60):
    """
    Video2Rhythm is trained on 60fps, therefore upsample original skeleton sequence to 60 fps by linear interpolation.
    :param skel: shape (T, 17, 2)
    :return: upsample_skel  shape (T * 60 / frame_rate, 17, 2)
    """
    if frame_rate == 60:
        return skel
    T, N, _ = skel.shape
    new_T = int(T * 60 / frame_rate)
    upsample_skel = np.zeros((new_T, N, 2))
    fill_index = []
    for i in range(skel.shape[0]):
        re_index = round(i * 60 / frame_rate)
        upsample_skel[re_index] = skel[i]
        fill_index.append(re_index)
    fill_index_set = set(fill_index)
    l_ptr, r_ptr = 0, 1
    for i in range(new_T):
        if i in fill_index_set and i > 0:
            r_ptr += 1
            l_ptr += 1
            continue
        if r_ptr > len(fill_index) - 1:
            upsample_skel[i] = upsample_skel[fill_index[r_ptr - 1]]
        else:
            upsample_skel[i] = (upsample_skel[fill_index[l_ptr]] * (fill_index[r_ptr] - i) +
                                upsample_skel[fill_index[r_ptr]] * (i - fill_index[l_ptr])) / (fill_index[r_ptr] - fill_index[l_ptr])
    return upsample_skel



def video_to_rhythm(skel, frame_rate, beat_est_model, save_path):
    """
    Video2Rhythm stage: Given a sequence of input skeleton stream, perform 3 steps:
    i. Chunk each skeleton sequence into 5-second chunks.
    ii. Upsample the skeleton sequence of first 5 seconds (used for beat estimation) to 60fps for beat estimation.
    iii. Feed the skeleton sequence (first 5 seconds) into `SkelGraphTransformer' to calculate beat distribution, and apply
    BeatTracker for beat estimation.
    iv. Frequency-analysis to extract strong onsets on 1-second input basis.

    :param skel: shape (T, N_joints, 2)
    :param frame_rate: frame rate of original video
    :param beat_est_model: SkelGraphTransformer for beat estimation
    :param save_path: path to save the rhythm file
    :return:
    """
    # BeatTracker initialization for determining beats from beat distribution
    proc = DBNBeatTrackingProcessor(fps=60)

    # Segment original skeleton sequence into 5-second chunks
    ori_skel = []
    total_length = 0
    for i in range(math.ceil(skel.shape[0] / (5 * frame_rate))):
        skel_seg = skel[int(i * 5 * frame_rate): int((i + 1) * 5 * frame_rate + 1)]
        skel_seg = np.diff(skel_seg, axis=0).reshape((-1, 17 * 2)) / 100
        ori_skel.append(skel_seg)
        total_length += skel_seg.shape[0]

    # Upsample skeleton to 60 fps for beat estimation by SkelGraphTransformer
    up_skel = interpolate_skel(skel, frame_rate)
    sample_skel = up_skel[:min(301, up_skel.shape[0])].reshape((-1, 17 * 2)) / 100
    sample_skel = np.diff(sample_skel, axis=0).T
    sample_skel = torch.from_numpy(sample_skel).float().cuda()
    inp_skel = sample_skel.view(17, 2, -1).unsqueeze(-1).permute(1, 2, 0, 3)

    # SkelGraphTransformer Inference, and Estimate Beat using BeatTracker
    beat_est_model.eval()
    with torch.no_grad():
        output = beat_est_model(inp_skel.unsqueeze(0))
        prob_out = torch.sigmoid(output.reshape(1, -1)).cpu().numpy()
        try:
            beat_out_time = proc(prob_out[0])
        except Exception as e:
            print("Beat Estimation Error!", e)
        tempo = 60 / (sample_skel.shape[-1] / 60) * len(beat_out_time)

        beat_out = np.zeros((total_length,))
        last_pred = 0
        beat_interval = (beat_out_time[-1] - beat_out_time[0]) / (len(beat_out_time) - 1)
        beat_interval = int(round(beat_interval * frame_rate))
        for m, t in enumerate(beat_out_time):
            beat_out[int(round(t * frame_rate))] = 1.
            if m == len(beat_out_time) - 1:
                last_pred = int(round(t * frame_rate))
        for beat_idx in range(last_pred + beat_interval, len(beat_out), beat_interval):
            beat_out[beat_idx] = 1.

    # Calculate the Strong Onset based on downsampled kinematic offset (we use low fps like 15 to 25 fps here), and
    # join with the estimated beats on every 5-second segments, then save.
    for l, skel_seg in enumerate(ori_skel):
        skel_seg_length = skel_seg.shape[0]
        if frame_rate == 60:
            sample_skel = skel_seg[::4, :]  # For AIST, downsample by a factor of 4 for skeletons
        elif frame_rate == 50:
            sample_skel = skel_seg[::2, :]
        elif frame_rate == 30:
            sample_skel = skel_seg[::2, :]
        else:
            sample_skel = skel_seg
        if sample_skel.shape[0] < 8:
            continue
        kinematic_offsets = extract_kinematic_offsets(sample_skel.reshape(-1, 17, 2), n_bins=12)
        kinematic_spec = torch.stft(torch.from_numpy(kinematic_offsets).unsqueeze(0), n_fft=16, return_complex=False)[0]

        kinematic_freq_mag = torch.norm(kinematic_spec, dim=-1)[0, :].numpy()
        total_kin_len = len(kinematic_freq_mag)
        kinematic_on = []
        num_sec = math.ceil(skel_seg_length / frame_rate)

        # Every second, extract the movement with top-10 frequency magnitude
        for i in range(num_sec):
            low_index = math.floor(total_kin_len / num_sec * i)
            high_index = math.ceil(total_kin_len / num_sec * (i + 1))
            kinematic_seg = kinematic_freq_mag[low_index: high_index]
            sort_idx = np.argsort(-kinematic_seg)
            on_idx = [low_index + j for j in sort_idx[:max(len(sort_idx) // 10, 1)]]
            kinematic_on.extend(on_idx)

        # Match Hop Size 4
        kinematic_upsample_on = []
        for i in range(sample_skel.shape[0]):
            if i // 4 in set(kinematic_on):
                kinematic_upsample_on.append(1.)
            else:
                kinematic_upsample_on.append(0.)

        # For 60/30fps k_beats also upsample
        if frame_rate in [30, 60, 50]:
            k_beats = []
            for i in range(skel_seg_length):
                if frame_rate != 50 and kinematic_upsample_on[i // int(frame_rate // 15)] == 1:
                    k_beats.append(1.)
                elif frame_rate == 50 and kinematic_upsample_on[i // 2] == 1:
                    k_beats.append(1.)
                else:
                    k_beats.append(0.)
        else:
            k_beats = kinematic_upsample_on

        beat_out_seg = beat_out[int(l * (5 * frame_rate)): int((l + 1) * (5 * frame_rate))]

        taps = ((beat_out_seg + k_beats) > 0).astype("float")
        step_length = 60 / tempo / 4
        taps_downsamp = np.zeros((math.ceil((len(taps) / frame_rate) / step_length), 9))

        # Convert the style/taps into matrix format, this is for the next stage Rhythm2Drum inference.
        for i, t in enumerate(taps):
            if t == 1.0:
                time = i / frame_rate
                index = int(time / step_length)
                if index < taps_downsamp.shape[0]:
                    taps_downsamp[index, 3] = 1.
        # Save every 5-second segment.
        np.save(save_path + "/{}_to_{}_with_bpm_{}.npy".format(int(l * 5 * frame_rate), int(l * 5 * frame_rate + skel_seg_length), int(tempo)),
                taps_downsamp[:, 3], allow_pickle=True)


def rhythm_to_drum(onset_model, vo_model, taps_path, save_path):
    """
    Stage 2 Rhythm2Drum: Convert the `style' matrix in Stage 1 into a Polyphonic Midi. This is achieved in 2 steps:
    i. Use a GrooTransformer to translate `style' into a 2D `hit' matrix.
    ii. Apply a UNet-based model to generate `velocity' and `offsets' for the Drum Midi
    :param onset_model:
    :param vo_model:
    :param taps_path:
    :param save_path:
    :return:
    """
    with open("./rhythm2drum/vocab.json", "r") as fin:
        vocab_set = json.load(fin)
    idx_to_vocab = {v: k for k, v in vocab_set.items()}
    converter = GrooveConverter(
        split_bars=2,
        steps_per_quarter=4,
        quarters_per_bar=4,
        pitch_classes=DRUM_PITCH_CLASSES["gmd"],
        humanize=False,
        tapify=True,
        fixed_velocities=True
    )
    onset_model.eval()
    vo_model.eval()
    with torch.no_grad():
        filenames = sorted([fn for fn in os.listdir(taps_path) if 'drum' not in fn and 'piano' not in fn and 'guitar' not in fn], key=lambda fn: int(fn.split("_")[0]))
        indices_list = []
        qpm = 0
        for fn in filenames:
            fp = os.path.join(taps_path, fn)
            taps = np.load(fp)
            for i in range(taps.shape[0]):
                taps[i] = np.random.choice([0, 1])

            qpm = int(float(fn[:-4].split("_")[-1]))
            taps = torch.from_numpy(taps).type(torch.LongTensor).cuda().unsqueeze(0)
            indices = onset_model.sample(taps, taps.shape[1]).cpu().numpy()
            indices_list.append(indices)
        indices = np.concatenate(indices_list, axis=1)

        onset_arrays = []
        for m in range(indices.shape[0]):
            on_set = [idx_to_vocab[idx] for idx in indices[m]]
            onset_arr = np.array([np.array([ch for ch in code]).astype("float") for code in on_set])
            onset_arrays.append(np.expand_dims(onset_arr, axis=0))
        onset_arrays = np.concatenate(onset_arrays, axis=0)
        onset_arrays = torch.from_numpy(onset_arrays).float().cuda().unsqueeze(1)
        velocity, offset = vo_model(onset_arrays)

        v = velocity[0].cpu().numpy()
        o = offset[0].cpu().numpy()
        gen = np.concatenate([onset_arrays[0, 0].cpu().numpy(), v, o], axis=1)
        out_seq = converter.from_tensors(np.expand_dims(gen, 0), qpm=qpm)
        note_seq.midi_io.note_sequence_to_midi_file(out_seq[0], save_path)


def drum_to_music(drum_midi_path, top_k, temperature, model_path, instrument="guitar"):
    """
    Generate guitar or piano Midi conditioned on DrumMidi.
    :param drum_midi_path:
    :param top_k:
    :param temperature:
    :param model_path:
    :param instrument:
    :return:
    """
    model = load_model(model_path)
    words = extract_events(drum_midi_path)
    drums_input = events2data(words)
    output_instrument_path = os.path.join(os.path.dirname(drum_midi_path), instrument + ".mid")
    generate(model, temperature, top_k, output_instrument_path, drums_input)
    merge_midi_path = os.path.join(os.path.dirname(drum_midi_path), "drum_" + instrument + ".mid")
    merge_midi_files(drum_midi_path, output_instrument_path, merge_midi_path)


def add_audio_to_video(drum_midi_path, video_path, audio_save_path, video_save_path):
    """
    Convert Midi to Waveform using FluidSynth and replace the original audio with generated audio in video.
    :param drum_midi_path:
    :param video_path:
    :param audio_save_path:
    :param video_save_path:
    :return:
    """
    pm = pretty_midi.PrettyMIDI(drum_midi_path)
    wav = pm.fluidsynth(fs=16000)
    sf.write(audio_save_path, wav, 16000)

    videoclip = VideoFileClip(video_path)
    clip_duration = videoclip.duration
    audioclip = AudioFileClip(audio_save_path).set_duration(clip_duration)

    new_audioclip = CompositeAudioClip([audioclip])
    videoclip = videoclip.set_audio(new_audioclip)
    videoclip.write_videofile(video_save_path,
                              codec="libx264",
                              audio_codec="aac")


def infer_pipeline(skel_path, video_path, save_path, video_fps, model_path_s1, model_path_s21, model_path_s22,
                   model_path_s3, temperature = 1.2, top_k = 5):
    """

    :param skel_path:
    :param video_path:
    :param save_path:
    :param video_fps:
    :param model_path_s1:
    :param model_path_s21:
    :param model_path_s22:
    :param model_path_s3:
    :param temperature:
    :param top_k:
    :return:
    """
    fn = os.path.basename(skel_path)
    save_path = os.path.join(save_path, fn[:-4])

    # Stage 1: Video2Rhythm
    model_s1 = SkelGraphTransformer(use_rel_attn=True, add_self_sim=True)
    model_s1.cuda()
    model_s1.load_state_dict(torch.load(model_path_s1))

    os.makedirs(save_path, exist_ok=True)
    skel = np.load(skel_path)[:, :, :2]

    video_to_rhythm(skel, video_fps, model_s1, save_path)


    # Stage 2: Rhythm2Drum
    model_s21 = GrooTransformer(n_src_vocab=2, n_trg_vocab=152, d_word_vec_enc=128, d_word_vec_dec=128,
                                  mask_zero=True)
    model_s21.cuda()
    model_s21.load_state_dict(torch.load(model_path_s21))
    input_shape = (1, 9, 32)
    model_s22 = Onset2VO(input_shape)
    model_s22.cuda()
    model_s22.load_state_dict(torch.load(model_path_s22))
    save_midi_path = os.path.join(save_path, "drum.mid")
    rhythm_to_drum(model_s21, model_s22, save_path, save_midi_path)

    save_audio_path = os.path.join(save_path, "drum.wav")
    save_video_path = os.path.join(save_path, "drum.mp4")
    add_audio_to_video(save_midi_path, video_path, save_audio_path, save_video_path)

    # Stage 3: Drum2Music
    drum_midi_path = save_midi_path
    instrument = "guitar" if "guitar" in model_path_s3 else "piano"
    drum_to_music(drum_midi_path, top_k, temperature, model_path_s3, instrument=instrument)

    save_audio_path = os.path.join(save_path, "drum_{}.wav".format(instrument))
    save_video_path = os.path.join(save_path, "drum_{}.mp4".format(instrument))
    merge_midi_path = os.path.join(os.path.dirname(drum_midi_path), "drum_" + instrument + ".mid")
    add_audio_to_video(merge_midi_path, video_path, save_audio_path, save_video_path)


if __name__ == "__main__":
    # An example inference pipeline
    video_name = "24_manskate_0"
    video_fps = 24.
    video_path = f"./examples/videos/{video_name}.mp4"
    skel_path = f"./examples/preprocessed_skeletons/{video_name}.npy"
    save_path = "./outputs/"
    model_path_s1 = "./checkpoints/graph_transformer_relattn_selfsim/85.pth"
    model_path_s21 = "./checkpoints/grootransformer/12.pth"
    model_path_s22 = "./checkpoints/onset2vo/22.pth"
    # Use guitar checkpoint, please uncomment line below and comment piano checkpoint path
    # model_path_s3 = "./checkpoints/drum2music/model-guitar-new-emb-noshare-rel.pt"
    # Use piano checkpoint
    model_path_s3 = "./checkpoints/drum2music/model-piano-new-emb-noshare-rel.pt"
    os.makedirs(save_path, exist_ok=True)
    infer_pipeline(skel_path, video_path, save_path, video_fps, model_path_s1, model_path_s21, model_path_s22,
                   model_path_s3)




