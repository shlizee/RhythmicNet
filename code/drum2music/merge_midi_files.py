"""
RhyhmicNet Drum2Music stage merge midi scripts of "How does it sound?: Generation of Rhythmic
Soundtracks for Human Movement Videos" 
Copyright (c) 2021-2022 University of Washington. Developed in UW NeuroAI Lab by Xiulong Liu.
"""


import pretty_midi
import glob
def merge_midi_files(file_1_path, file_2_path, save_path):
    midi_1 = pretty_midi.PrettyMIDI(file_1_path)
    print(midi_1.instruments)
    print(midi_1.time_signature_changes)
    print(midi_1.get_tempo_changes())
    midi_2 = pretty_midi.PrettyMIDI(file_2_path)
    print(midi_2.instruments)
    print(midi_2.time_signature_changes)
    print(midi_2.get_tempo_changes())
    midi_1.instruments.append(midi_2.instruments[0])
    print(midi_1.instruments)
    midi_1.write(save_path)

if __name__ == '__main__':
    file_1 = 'outputs/paper_samples/in_the_wild/workout_seg_0.mid'
    file_2 = 'outputs/paper_drum_to_piano/in_the_wild/workout_seg_0.mid'
    save_path = 'outputs/paper_drum_to_piano/in_the_wild/merge/workout_seg_0.mid'
    merge_midi_files(file_1, file_2, save_path)