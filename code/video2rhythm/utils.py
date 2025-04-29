"""
Kinematic computation code of "How does it sound?: Generation of Rhythmic
Soundtracks for Human Movement Videos" 
Copyright (c) 2021-2022 University of Washington. Developed in UW NeuroAI Lab by Xiulong Liu.
"""
import numpy as np
import librosa


def extract_kinematic_offsets(body_mov, n_bins=9):
    """

    :param body_mov: (T x N_joints x 2)
    :param n_bins: #bins for angle
    :return:
    """
    T, N, _ = body_mov.shape
    # print(T, N)
    directogram = np.zeros((n_bins, T))
    for t in range(T):
        mot_diff = body_mov[t]
        mot_mag = np.linalg.norm(mot_diff, axis=-1)
        cos = mot_diff[:,0] / (mot_mag + 1e-5)
        mot_angle = np.arccos(np.clip(cos, -1.0, 1.0))
        mot_angle = np.rad2deg(mot_angle)
        for k in range(mot_angle.shape[0]):
            try:
                directogram[min(int(mot_angle[k]) // int(180 // n_bins), n_bins - 1), t] += mot_mag[k]
            except:
                print(mot_angle[k])
    kinematic_offsets = librosa.onset.onset_strength(S=directogram, center=False)
    return kinematic_offsets