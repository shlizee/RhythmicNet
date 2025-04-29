DRUM_PITCH_CLASSES = {
    'target': [
        [36],  # kick drum
        [37, 38, 39, 40],  # snare drum
        [41, 58],  # low tom
        [42, 44],  # closed hi-hat
        [43],  # mid tom
        [45],  # high tom
        [46],  # open hi-hat
        [47],  # ride
        [48],  # crash
    ],
    'gmd': [
        [36],  # kick drum
        [38, 37, 40],  # snare drum
        [42, 22, 44],  # closed hi-hat
        [46, 26],  # open hi-hat
        [43, 58],  # low tom
        [47, 45],  # mid tom
        [50, 48],  # high tom
        [49, 52, 55, 57],  # crash cymbal
        [51, 53, 59],  # ride cymbal
    ]
}

NUM_DRUM_PITCH_CLASSES = len(DRUM_PITCH_CLASSES['target'])
SEQUENCE_LENGTH = 32

# TODO: Move preprocessing into training pipeline
DATADIRS = {"gmd": "/data/groove/"}