# Inference Code and Pipeline for `RhythmicNet' published in NeurIPS 2021 "How does it Sound?" (see citation below).
## Package Dependencies
This codebase is developed for Python3.7. To get started, please create a conda environment by:

'''
conda create --name rhythmicnet python=3.7
conda activate rhythmicnet
'''

We also tested it on Python 3.8. For Python 3.8, you could run the following:

'''
conda create --name rhythmicnet python==3.8
conda activate rhythmicnet
'''

For Python 3.7, pytorch, the current codebase is tested to work well with torch 1.11 with CUDA 10.2.
For Linux, run "conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch".

For Python 3.8, pytorch, the codebase works well with torch==2.0.1 and numpy==1.23.1 with CUDA 11.4.

Please check ``requirements.txt`` for package dependencies, and then run "pip install -r requirements.txt" or install missing packages individually in your conda environment.

For "pyfluidsynth" and "pretty-midi" packages, after doing "pip install", Make sure to install "libfluidsynth-dev" as well by running "sudo apt-get install libfluidsynth-dev", then "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7", you can then successfully "import fluidsynth, pretty_midi".

## Getting Started
The entire pipeline is integrated in `inference.py` file, connecting the 3 stages from `Video2Rhythm` (`video2rhythm` folder),
to `Rhythm2Drum` (`rhythm2drum` folder) to `Drum2Music` (`drum2music`folder). To test the inference code: 

1. Put your video under `example/videos/` folder.
2. Prepare your extracted and preprocessed skeleton sequence file like a single `xxx.npy` consisting of a numpy array
 of shape `[T, 17, 3]`, where `T' is the number of frames.

    i. Extract the body keypoints of every frame in the video using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
       `./build/examples/openpose/openpose.bin --write_json [your saved keypoints folder] --write_video [your video]`
       
    ii. Run ``python utils/preprocessed_skeleton.py --raw_skel_folder [your saved keypoints folder]`` and it will save the
    preprocessed skeleton file under ``examples/preprocessed_skeleton/[video name]`` folder.
3. Once you get the video file and preprocessed skeleton file ready, run the inference pipeline code ``python inference.py``.
   In the ``__main__`` part, you need to adjust `video_fps` to be your test video frame rate, and you could switch between `piano`
   and ``guitar`` Midi generation by changing the Stage 3 checkpoint `model_path_s3` (Check the code for more details).
   
4. After running `inference.py`, you will see the all intermediate outputs (including drum + piano/guitar Midi, waveforms) and the final video with "drum only" or "drum + piano/guitar"
under the ``outputs/[video name]/`` folder.

5. Since the system consists of generative models, each different run for the same video will still produce different soundtracks. If you are not satisfied with
the output sound, you could regenerate the soundtrack. To do so, please remove the entire `outputs/[video name]` folder and rerun ``python inference.py``.

Our system is using useful third-party codebases (in original or adapted forms): 
mm-skeleton (https://github.com/open-mmlab/mmskeleton/), 
Magenta's codebasae (https://github.com/magenta/), 
Remi (https://github.com/YatingMusic/remi/), 
transformer-xl (https://github.com/kimiyoung/transformer-xl/)
pytorch transformer implementation from https://github.com/hyunwoongko/transformer/tree/master. 

These codebases are located in third_party folder of this repository and their respective terms of use and licenses are located therein as well.
If you find our work and codebase useful, please consider citing our work along with the above codebases and their associated papers.

## Contact
Xiulong Liu xl1995@uw.edu / liuxiulong1995@gmail.com or Eli Shlizerman, shlizee@uw.edu.

## Citation
```
@inproceedings{NEURIPS2021_f4e369c0,
 author = {Su, Kun and Liu, Xiulong and Shlizerman, Eli},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {29258--29273},
 publisher = {Curran Associates, Inc.},
 title = {How Does it Sound?},
 url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/f4e369c0a468d3aeeda0593ba90b5e55-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
