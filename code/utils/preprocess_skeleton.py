import os, glob, json, argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='preprocess skeletons')
    parser.add_argument('--raw_skel_folder', type=str, help='raw skeleton folder')
    args = parser.parse_args()
    raw_skeleton_folder = args.raw_skel_folder
    for filepath in glob.glob(raw_skeleton_folder):
        basename = filepath[:-1].split('/')[-1]
        skel_file = [fn for fn in glob.glob(filepath+"*")]
        sort_skel_file = sorted(skel_file, key=lambda fn: int(os.path.basename(fn).split("_")[-2]))
        vid_skeleton_list = []
        for f in sort_skel_file:
            with open(f, "r") as fin:
                skel = json.load(fin)
            if len(skel["people"]) > 0:
                if len(skel["people"]) > 1:
                    print("Multiple person occur in {}".format(filepath))
                skel_list = skel["people"][0]["pose_keypoints_2d"]
                skel_arr = np.array(skel_list).reshape((-1, 3))[list(range(15))+[19, 22], :]
            else:
                skel_arr = np.zeros((17, 3))
                print("Missing frame in file {}".format(os.path.basename(f).split("/")[-1]))
            vid_skeleton_list.append(np.expand_dims(skel_arr, axis=0))
        vid_skeleton_arr = np.concatenate(vid_skeleton_list, axis=0)

        np.save("./examples/preprocessed_skeletons/{}.npy".format(basename), vid_skeleton_arr, allow_pickle=True)