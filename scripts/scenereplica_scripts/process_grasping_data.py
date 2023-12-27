import os
from bop_toolkit_lib import inout
import scipy.io as sio
import numpy as np
import argparse
from utils_pose_estimation import *
import open3d as o3d
import re
# algo:
    # if in realworld, no pose found, perception failure 100%
    # if in realworld, its able to grasp and lift, not a perception failure

def read_mat_file(filename):
    contents = sio.loadmat(filename)
    return contents


def load_grasping_results(path):
    """Loads 6D object pose estimates from a file.

    :param path: Path to a file with pose estimates.
    :param version: Version of the results.
    :return: List of loaded poses.
    """
    results = []
    with open(path, "r") as f:
        line_id = 0
        for line in f:
            line_id += 1
            if line_id == 1:
                continue
            else:
                line = re.sub(' +', ' ', line)
                line = re.sub('\t+', ' ', line)
                elems = line.split(" ")
                if len(elems) != 8:
                    continue

                result = {
                    "scene_id": int(elems[0]),
                    "object_name": elems[1],
                    "grasp_find": int(elems[2]),
                    "gripper_close": int(elems[3]),
                    "lift": int(elems[4]),
                    "rotate": int(elems[5]),
                    "move_to_dropoff": int(elems[6]),
                    "drop": int(elems[7]),
                }

                results.append(result)

    return results


def save_grasping_results(path, results):
    """Saves 6D object pose estimates to a file.

    :param path: Path to the output file.
    :param results: Dictionary with pose estimates.
    :param version: Version of the results.
    """
    # See docs/bop_challenge_2019.md for details.
    lines = ["scene_id,im_id,obj_id,grasp_find,gripper_close,lift,rotate,move_to_dropoff,drop"]
    for res in results:
        lines.append(
            "{scene_id},{im_id},{obj_id},{grasp_find},{gripper_close},{lift},{rotate},{move_to_dropoff},{drop}".format(
                scene_id=res["scene_id"],
                im_id=res["im_id"],
                obj_id=res["obj_id"],
                grasp_find=res['grasp_find'],
                gripper_close=res['gripper_close'],
                lift=res['lift'],
                rotate=res['rotate'],
                move_to_dropoff=res['move_to_dropoff'],
                drop=res['drop'],
            )
        )

    with open(path, "w") as f:
        f.write("\n".join(lines))


def make_args():
    parser = argparse.ArgumentParser(
        description="pose analysis"
    )

    parser.add_argument(
        "-e",
        "--exp_dir",
        default="./gdrnpp",
        help="evaluation directory like poserbpf, posecnn etc..,",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="gdrnpp",
        help="evaluation method like poserbpf, posecnn etc..,",
    )    
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default="../../data",
        help="Path to parent of model dataset, grasp and scenes dir",
    )

    args = parser.parse_args()
    return args


def parse_object_name(name, classes):
    name = name.strip()
    for cls in classes[::-1]:
        ns = cls.split('_')
        cls_new = "".join(ns[1:])
        if cls_new in name or name in cls_new:
            return cls
    print(f'cannot parse {name}')
    return None


if __name__ == "__main__":

    classes_all = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                    '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                    '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                    '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')      

    args = make_args()
    output_dir = './results'
    model = args.model

    # list scenes
    filename = os.path.join(args.data_dir, "final_scenes/scene_ids.txt")
    scene_ids = sorted(np.loadtxt(filename).astype(np.int32))

    results_new = []
    for order in ['random', 'nearest_first']:
        # output file
        filename = os.path.join(output_dir, model + f'_{order}_grasping.csv')
        results = load_grasping_results(filename)

        for exp_data in results:

            # parse object name
            name = exp_data['object_name']
            object_name = parse_object_name(name, classes_all)
            print(object_name)

            # read ground truth, get the order
            gt_file = os.path.join(args.data_dir, "final_scenes/metadata", "meta-%06d.mat"%int(exp_data["scene_id"]))
            gt_data = read_mat_file(gt_file)
            gt_order = str(gt_data[order][0])     # order in object names
            gt_order = gt_order.split(",")

            # find image id
            im_id = gt_order.index(object_name)
            print(im_id, gt_order)

            # write to file, format scene_id,im_id,obj_id,score,R,t,time
            res = {}
            scene_id = scene_ids.index(int(exp_data["scene_id"]))
            if order == 'nearest_first':
                scene_id = 2 * scene_id
            else:
                scene_id = 2 * scene_id + 1

            res['scene_id'] = scene_id
            res['im_id'] = im_id
            res['obj_id'] = classes_all.index(object_name) + 1
            res["grasp_find"] = exp_data['grasp_find']
            res["gripper_close"] = exp_data['gripper_close']
            res["lift"] = exp_data['lift']
            res["rotate"] = exp_data['rotate']
            res["move_to_dropoff"] = exp_data['move_to_dropoff']
            res["drop"] = exp_data['drop']
            results_new.append(res)
    print(results_new)
    print(len(results_new))

    filename = os.path.join(output_dir, model + '_grasping.csv')
    save_grasping_results(filename, results_new)
    print(filename)