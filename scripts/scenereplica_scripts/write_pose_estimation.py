import os
from bop_toolkit_lib import inout
import scipy.io as sio
import numpy as np
import argparse
from utils_pose_estimation import *
import open3d as o3d
# algo:
    # if in realworld, no pose found, perception failure 100%
    # if in realworld, its able to grasp and lift, not a perception failure

def read_mat_file(filename):
    contents = sio.loadmat(filename)
    return contents

def extract_experiment_metadata(filename, gt=False):
    filename_pattern = re.compile(r'.*method-(?P<method>\w+)_scene-(?P<scene>\d+)_ord-(?P<order>\w+)$')
    exp_data = filename_pattern.match(filename)
    return exp_data.group

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
        "-d",
        "--data_dir",
        type=str,
        default="/home/benchmark/Datasets/benchmarking/",
        help="Path to parent of model dataset, grasp and scenes dir",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    classes_all = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                    '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                    '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                    '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')   

    T_bc = np.array([[-1.95724205e-05, -7.89282651e-01,  6.14030045e-01, 1.60310904e-01],
                     [-1.00000000e+00,  1.54481744e-05, -1.20180512e-05, 1.99962018e-02],
                     [-3.86463084e-12, -6.14030045e-01, -7.89282652e-01, 1.39498556e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])     

    args = make_args()
    exp_files = os.listdir(args.exp_dir)  # these are not sorted now
    exp_files = sort_files_by_scene_number(exp_files)
    model_dir = os.path.join(args.data_dir, 'models')

    # compute model bounding box centers
    bbox_centers = {}
    for i, model in enumerate(classes_all):
        mesh_name = 'textured_simple.obj'
        # read obj file to compute inertia
        filename = os.path.join(model_dir, model, mesh_name)
        mesh = o3d.io.read_triangle_mesh(filename)
        center = mesh.get_axis_aligned_bounding_box().get_center()
        bbox_centers[model] = center
        print(model, center)    

    # read the results from the exp_files
    head_tail = os.path.split(args.exp_dir)
    model = head_tail[1]
    print('model to be evaluated:', model)
    output_dir = './results'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # list scenes
    filename = os.path.join(args.data_dir, "final_scenes/scene_ids.txt")
    scene_ids = sorted(np.loadtxt(filename).astype(np.int32))

    # loop over all exp files
    results = []
    for file in exp_files:
        # filename = 'poserbpf/23-05-28_T193243_method-poserbpf_scene-27_ord-nearest_first'
        exp_data = extract_experiment_metadata(filename=file)
        order = exp_data("order")
        print(file)

        # read ground truth, get the order
        gt_file = os.path.join(args.data_dir, "final_scenes/metadata", "meta-%06d.mat"%int(exp_data("scene")))
        gt_data = read_mat_file(gt_file)
        gt_objects = [obj.strip() for obj in gt_data["object_names"]]
        gt_poses = gt_data["poses"]

        gt_order = str(gt_data[exp_data("order")][0])     # order in object names
        gt_order = gt_order.split(",")
        prev_objs = set()
        set_gt_order_dec = set(gt_order)
        # loop through each object
        print(f"+++++++--------------------------SCENE  {exp_data('scene')}--------------------------+++++++++")
        for seq, _object in enumerate(gt_order):
            print(f"+++-----------ACTUAL OBJECT Grasped: {_object}--------------+++")
            # read estimated poses
            est_data = read_pickle_file(os.path.join(args.exp_dir, file, f"gt_exp_data_{seq}.pk"))
            # gt_order_dec = set(gt_order) - prev_objs
            set_gt_order_dec.difference_update(prev_objs)
            prev_objs.add(_object)
            for object in set_gt_order_dec:
                print(f"object {object}")
                est_pose = est_data['estimated_poses'][object]
                if est_pose is None:
                    print(f"pose of {object} not detected! 100% perception failure")
                    continue

                # from list to np array
                T_bo = np.array(est_pose).astype(np.float32)
                T_co = np.linalg.inv(T_bc) @ T_bo
                # difference between ycb video and BOP
                T_ob = np.eye(4)
                T_ob[:3, 3] = bbox_centers[object]
                RT = T_co @ T_ob                

                # write to file, format scene_id,im_id,obj_id,score,R,t,time
                res = {}
                res['scene_id'] = scene_ids.index(int(exp_data("scene")))
                res['im_id'] = seq
                res['obj_id'] = classes_all.index(object) + 1
                res['score'] = 1.0
                res['R'] = RT[:3, :3]
                res['t'] = 1000 * RT[:3, 3]
                res['time'] = 1.0
                results.append(res)

    # output file
    output_filename = os.path.join(output_dir, model + '_ycbv-test.csv')
    inout.save_bop_results(output_filename, results)