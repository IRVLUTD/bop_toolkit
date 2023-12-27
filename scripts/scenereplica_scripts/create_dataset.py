import os
import re
import cv2
import scipy
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils_pose_estimation import *
import open3d as o3d
from transforms3d.quaternions import mat2quat
from bop_toolkit_lib import inout

this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'ycb_render')
sys.path.insert(0, lib_path)
from ycb_renderer import YCBRenderer


def extract_experiment_metadata(filename, gt=False):
    filename_pattern = re.compile(r'scene_(?P<scene>\d+)_(?P<order>\w+)$')
    exp_data = filename_pattern.match(filename)
    return exp_data.group


if __name__ == "__main__":

    # {"img_width": 640, "img_height": 480, "fx": 530.1507980, "fy": 527.83633424, "x_offset": 321.85101905, "y_offset": 232.7456859}
    cam_K = [530.1507980, 0.0, 321.85101905, 0.0, 527.83633424, 232.7456859, 0.0, 0.0, 1.0]         # real fetch camera
    # cam_K = [554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0]           # gazebo camera
    depth_scale = 1.0
    width = 640
    height = 480
    camera_info = {'cam_K': cam_K, 'depth_scale': depth_scale}

    T_bc = np.array([[-1.95724205e-05, -7.89282651e-01,  6.14030045e-01, 1.60310904e-01],
                     [-1.00000000e+00,  1.54481744e-05, -1.20180512e-05, 1.99962018e-02],
                     [-3.86463084e-12, -6.14030045e-01, -7.89282652e-01, 1.39498556e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

    # define all the classes
    classes_all = ('002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                    '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                    '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                    '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
    class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                              (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
                              (192, 0, 0), (0, 192, 0), (0, 0, 192)]
    num_classes = len(classes_all)

    data_dir = '../../data'
    model_dir = '/home/yuxiang/Projects/SceneReplica/data/models'
    root_dir = '../../data/scenereplica_data'
    dst_dir = '../../data/ycbv/test'
    method = 'ground_truth'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

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

    print('loading 3D models')
    renderer = YCBRenderer(width=width, height=height, gpu_id=0, render_marker=False)
    model_mesh_paths = [os.path.join(model_dir, cls, 'textured_simple.obj') for cls in classes_all]
    model_texture_paths = [os.path.join(model_dir, cls, 'texture_map.png') for cls in classes_all]
    model_colors = [class_colors_all[i] for i in range(num_classes)]
    print(model_mesh_paths)
    renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)
    renderer.set_camera_default()

    exp_files = sorted(os.listdir(root_dir))  # these are not sorted now
    print(exp_files, len(exp_files))

    targets = []
    for scene_id, file in enumerate(exp_files):
        # filename = 'poserbpf/23-05-28_T193243_method-poserbpf_scene-27_ord-nearest_first'
        exp_data = extract_experiment_metadata(filename=file)
        scene = exp_data("scene")
        order = exp_data("order")

        # read ground truth, get the order
        gt_file = os.path.join(data_dir, "final_scenes/metadata", "meta-%06d.mat" % int(scene))
        print(file)
        print(gt_file, scene, order)
        gt_data = scipy.io.loadmat(gt_file)
        gt_objects = [obj.strip() for obj in gt_data["object_names"]]
        gt_poses = gt_data["poses"]
        gt_order = str(gt_data[exp_data("order")][0])     # order in object names
        gt_order = gt_order.split(",")        

        # create dir
        dirname = 'scene_%03d_order_%s' % (int(scene), order)
        dirname = '%06d' % scene_id
        dirpath = os.path.join(dst_dir, dirname)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)        
            os.makedirs(dirpath + '/rgb')
            os.makedirs(dirpath + '/depth')
        print(dirname, gt_objects)

        # scene info
        scene_info = {'scene': scene, 'order': order}
        dst_file = os.path.join(dirpath, 'scene_info.json')
        inout.save_json(dst_file, scene_info)

        # create scene gt
        scene_gt = {}
        scene_camera = {}
        cls_indexes = []
        poses_all = []
        # copy files and compute gt object poses
        prev_objs = set()
        set_gt_order_dec = set(gt_order)
        for seq, _object in enumerate(gt_order):
            # gt
            im_id = f'{seq}'
            im_gts = []
            scene_camera[im_id] = camera_info

            # gt_order_dec = set(gt_order) - prev_objs
            set_gt_order_dec.difference_update(prev_objs)
            prev_objs.add(_object)
            for object in set_gt_order_dec:

                # get the ground truth pose
                object_gt_pose = gt_poses[gt_objects.index(object)]
                object_gt_pose = convert_standard_to_rosqt(object_gt_pose)
                T_bo = ros_qt_to_rt(object_gt_pose[3:], object_gt_pose[:3]) # get object pose as 4x4
                T_co = np.linalg.inv(T_bc) @ T_bo
                RT = T_co

                qt = np.zeros((7,), dtype=np.float32)
                qt[:3] = RT[:3, 3]
                qt[3:] = mat2quat(RT[:3, :3])
                poses_all.append(qt.copy())

                # get the object index in _classes_all
                cls_index = classes_all.index(object) + 1 
                cls_indexes.append(cls_index - 1)

                # difference between ycb video and BOP
                T_ob = np.eye(4)
                T_ob[:3, 3] = bbox_centers[object]
                RT = T_co @ T_ob

                gt = {}                
                gt["cam_R_m2c"] = RT[:3, :3].flatten().tolist()
                T = 1000 * RT[:3, 3]     # convert to mm
                gt["cam_t_m2c"] = T.flatten().tolist()
                gt['obj_id'] = cls_index
                im_gts.append(gt)

                target = {}
                target['im_id'] = seq
                target['inst_count'] = 1
                target['obj_id'] = cls_index
                target['scene_id'] = scene_id
                targets.append(target)

            scene_gt[im_id] = im_gts
        
            # color
            src_file = os.path.join(root_dir, file, '%06d_color.png' % seq)
            im = cv2.imread(src_file)
            dst_file = os.path.join(dirpath, 'rgb', '%06d.png' % seq)
            cv2.imwrite(dst_file, im)

            # depth
            src_file = os.path.join(root_dir, file, '%06d_depth.png' % seq)
            dst_file = os.path.join(dirpath, 'depth', '%06d.png' % seq)
            command = f'cp {src_file} {dst_file}'
            os.system(command)

        # save scene gt
        dst_file = os.path.join(dirpath, 'scene_gt.json')
        inout.save_json(dst_file, scene_gt)

        dst_file = os.path.join(dirpath, 'scene_camera.json')
        inout.save_json(dst_file, scene_camera)

        # rendering
        intrinsic_matrix = np.array(cam_K).reshape((3, 3))
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        px = intrinsic_matrix[0, 2]
        py = intrinsic_matrix[1, 2]
        zfar = 10.0
        znear = 0.01
        image_tensor = torch.cuda.FloatTensor(height, width, 4)
        seg_tensor = torch.cuda.FloatTensor(height, width, 4)

        # set renderer
        renderer.set_light_pos([0, 0, 0])
        renderer.set_light_color([1, 1, 1])
        renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

        # pose
        renderer.set_poses(poses_all)
        frame = renderer.render(cls_indexes, image_tensor, seg_tensor)
        image_tensor = image_tensor.flip(0)
        im_render = image_tensor.cpu().numpy()
        im_render = np.clip(im_render, 0, 1)
        im_render = im_render[:, :, :3] * 255
        im_render = im_render.astype(np.uint8)

        dst_file = os.path.join(dirpath, 'scene-%06d.png' % scene_id)
        cv2.imwrite(dst_file, im_render[:, :, (2, 1, 0)])

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 2, 1)        
        # plt.imshow(im_render)
        # src_file = os.path.join(dirpath, 'rgb', '000000.png')
        # im = cv2.imread(src_file)
        # image_disp = 0.4 * im.astype(np.float32) + 0.6 * im_render.astype(np.float32)
        # image_disp = np.clip(image_disp, 0, 255).astype(np.uint8)
        # ax = fig.add_subplot(1, 2, 2)        
        # plt.imshow(image_disp[:, :, (2, 1, 0)])
        # plt.show()    

    # save targets
    dst_file = os.path.join(dst_dir, '../test_targets_bop19.json')
    inout.save_json(dst_file, targets)