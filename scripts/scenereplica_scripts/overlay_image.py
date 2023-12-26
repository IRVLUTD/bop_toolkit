import os
import cv2
import argparse
import numpy as np
from bop_toolkit_lib import inout
import matplotlib.pyplot as plt


def make_args():
    parser = argparse.ArgumentParser(
        description="Generate grid and spawn objects", add_help=True
    )
    parser.add_argument(
        "-i",
        "--scene_id",
        type=int,
        default=0,
        help="Index for the scene to be show",
    )    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = make_args()
    scene_id = args.scene_id
    image_id = 0

    dst_dir = '../../data/ycbv/test'
    reference_dir = '../../data/final_scenes/metadata'

    # read scene image
    filename = os.path.join(dst_dir, '%06d' % scene_id, 'rgb', '%06d.png' % image_id)
    print(filename)
    im = cv2.imread(filename)

    # read scene info
    filename = os.path.join(dst_dir, '%06d' % scene_id, 'scene_info.json')
    scene_info = inout.load_json(filename)
    print(scene_info)

    # read pose image
    filename = os.path.join(reference_dir, 'pose-%06d.png' % int(scene_info['scene']))
    im_ref = cv2.imread(filename)

    # read rendered image
    filename = os.path.join(dst_dir, '%06d' % scene_id, 'scene-%06d.png' % scene_id)
    im_render = cv2.imread(filename)

    # overlay
    image_disp = 0.4 * im.astype(np.float32) + 0.6 * im_ref.astype(np.float32)
    image_disp = np.clip(image_disp, 0, 255).astype(np.uint8)

    image_disp_1 = 0.4 * im.astype(np.float32) + 0.6 * im_render.astype(np.float32)
    image_disp_1 = np.clip(image_disp_1, 0, 255).astype(np.uint8)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(im[:, :, (2, 1, 0)])
    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(im_ref[:, :, (2, 1, 0)])
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(image_disp[:, :, (2, 1, 0)])
    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(image_disp_1[:, :, (2, 1, 0)])    
    plt.show()
