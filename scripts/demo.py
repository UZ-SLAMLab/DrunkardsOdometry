import sys

sys.path.append('.')

import argparse
import copy

from data_readers.demo import DemoDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm


def prepare_images_and_depths(image1, image2, depth1, depth2):
    """ padding, normalization, and scaling """

    ht, wd = image1.shape[-2:]
    pad_h = (-ht) % 8
    pad_w = (-wd) % 8

    image1 = F.pad(image1, [0, pad_w, 0, pad_h], mode='replicate')
    image2 = F.pad(image2, [0, pad_w, 0, pad_h], mode='replicate')
    depth1 = F.pad(depth1[:, None], [0, pad_w, 0, pad_h], mode='replicate')[:, 0]
    depth2 = F.pad(depth2[:, None], [0, pad_w, 0, pad_h], mode='replicate')[:, 0]

    image1 = normalize_image(image1.float())
    image2 = normalize_image(image2.float())

    depth1 = depth1.float()
    depth2 = depth2.float()

    return image1, image2, depth1, depth2

@torch.no_grad()
def test(model, args):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
    dataset = DemoDataset(root=args.datapath, intrinsics=args.intrinsics, res_factor=args.res_factor,
                          depth_factor=args.depth_factor, depth_limit_bottom=args.depth_limit_bottom,
                          depth_limit_top=args.depth_limit_top)
    loader = DataLoader(dataset, **loader_args)

    f_poses = open(os.path.join(args.save_path, 'pose_est.txt'), 'w')

    for i_batch, test_data_blob in enumerate(tqdm(loader)):
        image1, image2, depth1, depth2, intrinsics, valid_mask, depth_scale_factor = \
            [data_item.cuda() for data_item in test_data_blob]

        # pad and normalize images
        image1, image2, depth1, depth2 = prepare_images_and_depths(image1, image2, depth1, depth2)

        T, pose = model(
        **dict(image1=image1, image2=image2, depth1=depth1, depth2=depth2,
               intrinsics=intrinsics, iters=12, train_mode=False,
               depth_scale_factor=depth_scale_factor))

        # Uncomment if you want 2D and 3D flow and valid flow pixels
        # flow2d_est, flow3d_est, valid = pops.induced_flow(T, depth1, intrinsics, min_depth=0.01, max_depth=0.3)
        # valid = valid > 0.5
        # valid_mask *= valid.unsqueeze(-1)

        # Write absolut transformation poses in world-to-camera format (T_w_c)
        pose = pose.squeeze().cpu().detach().numpy()
        if i_batch == 0:
            pose1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype="float32")
            f_poses.writelines('{:010d}'.format(i_batch) + ' ' + str(pose1[0]) + ' ' + str(pose1[1]) + ' ' +
                               str(pose1[2]) + ' ' + str(pose1[3]) + ' ' + str(pose1[4]) + ' ' + str(pose1[5]) +
                               ' ' + str(pose1[6]) + '\n')

        pose2 = pops.relative_to_absoult_poses(pose1, pose)
        f_poses.writelines('{:010d}'.format(i_batch + 1) + ' ' + str(pose2[0]) + ' ' + str(pose2[1]) + ' ' +
                           str(pose2[2]) + ' ' + str(pose2[3]) + ' ' + str(pose2[4]) + ' ' + str(pose2[5]) + ' ' +
                           str(pose2[6]) + '\n')
        pose1 = copy.deepcopy(pose2)

    f_poses.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datapath', type=str, required=True, help='full path to folder containing the scenes')
    parser.add_argument("--save_path", type=str, help="specify full path. Results will be saved here")
    parser.add_argument('--pose_bias', type=float, default=0.01,
                        help='bias to be multiplied to the estimated delta_pose of the model in each iteration.')
    parser.add_argument('--radius', type=int, default=32)
    parser.add_argument('--res_factor', type=float, default=1.)
    parser.add_argument('--depth_factor', type=float, default=1., help='factor to multiply the read depth pixel values')
    parser.add_argument('--depth_limit_bottom', type=float, default=0.,
                        help='mask out pixels with depth values below this limit')
    parser.add_argument('--depth_limit_top', type=float, default=float("inf"),
                        help='mask out pixels with depth values over this limit')
    parser.add_argument('--intrinsics', type=float, nargs='+', required=True,
                        help='intrinsics: fx, fy, cx, cy')
    args = parser.parse_args()
    print(args)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    import importlib

    model = importlib.import_module('drunkards_odometry.model').DrunkardsOdometry(args)
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.cuda()
    model.eval()

    if args.save_path:
        original_save_path = args.save_path
    else:
        original_save_path = os.path.join(os.getcwd(), 'evaluations_demo')  # todo chequear que el os.getcwd() me devuelve el path padre de este script, es decri, el main folder

    args.save_path = os.path.join(original_save_path, args.name)
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    test(model, args)
