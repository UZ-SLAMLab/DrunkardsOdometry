import sys

sys.path.append('.')

import argparse
import copy

from data_readers.hamlyn import HamlynDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_readers.frame_utils import *
from utils import *


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

    return image1, image2, depth1, depth2, (pad_w, pad_h)

@torch.no_grad()
def test(model, scene, args):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'drop_last': False}

    test_dataset = HamlynDataset(root=args.datapath, scenes_to_use=scene)
    test_loader = DataLoader(test_dataset, **loader_args)

    f_poses = open(os.path.join(args.save_path, 'stamped_traj_estimate.txt'), 'w')

    for i_batch, test_data_blob in enumerate(tqdm(test_loader)):
        # if i_batch == 10:
        #     num_frames = 10
        #     break

        image1, image2, depth1, depth2, intrinsics, valid_mask, depth_scale_factor = [data_item.cuda() for data_item in test_data_blob]

        # pad and normalize images  #todo para que hace el padding?
        image1, image2, depth1, depth2, padding = prepare_images_and_depths(image1, image2, depth1, depth2)

        Ts, Ts_def, pose = model(
        **dict(image1=image1, image2=image2, depth1=depth1, depth2=depth2,
               intrinsics=intrinsics, valid_mask=valid_mask, iters=12, train_mode=False,
               depth_scale_factor=depth_scale_factor))

        # use transformation field to extract 2D and 3D flow
        flow2d_est, flow3d_est, valid = pops.induced_flow(Ts, depth1, intrinsics)
        valid = valid > 0.5
        valid_mask *= valid.unsqueeze(-1)

        pose = pose.squeeze().cpu().detach().numpy()

        if i_batch == 0:
            pose1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype="float32")
            f_poses.writelines('{:010d}'.format(i_batch) + ' ' + str(pose1[0]) + ' ' + str(pose1[1]) + ' ' + str(pose1[2]) + ' ' + str(
                    pose1[3]) + ' ' + str(pose1[4]) + ' ' + str(pose1[5]) + ' ' + str(pose1[6]) + '\n')

        pose2 = pops.relative_to_absoult_poses(pose1, pose)  # Absolut pose from camera to world # todo comprobar que esto esta bien

        f_poses.writelines('{:010d}'.format(i_batch + 1) + ' ' + str(pose2[0]) + ' ' + str(pose2[1]) + ' ' + str(pose2[2]) + ' ' + str(pose2[3]) + ' ' + str(pose2[4]) + ' ' + str(pose2[5]) + ' ' + str(pose2[6]) + '\n')

        pose1 = copy.deepcopy(pose2)

    f_poses.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='name your experiment')
    parser.add_argument('--network', default='drunkards_odometry.model')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datapath', type=str, required=True, help='full path to folder containing the scenes')
    parser.add_argument('--test_scenes', type=str, nargs='+', default=["test1", "test1_backward", "test17", "test17_backward", "test22", "test22_backward"], help='scenes used for training')
    parser.add_argument("--save_path", type=str, default='/media/david/DiscoDuroLinux/Datasets/evaluations_drunk/evaluations_hamlyn', help="if specified, logs and results will be saved here")
    parser.add_argument('--pose_rot_error_vector_type', type=str, default='lie_log', choices=['euler', 'quaternion', 'lie_log'], help='which type of pose rotational error to use')
    parser.add_argument('--pose_rot_error_module_type', type=str, default='axisangle_module', choices=['euler', 'quaternion', 'lie_log', 'axisangle_module'], help='which type of pose rotational error to use')
    parser.add_argument('--pose_bias', type=float, default=0.01, help='bias to be multiplied to the estimated delta_pose of the model in each iteration.')
    parser.add_argument('--radius', type=int, default=32)

    args = parser.parse_args()

    import importlib

    MODEL = importlib.import_module('drunkards_odometry.model').RAFT3D_pose_v4_PoseCNN_noPre
    model = torch.nn.DataParallel(MODEL(args))
    checkpoint = torch.load(args.ckpt)

    if os.path.basename(args.ckpt) in ['raft3d_laplacian.pth', 'drunkards_odometry.pth']:
        model.load_state_dict(checkpoint, strict=False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.cuda()
    model.eval()

    original_save_path = args.save_path

    for scene in args.test_scenes:
        args.save_path = os.path.join(original_save_path, scene, args.name)
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)

        test(model, scene, args)
