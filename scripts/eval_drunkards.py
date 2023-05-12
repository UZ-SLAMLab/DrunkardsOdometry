import sys

sys.path.append('.')

import argparse
import copy
from data_readers.drunkards import DrunkDataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

    return image1, image2, depth1, depth2


def compute_errors(Ts, pose, flow_gt, depth1, depth2, intrinsics, pose_gt, valid_mask, metrics, count_all):
    """ Loss function defined over sequence of flow predictions """
    # Use transformation field to extract 2D and 3D flow
    flow2d_est, flow3d_est, valid = pops.induced_flow(Ts, depth1, intrinsics)
    valid = valid > 0.5
    valid_mask *= valid.unsqueeze(-1)

    fl_gt, dz_gt = flow_gt.split([2, 1], dim=-1)
    fl_est, dz_est = flow2d_est.split([2, 1], dim=-1)  # fl_est is optical flow 2d in pixels, dz_est is inverse depth

    # Exclude pixels with extreme flow
    mag = torch.sum(fl_gt ** 2, dim=-1).sqrt()
    valid = mag < 250
    valid_mask *= valid.unsqueeze(-1)

    mag = torch.sum(fl_est ** 2, dim=-1).sqrt()
    valid = mag < 250
    valid_mask *= valid.unsqueeze(-1)

    flow3d_gt = pops.backproject_flow3d(fl_gt, depth1, depth2, intrinsics)
    flow3d_tra_error_RMSE = get_flow3d_tra_errors(flow3d_est, flow3d_gt, valid_mask)

    _, pose_tra_error_RMSE, _, pose_rot_error_axisangle_module = get_pose_errors(pose, pose_gt)

    epe_2d = (fl_est - fl_gt).norm(dim=-1)  # Euclidean distance, L2 norm
    epe_2d = epe_2d.view(-1)[valid_mask.view(-1)].double().cpu().numpy()

    count_all += epe_2d.shape[0]  # num of valid pixels

    epe_dz = (dz_est - dz_gt).norm(dim=-1)
    epe_dz = epe_dz.view(-1)[valid_mask.view(-1)].double().cpu().numpy()  # inverse depth change error

    metrics['epe_2d'] += epe_2d.sum()
    metrics['epe_dz'] += epe_dz.sum()
    metrics['epe_2d_1px'] += np.count_nonzero(epe_2d < 1.0)
    metrics['epe_2d_3px'] += np.count_nonzero(epe_2d < 3.0)
    metrics['epe_2d_5px'] += np.count_nonzero(epe_2d < 5.0)
    metrics['flow3d_tra_error_RMSE'] += flow3d_tra_error_RMSE.sum()
    metrics['flow3d_tra_error_1cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .01)
    metrics['flow3d_tra_error_5cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .05)
    metrics['flow3d_tra_error_10cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .1)
    metrics['flow3d_tra_error_20cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .2)
    metrics['pose_tra_error_RMSE'] += pose_tra_error_RMSE
    metrics['pose_rot_error_axisangle_module'] += pose_rot_error_axisangle_module

    return metrics, count_all

@torch.no_grad()
def test(model, scene, args):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
    test_dataset = DrunkDataset(root=args.datapath,
                                 difficulty_level=args.difficulty_level,
                                 do_augment=False,
                                 res_factor=args.res_factor,
                                 scenes_to_use=scene,
                                 depth_augmentor=False,
                                 mode='test')
    test_loader = DataLoader(test_dataset, **loader_args)
    num_frames = len(test_loader)

    count_all = 0
    metrics = {'epe_2d': 0.0, 'epe_dz': 0.0, 'epe_2d_1px': 0.0,
               'epe_2d_3px': 0.0, 'epe_2d_5px': 0.0, 'flow3d_tra_error_RMSE': 0.0,
               'flow3d_tra_error_1cm': 0.0,
               'flow3d_tra_error_5cm': 0.0, 'flow3d_tra_error_10cm': 0.0, 'flow3d_tra_error_20cm': 0.0,
               'pose_tra_error_RMSE': 0.0, 'pose_rot_error_axisangle_module': 0.0
               }

    f_poses = open(os.path.join(args.save_path, 'pose_est.txt'), 'w')
    f_metrics = open(os.path.join(args.save_path, 'metrics.txt'), 'w')

    for i_batch, data_blob in enumerate(tqdm(test_loader)):
        image1, image2, depth1, depth2, pose_gt, intrinsics, flowxyz_gt, valid_mask, depth_scale_factor = [data_item.cuda() for data_item in data_blob]

        # pad and normalize images
        image1, image2, depth1, depth2 = prepare_images_and_depths(image1, image2, depth1, depth2)

        Ts, pose = model(
            **dict(image1=image1, image2=image2, depth1=depth1, depth2=depth2,
                   intrinsics=intrinsics, iters=12, train_mode=False,
                   depth_scale_factor=depth_scale_factor))

        metrics, count_all = compute_errors(Ts, pose, flowxyz_gt, depth1, depth2, intrinsics, pose_gt, valid_mask, metrics, count_all)

        # Write absolut transformation poses in world-to-camera format (T_w_c)
        pose = pose.squeeze().cpu().detach().numpy()
        if i_batch == 0:
            pose1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype="float32")  # tx, ty, tz, qx, qy, qz, qw
            f_poses.writelines('{:010d}'.format(i_batch) + ' ' + str(pose1[0]) + ' ' + str(pose1[1]) + ' ' + str(pose1[2]) + ' ' + str(
                    pose1[3]) + ' ' + str(pose1[4]) + ' ' + str(pose1[5]) + ' ' + str(pose1[6]) + '\n')

        pose2 = pops.relative_to_absoult_poses(pose1, pose)
        f_poses.writelines('{:010d}'.format(i_batch + 1) + ' ' + str(pose2[0]) + ' ' + str(pose2[1]) + ' ' + str(pose2[2]) + ' ' + str(pose2[3]) + ' ' + str(pose2[4]) + ' ' + str(pose2[5]) + ' ' + str(pose2[6]) + '\n')
        pose1 = copy.deepcopy(pose2)

    # Average metrics over all valid pixels
    print("Metrics for the scene " + str(scene) + ":")
    for key in metrics:
        if key in {'pose_tra_error_RMSE', 'pose_rot_error_axisangle_module'}:
            print(key, (metrics[key] / num_frames).item())
            f_metrics.writelines(key + '    ' + str((metrics[key] / num_frames).item()) + '\n')
        else:
            print(key, (metrics[key] / count_all).item())
            f_metrics.writelines(key + '    ' + str((metrics[key] / count_all).item()) + '\n')

    f_poses.close()
    f_metrics.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='name your experiment')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datapath', type=str, required=True, help='full path to folder containing the scenes')
    parser.add_argument('--difficulty_level', type=int, choices=[0, 1, 2, 3],
                        help='drunk dataset diffculty level to use')
    parser.add_argument('--res_factor', type=int, default=1, help='reduce resolution by a factor')
    parser.add_argument('--test_scenes', type=int, nargs='+', default=[0, 4, 5], help='scenes used for testing')
    parser.add_argument("--save_path", type=str, help="specify full path. Results will be saved here")
    parser.add_argument('--pose_bias', type=float, default=0.01, help='bias to be multiplied to the estimated delta_pose of the model in each iteration.')
    parser.add_argument('--radius', type=int, default=32)

    args = parser.parse_args()

    import importlib

    model = importlib.import_module('drunkards_odometry.model').DrunkardsOdometry
    # model = torch.nn.DataParallel(model(args))  #todo igual no hace falta
    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.cuda()
    model.eval()

    if args.save_path:
        original_save_path = args.save_path
    else:
        original_save_path = os.path.join(os.getcwd(), 'evaluations_drunk')  # todo chequear que el os.getcwd() me devuelve el path padre de este script, es decri, el main folder

    for scene in args.test_scenes:
        args.save_path = os.path.join(original_save_path, "{:05d}".format(scene), "level{}".format(args.difficulty_level), args.name)
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)

        test(model, scene, args)
