import sys

sys.path.append('.')

import argparse
import copy
from data_readers.drunkards import DrunkDataset
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

def dense_from_quat_to_euler(Ts):
    batch_size, ht, wd = Ts.shape
    device = Ts.device
    twist = Ts.matrix()

    twist = twist.view((-1, 4, 4))
    twist_euler = pops.pose_from_matrix_to_euler(twist).view((batch_size, ht, wd, 6)).to(device)

    return twist_euler


def compute_errors(flow2d_est, pose, flow_gt, depth1, depth2, intrinsics, pose_gt, valid_mask, metrics, count_all, f_poses_errors):
    """ Loss function defined over sequence of flow predictions """

    fl_gt, dz_gt = flow_gt.split([2, 1], dim=-1)
    fl_est, dz_est = flow2d_est.split([2, 1], dim=-1)  # fl_est is optical flow 2d in pixels, dz_est is inverse depth

    flow3d_gt = pops.backproject_flow3d(fl_gt, depth1, depth2, intrinsics)
    flow3d_est = pops.backproject_flow3d(fl_est, depth1, depth2, intrinsics)

    mag = torch.sum(fl_gt ** 2, dim=-1).sqrt()
    valid = mag < 250
    valid_mask *= valid.unsqueeze(-1)

    mag = torch.sum(fl_est ** 2, dim=-1).sqrt()
    valid = mag < 250
    valid_mask *= valid.unsqueeze(-1)

    flow3d_tra_error_RMSE = get_flow3d_tra_errors(flow3d_est, flow3d_gt, valid_mask)

    pose_tra_error_ME, pose_tra_error_RMSE, pose_rot_error_ME, pose_rot_error_axisangle_module = get_pose_errors(pose, pose_gt)

    epe_2d = (fl_est - fl_gt).norm(dim=-1)  # Euclidean distance, L2 norm
    epe_2d = epe_2d.view(-1)[valid_mask.view(-1)].double().cpu().numpy()

    count_all += epe_2d.shape[0]  # num of valid pixels

    epe_2d_rel = (fl_est - fl_gt).norm(dim=-1) / fl_gt.norm(dim=-1)  # relative Euclidean distance, L2 norm
    epe_2d_rel = epe_2d_rel.view(-1)[valid_mask.view(-1)].double().cpu().numpy()

    epe_dz = (dz_est - dz_gt).norm(dim=-1)
    epe_dz = epe_dz.view(-1)[valid_mask.view(-1)].double().cpu().numpy()  # inverse depth change error

    epe_dz_rel = ((dz_est - dz_gt).norm(dim=-1).view(-1)[(valid_mask * (dz_gt != 0)).view(-1)] / dz_gt.norm(dim=-1).view(-1)[(valid_mask * (dz_gt != 0)).view(-1)]).double().cpu().numpy()

    metrics['epe_2d'] += epe_2d.sum()
    metrics['epe_2d_rel'] += epe_2d_rel.sum()
    metrics['epe_dz'] += epe_dz.sum()
    metrics['epe_dz_rel'] += epe_dz_rel.sum()
    metrics['epe_2d_1px'] += np.count_nonzero(epe_2d < 1.0)
    metrics['epe_2d_3px'] += np.count_nonzero(epe_2d < 3.0)
    metrics['epe_2d_5px'] += np.count_nonzero(epe_2d < 5.0)

    metrics['flow3d_tra_error_RMSE'] += flow3d_tra_error_RMSE.sum()
    metrics['flow3d_tra_error_1cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .01)
    metrics['flow3d_tra_error_5cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .05)
    metrics['flow3d_tra_error_10cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .1)
    metrics['flow3d_tra_error_20cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .2)

    metrics['pose_tra_error_ME'] += pose_tra_error_ME
    metrics['pose_tra_error_RMSE'] += pose_tra_error_RMSE
    metrics['pose_rot_error_ME'] += pose_rot_error_ME
    metrics['pose_rot_error_axisangle_module'] += pose_rot_error_axisangle_module

    metrics['pose_tra_error_RMSE_median'].append(pose_tra_error_RMSE)
    metrics['pose_rot_error_axisangle_module_median'].append(pose_rot_error_axisangle_module)

    f_poses_errors.writelines(str(pose_tra_error_RMSE.item()) + ' ' + str(pose_rot_error_axisangle_module.item()) + '\n')

    return metrics, count_all

def compute_errors_only_flow(flow2d_est, Ts, flow_gt, Ts_gt, depth1, depth2, intrinsics, valid_mask, metrics, count_all):
    """ Loss function defined over sequence of flow predictions """

    fl_gt, dz_gt = flow_gt.split([2, 1], dim=-1)
    fl_est, dz_est = flow2d_est.split([2, 1], dim=-1)  # fl_est is optical flow 2d in pixels, dz_est is inverse depth

    flow3d_gt = pops.backproject_flow3d(fl_gt, depth1, depth2, intrinsics)
    flow3d_est = pops.backproject_flow3d(fl_est, depth1, depth2, intrinsics)

    mag = torch.sum(fl_gt ** 2, dim=-1).sqrt()
    valid = mag < 250
    valid_mask *= valid.unsqueeze(-1)

    mag = torch.sum(fl_est ** 2, dim=-1).sqrt()
    valid = mag < 250
    valid_mask *= valid.unsqueeze(-1)

    flow3d_tra_error_RMSE = get_flow3d_tra_errors(flow3d_est, flow3d_gt, valid_mask)

    epe_2d = (fl_est - fl_gt).norm(dim=-1)  # Euclidean distance, L2 norm
    epe_2d = epe_2d.view(-1)[valid_mask.view(-1)].double().cpu().numpy()

    flow_est_module = fl_est.norm(dim=-1)
    flow_est_module = flow_est_module.view(-1)[valid_mask.view(-1)]
    flow_est_module_mean = flow_est_module.mean()
    flow_est_module_min = flow_est_module.min()
    flow_est_module_max = flow_est_module.max()
    flow_gt_module = fl_gt.norm(dim=-1)
    flow_gt_module = flow_gt_module.view(-1)[valid_mask.view(-1)]
    flow_gt_module_mean = flow_gt_module.mean()
    flow_gt_module_min = flow_gt_module.min()
    flow_gt_module_max = flow_gt_module.max()

    count_all += epe_2d.shape[0]  # num of valid pixels

    epe_2d_rel = (fl_est - fl_gt).norm(dim=-1) / fl_gt.norm(dim=-1)  # relative Euclidean distance, L2 norm
    epe_2d_rel = epe_2d_rel.view(-1)[valid_mask.view(-1)].double().cpu().numpy()

    epe_dz = (dz_est - dz_gt).norm(dim=-1)
    epe_dz = epe_dz.view(-1)[valid_mask.view(-1)].double().cpu().numpy()  # inverse depth change error

    epe_dz_rel = ((dz_est - dz_gt).norm(dim=-1).view(-1)[(valid_mask * (dz_gt != 0)).view(-1)] / dz_gt.norm(dim=-1).view(-1)[(valid_mask * (dz_gt != 0)).view(-1)]).double().cpu().numpy()

    metrics['epe_2d'] += epe_2d.sum()
    metrics['epe_2d_rel'] += epe_2d_rel.sum()
    metrics['epe_dz'] += epe_dz.sum()
    metrics['epe_dz_rel'] += epe_dz_rel.sum()
    metrics['epe_2d_1px'] += np.count_nonzero(epe_2d < 1.0)
    metrics['epe_2d_3px'] += np.count_nonzero(epe_2d < 3.0)
    metrics['epe_2d_5px'] += np.count_nonzero(epe_2d < 5.0)

    metrics['flow3d_tra_error_ME'] += flow3d_tra_error_ME.sum()
    metrics['flow3d_tra_error_RMSE'] += flow3d_tra_error_RMSE.sum()
    # metrics['flow3d_rot_error_ME'] += flow3d_rot_error_ME.sum()
    # metrics['flow3d_rot_error_RMSE'] += flow3d_rot_error_RMSE.sum()
    metrics['flow3d_tra_error_ME_rel'] += flow3d_tra_error_ME_rel.sum()
    # metrics['flow3d_rot_error_ME_rel'] += flow3d_rot_error_ME_rel.sum()
    metrics['flow3d_tra_error_RMSE_rel'] += flow3d_tra_error_RMSE_rel.sum()
    # metrics['flow3d_rot_error_RMSE_rel'] += flow3d_rot_error_RMSE_rel.sum()
    # metrics['flow3d_rot_error_axisangle_module'] += flow3d_rot_error_axisangle_module.sum()
    # metrics['flow3d_rot_error_axisangle_module_rel'] += flow3d_rot_error_axisangle_module_rel.sum()
    metrics['flow3d_tra_error_1cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .01)
    metrics['flow3d_tra_error_5cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .05)
    metrics['flow3d_tra_error_10cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .1)
    metrics['flow3d_tra_error_20cm'] += torch.count_nonzero(flow3d_tra_error_RMSE < .2)

    # metrics['flow_est_module_mean'].append(flow_est_module_mean)
    # metrics['flow_est_module_max'].append(flow_est_module_max)
    # metrics['flow_est_module_min'].append(flow_est_module_min)
    # metrics['flow_gt_module_mean'].append(flow_gt_module_mean)
    # metrics['flow_gt_module_max'].append(flow_gt_module_max)
    # metrics['flow_gt_module_min'].append(flow_gt_module_min)

    return metrics, count_all

def compute_errors_only_pose(pose, pose_gt, metrics):
    pose_tra_error, pose_rot_error, pose_tra_rel_error, pose_rot_rel_error = get_pose_errors(pose, pose_gt, alpha=0.5)
    pose_module, pose_module_gt, pose_module_error, pose_module_error_rel, pose_direction_x, pose_direction_y, pose_direction_z, pose_direction_x_gt, pose_direction_y_gt, pose_direction_z_gt, pose_direction_x_error, pose_direction_y_error, pose_direction_z_error, pose_direction_x_error_rel, pose_direction_y_error_rel, pose_direction_z_error_rel = get_pose_module_and_direction_errors(pose[:, :3], pose_gt[:, :3])

    pose_tra_rel_error_x = ((pose[:, 0] - pose_gt[:, 0]) / abs(pose_gt[:, 0])).double().cpu().numpy()
    pose_tra_rel_error_y = ((pose[:, 1] - pose_gt[:, 1]) / abs(pose_gt[:, 1])).double().cpu().numpy()
    pose_tra_rel_error_z = ((pose[:, 2] - pose_gt[:, 2]) / abs(pose_gt[:, 2])).double().cpu().numpy()

    metrics['pose_tra_error'] += pose_tra_error
    metrics['pose_rot_error'] += pose_rot_error
    metrics['pose_tra_rel_error'] += pose_tra_rel_error
    metrics['pose_rot_rel_error'] += pose_rot_rel_error
    metrics['pose_tra_rel_error_x'] += pose_tra_rel_error_x
    metrics['pose_tra_rel_error_y'] += pose_tra_rel_error_y
    metrics['pose_tra_rel_error_z'] += pose_tra_rel_error_z
    metrics['pose_module'] += pose_module
    metrics['pose_module_gt'] += pose_module_gt
    metrics['pose_module_error'] += pose_module_error
    metrics['pose_module_error_rel'] += pose_module_error_rel
    metrics['pose_direction_x'] += pose_direction_x
    metrics['pose_direction_y'] += pose_direction_y
    metrics['pose_direction_z'] += pose_direction_z
    metrics['pose_direction_x_gt'] += pose_direction_x_gt
    metrics['pose_direction_y_gt'] += pose_direction_y_gt
    metrics['pose_direction_z_gt'] += pose_direction_z_gt
    metrics['pose_direction_x_error'] += pose_direction_x_error
    metrics['pose_direction_y_error'] += pose_direction_y_error
    metrics['pose_direction_z_error'] += pose_direction_z_error
    metrics['pose_direction_x_error_rel'] += pose_direction_x_error_rel
    metrics['pose_direction_y_error_rel'] += pose_direction_y_error_rel
    metrics['pose_direction_z_error_rel'] += pose_direction_z_error_rel

    return metrics


@torch.no_grad()
def test(model, scene, args):
    loader_args = {'batch_size': 1, 'shuffle': False, 'num_workers': 0, 'drop_last': False}
    test_dataset = DrunkDataset(do_augment=False,
                                 root=args.datapath,
                                 difficulty_level=args.difficulty_level,
                                 res_factor=args.res_factor,
                                 scenes_to_use=scene,
                                 depth_augmentor=False,
                                 mode='test')
    test_loader = DataLoader(test_dataset, **loader_args)

    count_all = 0
    if args.method == 'only_poseCNN':
        metrics = {'pose_tra_error': 0.0, 'pose_rot_error': 0.0, 'pose_tra_rel_error': 0.0, 'pose_rot_rel_error': 0.0,
                   'pose_tra_rel_error_x': 0.0, 'pose_tra_rel_error_y': 0.0, 'pose_tra_rel_error_z': 0.0,
                   'pose_module': 0.0, 'pose_module_gt': 0.0, 'pose_module_error': 0.0, 'pose_module_error_rel': 0.0,
                   'pose_direction_x': 0.0, 'pose_direction_y': 0.0, 'pose_direction_z': 0.0,
                   'pose_direction_x_gt': 0.0, 'pose_direction_y_gt': 0.0, 'pose_direction_z_gt': 0.0,
                   'pose_direction_x_error': 0.0, 'pose_direction_y_error': 0.0, 'pose_direction_z_error': 0.0,
                   'pose_direction_x_error_rel': 0.0, 'pose_direction_y_error_rel': 0.0,
                   'pose_direction_z_error_rel': 0.0}
    else:
        # metrics = {'epe_2d': 0.0, 'epe_2d_rel': 0.0, 'epe_dz': 0.0, 'epe_dz_rel': 0.0, 'epe_2d_1px': 0.0, 'epe_2d_3px': 0.0, 'epe_2d_5px': 0.0, 'flow3d_tra_error_ME': 0.0, 'flow3d_tra_error_RMSE': 0.0, 'flow3d_rot_error_ME': 0.0, 'flow3d_rot_error_RMSE': 0.0, 'flow3d_tra_error_ME_rel': 0.0, 'flow3d_rot_error_ME_rel': 0.0, 'flow3d_tra_error_RMSE_rel': 0.0, 'flow3d_rot_error_RMSE_rel': 0.0, 'flow3d_rot_error_axisangle_module': 0.0, 'flow3d_rot_error_axisangle_module_rel': 0.0, 'flow3d_tra_error_5cm': 0.0, 'flow3d_tra_error_10cm': 0.0, 'flow3d_tra_error_20cm': 0.0, 'flow3d_def_tra_error_ME': 0.0, 'flow3d_def_tra_error_RMSE': 0.0, 'flow3d_def_rot_error_ME': 0.0, 'flow3d_def_rot_error_RMSE': 0.0, 'flow3d_def_tra_error_ME_rel': 0.0, 'flow3d_def_rot_error_ME_rel': 0.0, 'flow3d_def_tra_error_RMSE_rel': 0.0, 'flow3d_def_rot_error_RMSE_rel': 0.0, 'flow3d_def_rot_error_axisangle_module': 0.0, 'flow3d_def_rot_error_axisangle_module_rel': 0.0, 'flow3d_def_tra_error_5cm': 0.0, 'flow3d_def_tra_error_10cm': 0.0, 'flow3d_def_tra_error_20cm': 0.0, 'pose_tra_error_ME': 0.0, 'pose_tra_error_RMSE': 0.0, 'pose_rot_error_ME': 0.0, 'pose_rot_error_RMSE': 0.0, 'pose_tra_error_ME_rel': 0.0, 'pose_tra_error_RMSE_rel': 0.0, 'pose_rot_error_ME_rel': 0.0, 'pose_rot_error_RMSE_rel': 0.0, 'pose_rot_error_axisangle_module': 0.0, 'pose_rot_error_axisangle_module_rel': 0.0, 'pose_tra_module': 0.0, 'pose_tra_module_gt': 0.0, 'pose_tra_x_rel': 0.0, 'pose_tra_y_rel': 0.0, 'pose_tra_z_rel': 0.0, 'pose_tra_x_gt_rel': 0.0, 'pose_tra_y_gt_rel': 0.0, 'pose_tra_z_gt_rel': 0.0, 'pose_tra_x_error': 0.0, 'pose_tra_y_error': 0.0, 'pose_tra_z_error': 0.0, 'pose_tra_x_error_rel': 0.0, 'pose_tra_y_error_rel': 0.0, 'pose_tra_z_error_rel': 0.0, 'pose_tra_error_RMSE_median': [], 'pose_rot_error_axisangle_module_median': [], 'pose_tra_error_RMSE_rel_median': [], 'pose_rot_error_axisangle_module_rel_median': [], 'flow_est_module_mean': [], 'flow_est_module_max': [], 'flow_est_module_min': [], 'flow_gt_module_mean': [], 'flow_gt_module_min': [], 'flow_gt_module_max': [], 'pose_tra_module_list': [], 'pose_tra_module_gt_list': []}
        metrics = {'epe_2d': 0.0, 'epe_2d_rel': 0.0, 'epe_dz': 0.0, 'epe_dz_rel': 0.0, 'epe_2d_1px': 0.0,
                   'epe_2d_3px': 0.0, 'epe_2d_5px': 0.0, 'flow3d_tra_error_ME': 0.0, 'flow3d_tra_error_RMSE': 0.0,
                   'flow3d_tra_error_ME_rel': 0.0,
                   'flow3d_tra_error_RMSE_rel': 0.0, 'flow3d_tra_error_1cm': 0.0,
                   'flow3d_tra_error_5cm': 0.0, 'flow3d_tra_error_10cm': 0.0, 'flow3d_tra_error_20cm': 0.0,
                   'pose_tra_error_ME': 0.0,
                   'pose_tra_error_RMSE': 0.0, 'pose_rot_error_ME': 0.0, 'pose_rot_error_RMSE': 0.0,
                   'pose_tra_error_ME_rel': 0.0, 'pose_tra_error_RMSE_rel': 0.0, 'pose_rot_error_ME_rel': 0.0,
                   'pose_rot_error_RMSE_rel': 0.0, 'pose_rot_error_axisangle_module': 0.0,
                   'pose_rot_error_axisangle_module_rel': 0.0, 'pose_tra_module': 0.0, 'pose_tra_module_gt': 0.0,
                   'pose_tra_x_rel': 0.0, 'pose_tra_y_rel': 0.0, 'pose_tra_z_rel': 0.0, 'pose_tra_x_gt_rel': 0.0,
                   'pose_tra_y_gt_rel': 0.0, 'pose_tra_z_gt_rel': 0.0, 'pose_tra_x_error': 0.0, 'pose_tra_y_error': 0.0,
                   'pose_tra_z_error': 0.0, 'pose_tra_x_error_rel': 0.0, 'pose_tra_y_error_rel': 0.0,
                   'pose_tra_z_error_rel': 0.0, 'pose_tra_error_RMSE_median': [],
                   'pose_rot_error_axisangle_module_median': [], 'pose_tra_error_RMSE_rel_median': [],
                   'pose_rot_error_axisangle_module_rel_median': [], 'pose_tra_module_list': [],
                   'pose_tra_module_gt_list': []}

    f_poses = open(os.path.join(args.save_path, 'stamped_traj_estimate.txt'), 'w')
    f_metrics = open(os.path.join(args.save_path, 'metrics.txt'), 'w')
    f_poses_errors = open(os.path.join(args.save_path, 'poses_errors.txt'), 'w')
    f_poses_errors.writelines('pose_tra_error_RMSE pose_rot_error_axisangle_module' + '\n')

    for i_batch, test_data_blob in enumerate(tqdm(test_loader)):
        num_frames = len(test_loader)

        image1, image2, depth1, depth2, pose_gt, intrinsics, flowxyz_gt, Ts_gt, Ts_def_gt, valid_mask, depth_scale_factor = [data_item.cuda() for data_item in test_data_blob]

        # pad and normalize images  #todo para que hace el padding?
        image1, image2, depth1, depth2, padding = prepare_images_and_depths(image1, image2, depth1, depth2)

        Ts, Ts_def, pose = model(
            **dict(image1=image1, image2=image2, depth1=depth1, depth2=depth2,
                   intrinsics=intrinsics, valid_mask=valid_mask, iters=12, train_mode=False,
                   depth_scale_factor=depth_scale_factor,
                   pose_bias=args.pose_bias))

        # use transformation field to extract 2D and 3D flow
        flow2d_est, flow3d_est, valid = pops.induced_flow(Ts, depth1, intrinsics)
        valid = valid > 0.5
        valid_mask *= valid.unsqueeze(-1)

        metrics, count_all = compute_errors(flow2d_est, pose, flowxyz_gt, depth1, depth2, intrinsics, pose_gt, valid_mask, metrics=metrics, count_all=count_all, f_poses_errors=f_poses_errors)

        if args.method != 'raft3D_def':
            pose = pose.squeeze().cpu().detach().numpy()

            if i_batch == 0:

                pose1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype="float32")
                f_poses.writelines('{:010d}'.format(i_batch) + ' ' + str(pose1[0]) + ' ' + str(pose1[1]) + ' ' + str(pose1[2]) + ' ' + str(
                        pose1[3]) + ' ' + str(pose1[4]) + ' ' + str(pose1[5]) + ' ' + str(pose1[6]) + '\n')

            pose2 = pops.relative_to_absoult_poses(pose1, pose)

            f_poses.writelines('{:010d}'.format(i_batch + 1) + ' ' + str(pose2[0]) + ' ' + str(pose2[1]) + ' ' + str(pose2[2]) + ' ' + str(pose2[3]) + ' ' + str(pose2[4]) + ' ' + str(pose2[5]) + ' ' + str(pose2[6]) + '\n')

            pose1 = copy.deepcopy(pose2)

    # Average results over all valid pixels
    print("Metrics scene " + str(scene) + ":")
    for key in metrics:
        if args.method != 'raft3D_def' and key in ['pose_tra_error_ME', 'pose_tra_error_RMSE', 'pose_rot_error_ME', 'pose_rot_error_RMSE', 'pose_tra_error_ME_rel', 'pose_tra_error_RMSE_rel', 'pose_rot_error_ME_rel', 'pose_rot_error_RMSE_rel', 'pose_rot_error_axisangle_module', 'pose_rot_error_axisangle_module_rel', 'pose_tra_module', 'pose_tra_module_gt', 'pose_tra_x_rel', 'pose_tra_y_rel', 'pose_tra_z_rel', 'pose_tra_x_gt_rel', 'pose_tra_y_gt_rel', 'pose_tra_z_gt_rel', 'pose_tra_x_error', 'pose_tra_y_error', 'pose_tra_z_error', 'pose_tra_x_error_rel', 'pose_tra_y_error_rel', 'pose_tra_z_error_rel']:
            print(key, (metrics[key] / num_frames).item())
            f_metrics.writelines(key + '    ' + str((metrics[key] / num_frames).item()) + '\n')
        elif args.method != 'raft3D_def' and key in ['pose_tra_error_RMSE_median', 'pose_rot_error_axisangle_module_median', 'pose_tra_error_RMSE_rel_median', 'pose_rot_error_axisangle_module_rel_median']:
            print(key[:-6] + 'mean', str(torch.mean(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key[:-6] + 'mean' + '    ' + str(torch.mean(torch.tensor(metrics[key])).item()) + '\n')

            print(key, str(torch.median(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key + '    ' + str(torch.median(torch.tensor(metrics[key])).item()) + '\n')

            print(key[:-6] + 'min', str(torch.min(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key[:-6] + 'min' + '    ' + str(torch.min(torch.tensor(metrics[key])).item()) + '\n')

            print(key[:-6] + 'max', str(torch.max(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key[:-6] + 'max' + '    ' + str(torch.max(torch.tensor(metrics[key])).item()) + '\n')

            print(key[:-6] + 'std', str(torch.std(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key[:-6] + 'std' + '    ' + str(torch.std(torch.tensor(metrics[key])).item()) + '\n')

        elif args.method != 'raft3D_def' and key in ['pose_tra_module_list', 'pose_tra_module_gt_list']:
            print(key[:-4] + 'mean', str(torch.mean(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key[:-4] + 'mean' + '    ' + str(torch.mean(torch.tensor(metrics[key])).item()) + '\n')

            print(key[:-4] + 'median', str(torch.median(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key[:-4] + 'median' + '    ' + str(torch.median(torch.tensor(metrics[key])).item()) + '\n')

            print(key[:-4] + 'min', str(torch.min(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key[:-4] + 'min' + '    ' + str(torch.min(torch.tensor(metrics[key])).item()) + '\n')

            print(key[:-4] + 'max', str(torch.max(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key[:-4] + 'max' + '    ' + str(torch.max(torch.tensor(metrics[key])).item()) + '\n')

            print(key[:-4] + 'std', str(torch.std(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key[:-4] + 'std' + '    ' + str(torch.std(torch.tensor(metrics[key])).item()) + '\n')

        elif key in ['flow_est_module_mean', 'flow_est_module_max', 'flow_est_module_min', 'flow_gt_module_mean', 'flow_gt_module_max', 'flow_gt_module_min']:
            print(key + '_mean', str(torch.mean(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key + '_mean' + '    ' + str(torch.mean(torch.tensor(metrics[key])).item()) + '\n')

            print(key + '_median', str(torch.median(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key + '_median' + '    ' + str(torch.median(torch.tensor(metrics[key])).item()) + '\n')

            print(key + '_min', str(torch.min(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key + '_min' + '    ' + str(torch.min(torch.tensor(metrics[key])).item()) + '\n')

            print(key + '_max', str(torch.max(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key + '_max' + '    ' + str(torch.max(torch.tensor(metrics[key])).item()) + '\n')

            print(key + '_std', str(torch.std(torch.tensor(metrics[key])).item()))
            f_metrics.writelines(key + '_std' + '    ' + str(torch.std(torch.tensor(metrics[key])).item()) + '\n')

        else:
            if 'flow3d' in key:
                if (args.method != 'raft3D_def') or (args.method == 'raft3D_def' and 'flow3d_def' not in key):
                    print(key, (metrics[key] / count_all).item())
                    f_metrics.writelines(key + '    ' + str((metrics[key] / count_all).item()) + '\n')
            elif args.method != 'raft3D_def' or (args.method == 'raft3D_def' and 'pose' not in key and '_def' not in key):
                print(key, metrics[key] / count_all)
                f_metrics.writelines(key + '    ' + str(metrics[key] / count_all) + '\n')

    f_poses.close()
    f_metrics.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='name your experiment')
    parser.add_argument('--network', default='drunkards_odometry.model')
    parser.add_argument('--ckpt', help='checkpoint to restore')
    parser.add_argument('--datapath', type=str, required=True, help='full path to folder containing the scenes')
    parser.add_argument('--difficulty_level', type=int, choices=[0, 1, 2, 3],
                        help='drunk dataset diffculty level to use')
    parser.add_argument('--res_factor', type=int, default=1, help='reduce resolution by a factor')
    parser.add_argument('--test_scenes', type=int, nargs='+', default=[0, 4, 5], help='scenes used for training')
    parser.add_argument("--save_path", type=str, default='/media/david/DiscoDuroLinux/Datasets/evaluations_drunk', help="if specified, logs and results will be saved here")
    parser.add_argument('--pose_bias', type=float, default=0.01, help='bias to be multiplied to the estimated delta_pose of the model in each iteration.')
    parser.add_argument('--radius', type=int, default=32)

    args = parser.parse_args()

    import importlib

    MODEL = importlib.import_module('drunkards_odometry.model').DrunkardsOdometry
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
        args.save_path = os.path.join(original_save_path, "{:05d}".format(scene), "level{}".format(args.difficulty_level), args.name)
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)

        test(model, scene, args)
