import sys
import argparse
import torch.optim as optim
import re

from autoclip.torch import QuantileClip
from torch.utils.data import DataLoader
from utils import *
from data_readers.drunkards import DrunkDataset

sys.path.append('.')


def loss_fn(flow2d_est, flow2d_rev, pose_list, flow_gt, depth1, depth2, intrinsics, pose_gt, valid_mask, args, mode,
            gamma=0.9):
    """ Loss function defined over sequence of flow predictions """
    fl_weight = args.fl_weight
    rv_weight = args.rv_weight
    dz_weight = args.dz_weight
    relative_tra_rot_weight = args.relative_tra_rot_weight
    pose_weight = args.pose_weight
    pose_cnn_weight = args.pose_cnn_weight

    N = len(flow2d_est)  # Number of iterations of the update block
    loss = 0.0
    fl_gt, dz_gt = flow_gt.split([2, 1], dim=-1)

    for i in range(N):
        w = gamma ** (N - i - 1)

        fl_est, dz_est = flow2d_est[i].split([2, 1], dim=-1)  # fl_est optical flow 2d in pixels, dz_est inverse depth

        # Optical flow loss after Gauss Newton step
        # Idea of using L1 Charbonnier loss SMURF: Self-Teaching Multi-Frame Unsupervised RAFT with Full-Image Warping
        fl_loss = valid_mask * L1_Charbonnier_loss(fl_est, fl_gt)
        fl_loss = fl_loss[fl_loss.nonzero(as_tuple=True)].mean()

        # Inverse depth loss
        dz_loss = valid_mask * L1_Charbonnier_loss(dz_est, dz_gt)
        dz_loss = dz_loss[dz_loss.nonzero(as_tuple=True)].mean()

        # Optical flow loss before Gauss Newton step
        fl_rev = flow2d_rev[i]
        rv_loss = valid_mask * L1_Charbonnier_loss(fl_rev, fl_gt)
        rv_loss = rv_loss[rv_loss.nonzero(as_tuple=True)].mean()

        # Relative camera pose loss
        if isinstance(pose_list, list):
            pose = pose_list[i]
            if i == 0:
                pose_cnn = pose_list[N]
        else:
            pose = pose_list

        pose_tra_error_ME, pose_tra_error_RMSE, pose_rot_error_ME, pose_rot_error_axisangle_module = \
            get_pose_errors(pose, pose_gt)
        pose_tra_loss = pose_tra_error_ME
        pose_rot_loss = pose_rot_error_ME

        if i == 0:
            pose_cnn_tra_error_ME, pose_cnn_tra_error_RMSE, pose_cnn_rot_error_ME, pose_cnn_rot_error_axisangle_module \
                = get_pose_errors(
                pose_cnn, pose_gt)
            pose_cnn_tra_loss = pose_cnn_tra_error_ME
            pose_cnn_rot_loss = pose_cnn_rot_error_ME

        loss_fl = w * fl_weight * fl_loss
        loss_dz = w * dz_weight * dz_loss
        loss_rv = w * rv_weight * rv_loss
        loss += loss_fl + loss_dz + loss_rv
        loss_pose_tra = w * pose_weight * pose_tra_loss
        loss_pose_rot = w * pose_weight * relative_tra_rot_weight * pose_rot_loss
        loss += loss_pose_tra + loss_pose_rot

        if i == 0:
            loss_pose_cnn_tra = w * pose_weight * pose_cnn_weight * pose_cnn_tra_loss
            loss_pose_cnn_rot = w * pose_weight * pose_cnn_weight * relative_tra_rot_weight * pose_cnn_rot_loss
            loss += loss_pose_cnn_tra + loss_pose_cnn_rot

    epe_2d = (fl_est - fl_gt).norm(dim=-1)  # Euclidean distance, L2 norm
    epe_2d = epe_2d.view(-1)[valid_mask.view(-1)]

    epe_dz = (dz_est - dz_gt).norm(dim=-1)
    epe_dz = epe_dz.view(-1)[valid_mask.view(-1)]  # inverse depth change error

    flow3d_gt = pops.backproject_flow3d(fl_gt, depth1, depth2, intrinsics)
    flow3d_est = pops.backproject_flow3d(fl_est, depth1, depth2, intrinsics)
    flow3d_tra_error_RMSE = get_flow3d_tra_errors(flow3d_est, flow3d_gt, valid_mask)
    flow3d_tra_error_1cm = torch.count_nonzero(flow3d_tra_error_RMSE < .01) / flow3d_tra_error_RMSE.size(0)
    flow3d_tra_error_5cm = torch.count_nonzero(flow3d_tra_error_RMSE < .05) / flow3d_tra_error_RMSE.size(0)
    flow3d_tra_error_10cm = torch.count_nonzero(flow3d_tra_error_RMSE < .1) / flow3d_tra_error_RMSE.size(0)
    flow3d_tra_error_20cm = torch.count_nonzero(flow3d_tra_error_RMSE < .2) / flow3d_tra_error_RMSE.size(0)

    suffix = '_val' if mode == 'val' else ''
    metrics = {}
    metrics['epe_2d' + suffix] = epe_2d.mean().item()
    metrics['epe_dz' + suffix] = epe_dz.mean().item()
    metrics['1px' + suffix] = (epe_2d < 1).float().mean().item()
    metrics['3px' + suffix] = (epe_2d < 3).float().mean().item()
    metrics['5px' + suffix] = (epe_2d < 5).float().mean().item()
    metrics['loss' + suffix] = loss
    metrics['loss_fl' + suffix] = loss_fl
    metrics['loss_dz' + suffix] = loss_dz
    metrics['loss_rv' + suffix] = loss_rv
    metrics['loss_pose_tra' + suffix] = loss_pose_tra
    metrics['loss_pose_rot' + suffix] = loss_pose_rot
    metrics['loss_pose_cnn_tra' + suffix] = loss_pose_cnn_tra
    metrics['loss_pose_cnn_rot' + suffix] = loss_pose_cnn_rot
    metrics['pose_cnn_tra_error_ME' + suffix] = pose_cnn_tra_error_ME.item()
    metrics['pose_cnn_tra_error_RMSE' + suffix] = pose_cnn_tra_error_RMSE.item()
    metrics['pose_cnn_rot_error_ME' + suffix] = pose_cnn_rot_error_ME.item()
    metrics['pose_cnn_rot_error_axisangle_module' + suffix] = pose_cnn_rot_error_axisangle_module.item()
    metrics['flow3d_tra_error_RMSE' + suffix] = flow3d_tra_error_RMSE.mean().item()
    metrics['flow3d_tra_error_1cm' + suffix] = flow3d_tra_error_1cm.item()
    metrics['flow3d_tra_error_5cm' + suffix] = flow3d_tra_error_5cm.item()
    metrics['flow3d_tra_error_10cm' + suffix] = flow3d_tra_error_10cm.item()
    metrics['flow3d_tra_error_20cm' + suffix] = flow3d_tra_error_20cm.item()
    metrics['pose_tra_error_ME' + suffix] = pose_tra_error_ME.item()
    metrics['pose_tra_error_RMSE' + suffix] = pose_tra_error_RMSE.item()
    metrics['pose_rot_error_ME' + suffix] = pose_rot_error_ME.item()
    metrics['pose_rot_error_axisangle_module' + suffix] = pose_rot_error_axisangle_module.item()

    return (loss, metrics) if mode == 'train' else metrics


def fetch_dataloader(args):
    gpuargs = {'shuffle': True, 'num_workers': args.num_workers, 'drop_last': True}
    train_dataset = DrunkDataset(root=args.datapath,
                                 difficulty_level=args.difficulty_level,
                                 do_augment=True,
                                 res_factor=args.res_factor,
                                 scenes_to_use=args.train_scenes,
                                 depth_augmentor=args.depth_augmentor,
                                 invert_order_prob=args.invert_order_prob,
                                 mode='train')
    val_dataset = DrunkDataset(root=args.datapath,
                               difficulty_level=args.difficulty_level,
                               do_augment=False,
                               res_factor=args.res_factor,
                               scenes_to_use=args.val_scenes,
                               depth_augmentor=False,
                               mode='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, **gpuargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, **gpuargs)
    args.num_steps = int(args.num_epochs * len(train_loader))

    return train_loader, val_loader


def fetch_optimizer(model, args):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.00001)
    clipper = QuantileClip(model.parameters(), quantile=0.9, history_length=1000)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps, pct_start=0.001, cycle_momentum=False)
    return optimizer, scheduler, clipper

def optimizer_to(optim, device):
    """https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385"""
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def progressbar(current_value, total_value, bar_lengh, progress_char):
    """https://stackoverflow.com/a/75033230"""
    percentage = int((current_value / total_value) * 100)
    progress = int((bar_lengh * current_value) / total_value)
    loadbar = "Progress: [{:{len}}]{}%".format(progress * progress_char, percentage, len=bar_lengh)
    print(loadbar, end='\r')


def train(args):
    import importlib

    MODEL = importlib.import_module('drunkards_odometry.model').DrunkardsOdometry
    model = MODEL(args)
    device = torch.device("cuda")

    train_loader, val_loader = fetch_dataloader(args)
    val_iter = iter(val_loader)
    optimizer, scheduler, clipper = fetch_optimizer(model, args)

    start = 0
    start_epoch = 0
    total_steps = 0
    create_new_name = True

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = os.getcwd()

    if args.continue_training_from_ckpt:
        assert args.ckpt, "To continue the training, a checkpoint must be provided."

    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if args.continue_training_from_ckpt:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_to(optimizer, device)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            loss = checkpoint['loss']
            total_steps = checkpoint['total_steps']
            clipper = QuantileClip(model.parameters())
            clipper.load_state_dict(checkpoint['clipper'])
            start_epoch = checkpoint['epoch'] + 1

            assert start_epoch < args.num_epochs, \
                f"The loaded model has already been trained for at least {args.num_epochs}."

            if re.search(r'^\d{6}\.pth$', os.path.split(args.ckpt)[1]):
                args.name = os.path.basename(os.path.split(args.ckpt)[0])
            else:
                args.name = os.path.splitext(os.path.split(args.ckpt)[1])[0]

            create_new_name = False

    if not os.path.isdir('%s/checkpoints/%s' % (save_path, args.name)) and \
            not os.path.isdir('%s/runs/%s' % (save_path, args.name)):
        os.makedirs('%s/checkpoints/%s' % (save_path, args.name))
    elif create_new_name:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        args.name = args.name + current_time
        os.makedirs('%s/checkpoints/%s' % (save_path, args.name))

    logger = Logger(args.name, total_steps, args.save_path, args.log_freq)

    model.to(device)
    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        print("--> Starting epoch ", str(epoch))
        for i_batch, data_blob in enumerate(train_loader, start=start):
            image1_, image2_, depth1, depth2, pose_gt, intrinsics, flowxyz_gt, valid_mask, depth_scale_factor = \
                [x.to(device) for x in data_blob]

            image1 = normalize_image(image1_.float())
            image2 = normalize_image(image2_.float())

            flow2d_est, flow2d_rev, pose, valid = model(**dict(
                image1=image1, image2=image2, depth1=depth1, depth2=depth2, intrinsics=intrinsics, iters=12,
                train_mode=True, depth_scale_factor=depth_scale_factor))

            valid_mask *= valid.unsqueeze(-1)

            loss, metrics = loss_fn(flow2d_est, flow2d_rev, pose, flowxyz_gt, depth1, depth2, intrinsics, pose_gt,
                                    valid_mask, args, 'train')
            total_steps = logger.push(metrics)

            optimizer.zero_grad()
            loss.backward()
            clipper.step()
            optimizer.step()
            scheduler.step()

            # Validation
            if (total_steps - 1) % args.log_freq == args.log_freq - 1:
                model.eval()

                with torch.no_grad():
                    data_blob = next(val_iter, None)
                    if data_blob is None:
                        val_iter = iter(val_loader)
                        data_blob = next(val_iter)

                    image1_, image2_, depth1, depth2, pose_gt, intrinsics, flowxyz_gt, valid_mask, depth_scale_factor \
                        = [x.to(device) for x in data_blob]

                    image1 = normalize_image(image1_.float())
                    image2 = normalize_image(image2_.float())

                    flow2d_est, flow2d_rev, pose, valid = model(
                        **dict(image1=image1, image2=image2, depth1=depth1, depth2=depth2,
                               intrinsics=intrinsics, iters=12, train_mode=True,
                               depth_scale_factor=depth_scale_factor))

                    valid_mask *= valid.unsqueeze(-1)

                    metrics = loss_fn(flow2d_est, flow2d_rev, pose, flowxyz_gt, depth1, depth2, intrinsics, pose_gt,
                                      valid_mask, args, 'val')
                    logger.push_val(metrics)

                model.train()

            progressbar(total_steps, args.num_steps, 30, '■')

        if (epoch + 1) % args.save_freq == 0:
            path = '%s/checkpoints/%s/%06d.pth' % (save_path, args.name, epoch)
            clipper = QuantileClip(model.parameters())

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'total_steps': total_steps,
                'clipper': clipper.state_dict(),
            }, path)

    print("   -->Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help='name your experiment')
    parser.add_argument('--ckpt', type=str, help='checkpoint to load')
    parser.add_argument('--continue_training_from_ckpt', action="store_true", help='continue training from loaded '
                        'model. Total steps, optimizer, scheduler, loss, clipper and number of trained epochs are '
                        'restored from checkpoint. Otherwise, training starts from zero, starting from the pretrained '
                        'loaded model.')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--datapath', type=str, required=True, help='full path to folder containing the scenes')
    parser.add_argument('--difficulty_level', type=int, choices=[0, 1, 2, 3],
                        help='drunk dataset diffculty level to use')
    parser.add_argument('--save_freq', type=int, default=1, help='number of epochs between model is saved')
    parser.add_argument('--log_freq', type=int, default=100, help='number of steps between logs are saved')
    parser.add_argument('--res_factor', type=float, default=1., help='reduce resolution by a factor')
    parser.add_argument('--train_scenes', type=int, nargs='+',
                        default=[1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19],
                        help='scenes used for training')
    parser.add_argument('--val_scenes', type=int, nargs='+', default=[0, 4, 5], help='scenes used for validating')
    parser.add_argument("--save_path", type=str, help="if specified, logs and checkpoint will be saved here. "
                        "Otherwise, stored in the project main folder")
    parser.add_argument('--fl_weight', type=float, default=1.)
    parser.add_argument('--rv_weight', type=float, default=0.2)
    parser.add_argument('--dz_weight', type=float, default=100.0)
    parser.add_argument('--relative_tra_rot_weight', type=float, default=1.)
    parser.add_argument('--pose_weight', type=float, default=200.0)
    parser.add_argument('--pose_cnn_weight', type=float, default=0.1)
    parser.add_argument("--depth_augmentor", action="store_true",
                        help="use depth augmentor in dataloader during training")
    parser.add_argument('--pose_bias', type=float, default=0.01,
                        help='bias to be multiplied to the estimated delta_pose of the model in each iteration.')
    parser.add_argument('--pct_start', type=float, default=0.001)
    parser.add_argument('--invert_order_prob', type=float, default=0.5, help='probability to invert the images order '
                        'in the dataloader, invert if the random probability is under this number.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--radius', type=int, default=32)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    print(args)
    train(args)
