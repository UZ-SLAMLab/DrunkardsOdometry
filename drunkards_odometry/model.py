import torch
import torch.nn as nn
import numpy as np

from lietorch import SE3

from .blocks.extractor import BasicEncoder
from .blocks.resnet import FPN
from .blocks.corr import CorrBlock
from .blocks.gru import ConvGRU
from .blocks.grid import GridFactor
from .sampler_ops import depth_sampler

from . import projective_ops as pops
from . import se3_field
from .blocks import ResnetEncoder, PoseDecoder


GRAD_CLIP = .01


def L1_Charbonnier_loss(x, y, alpha=0.5, eps=1e-6):
    diff = torch.add(x, -y)
    error = torch.pow(diff * diff + eps, alpha)
    return error


class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs() > GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x


class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


class GridSmoother(nn.Module):
    def __init__(self):
        super(GridSmoother, self).__init__()

    def forward(self, ae, wxwy):
        factor = GridFactor()
        ae = ae.permute(0, 2, 3, 1)

        wx = wxwy[:, 0].unsqueeze(-1)
        wy = wxwy[:, 1].unsqueeze(-1)
        wu = torch.ones_like(wx)
        J = torch.ones_like(wu).unsqueeze(-2)

        # residual terms
        ru = ae.unsqueeze(-2)
        rx = torch.zeros_like(ru)
        ry = torch.zeros_like(ru)

        factor.add_factor([J], wu, ru, ftype='u')
        factor.add_factor([J, -J], wx, rx, ftype='h')
        factor.add_factor([J, -J], wy, ry, ftype='v')
        factor._build_factors()

        ae = factor.solveAAt().squeeze(dim=-2)
        ae = ae.permute(0, 3, 1, 2).contiguous()

        return ae


class BasicUpdateBlock(nn.Module):
    """ Estimate updates to the camera pose and to the 3D deformable flow"""

    def __init__(self, args, hidden_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.gru = ConvGRU(hidden_dim, dilation=3)
        self.solver = GridSmoother()
        self.pose_bias = args.pose_bias

        self.corr_enc = nn.Sequential(
            nn.Conv2d(196, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 3 * 128, 1, padding=0))

        self.flow_enc = nn.Sequential(
            nn.Conv2d(15, 128, 7, padding=3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 3 * 128, 1, padding=0))

        self.ae_enc = nn.Conv2d(16, 3 * 128, 3, padding=1)

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 16, 1, padding=0),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip())

        self.delta_pose = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 6, 1, padding=0),
            GradientClip(),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip(),
            nn.Sigmoid())

        self.ae_wts = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 2, 1, padding=0),
            GradientClip(),
            nn.Softplus())

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 64 * 9, 1, padding=0),
            GradientClip())


    def forward(self, net, inp, corr, flow, dz, twist, twist_pose, ae):
        motion_info = torch.cat([flow, 10 * dz, 10 * twist, 10 * twist_pose], dim=-1)
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0, 3, 1, 2)

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        ae = self.ae_enc(ae)
        net = self.gru(net, inp, cor, mot, ae)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        delta_pose = self.delta_pose(net) * self.pose_bias
        weight = self.weight(net)

        edges = 5 * self.ae_wts(net)
        ae = self.solver(ae, edges)

        return net, mask, ae, delta, delta_pose, weight


class DrunkardsOdometry(nn.Module):
    """ Like pose_v3 but pose being initialize by PoseCNN that is trained together.
    """
    def __init__(self, args):
        super(DrunkardsOdometry, self).__init__()

        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 3

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = FPN(output_dim=hdim + 3 * hdim)
        self.update_block = BasicUpdateBlock(args, hidden_dim=hdim)

        # pose network
        self.pose_encoder = ResnetEncoder(18, pretrained=True, num_input_images=2)
        self.pose_decoder = PoseDecoder(self.pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

    def initializer(self, depth):
        """ Initialize rigid motion embeddings (ae) and coords"""
        batch_size, ht, wd = depth.shape
        device = depth.device

        y0, x0 = torch.meshgrid(torch.arange(ht), torch.arange(wd))
        coords0 = torch.stack([x0, y0], dim=-1).float()
        coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device)

        ae = torch.zeros(batch_size, 16, ht, wd, device=device)

        return ae, coords0

    def single_to_dense(self, pose, T):
        """
        Repeat an equal transformation for every pixel in the image.
        Input: pose in quaternions: tx, ty, tz, qx, qy, qz, qw
        """
        if pose.dim() == 1:  # Single tensor case
            pose = pose.unsqueeze(0)

        batch_size, ht, wd = T.shape
        numel = np.prod((ht, wd))
        data = torch.cat([pose[i].repeat(numel, 1) for i in range(batch_size)], dim=0)
        T_pose = SE3(data).view((batch_size, ht, wd))

        return T_pose

    def features_and_correlation(self, image1, image2):
        # Extract features and build correlation volume
        fmap1, fmap2 = self.fnet([image1, image2])

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # Extract context features using Resnet50
        net_inp = self.cnet(image1)
        net, inp = net_inp.split([128, 128 * 3], dim=1)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        return corr_fn, net, inp

    def invert_pose(self, pose_2_1):
        """ Invert pose """
        if pose_2_1.dim() == 1:  # Single tensor case
            batch_size = 1
        else:
            batch_size, _ = pose_2_1.shape

        pose_2_1 = pops.pose_from_quat_to_matrix(pose_2_1)
        pose_1_2 = torch.stack([torch.inverse(pose_2_1[i]) for i in range(batch_size)], dim=0)
        pose_1_2 = pops.pose_from_matrix_to_quat(pose_1_2)

        return pose_1_2

    def initializer_scene_flow(self, depth, pose_2_1):
        """ Initialize coords and transformation maps """
        batch_size, ht, wd = depth.shape
        device = depth.device
        numel = np.prod((ht, wd))

        # Invert pose
        pose_2_1 = pops.pose_from_quat_to_matrix(pose_2_1).to(device)
        pose_1_2 = torch.stack([torch.inverse(pose_2_1[i]) for i in range(batch_size)], dim=0)
        pose_1_2 = pops.pose_from_matrix_to_quat(pose_1_2)

        data = torch.cat([pose_1_2[i].repeat(numel, 1) for i in range(batch_size)], dim=0)
        T = SE3(data).view((batch_size, ht, wd))
        T_pose = SE3(data).view((batch_size, ht, wd))

        return T, T_pose

    def predict_pose(self, image1, image2):
        """ Predict pose between two RGB frames """
        pose_inputs = [image1, image2]
        pose_inputs = [self.pose_encoder(torch.cat(pose_inputs, 1))]
        rotation, translation = self.pose_decoder(pose_inputs)

        # Model estimates in logarithmic space, we map it to quaternions through exponential map
        rotation = rotation[:, 0].clone().squeeze()
        translation = translation[:, 0].clone().squeeze()
        pose = torch.cat((translation, rotation), -1)
        pose = SE3.exp(pose).vec()

        if pose.dim() == 1:
            pose = pose.unsqueeze(0)

        return pose

    def forward(self, image1, image2, depth1, depth2, intrinsics, iters=12, train_mode=False, depth_scale_factor=1.0):
        """ Estimate optical flow between pair of frames """
        # Intrinsics and depth at 1/8 resolution
        intrinsics_r8 = intrinsics / 8.0
        depth1_r8 = depth1[:, 3::8, 3::8]
        depth2_r8 = depth2[:, 3::8, 3::8]

        # Estimate an initial guess of the relative camera pose to transform from cam1 to cam2
        pose_cnn = self.predict_pose(image1, image2)
        pose_cnn_tra, pose_cnn_rot = pose_cnn.split([3, 4], dim=-1)
        pose_cnn = (torch.cat((pose_cnn_tra * depth_scale_factor.unsqueeze(-1), pose_cnn_rot), -1)).float()

        # Invert the pose (pose_2_1 -> pose_1_2) and use it to initialize the scene flow due to the camera movement (T_pose)
        T, T_pose = self.initializer_scene_flow(depth1_r8, pose_cnn)
        ae, coords0 = self.initializer(depth1_r8)
        pose = SE3(pose_cnn.unsqueeze(1).unsqueeze(1))

        corr_fn, net, inp = self.features_and_correlation(image1, image2)

        flow_est_list = []
        flow_rev_list = []
        pose_list = []

        for itr in range(iters):
            T = T.detach()
            T_pose = T_pose.detach()

            # Estimated pixel correspondence and inverse depth2 by T and ground truth depth1 for each pixel of camera 1
            coords1_xyz, _ = pops.projective_transform(T, depth1_r8, intrinsics_r8)
            coords1, zinv_proj = coords1_xyz.split([2, 1], dim=-1)

            # Ground truth inverse depth2 projected to camera 1 using the estimated pixel correspondences
            zinv, _ = depth_sampler(1.0 / depth2_r8, coords1)

            corr = corr_fn(coords1.permute(0, 3, 1, 2).contiguous())
            flow2d = coords1 - coords0

            dz = zinv.unsqueeze(-1) - zinv_proj
            twist = T.log()
            twist_pose = T_pose.log()

            net, mask, ae, delta, delta_pose, weight = self.update_block(net, inp, corr, flow2d, dz, twist, twist_pose, ae)

            # Apply rectifications
            target = coords1_xyz.permute(0, 3, 1, 2) + delta
            target = target.contiguous()

            # Model estimates the pose updates in logarithmic space, we map it to quaternions through exponential map
            pose = SE3.exp(delta_pose.permute(0, 2, 3, 1).contiguous()) * pose
            T_pose = self.single_to_dense(pose.vec().squeeze(), T)

            # Gauss-Newton step
            T = se3_field.step_inplace(T, ae, target, weight, depth1_r8, intrinsics_r8)

            if train_mode:
                flow2d_rev = target.permute(0, 2, 3, 1)[..., :2] - coords0
                flow2d_rev = se3_field.cvx_upsample(8 * flow2d_rev, mask)

                T_up = se3_field.upsample_se3(T, mask)  # T upsampled to original resolution
                flow2d_est, flow3d_est, _ = pops.induced_flow(T_up, depth1, intrinsics)

                flow_est_list.append(flow2d_est)
                flow_rev_list.append(flow2d_rev)
                pose_list.append(self.invert_pose(pose.vec().squeeze()))

        if train_mode:
            _, _, valid = pops.induced_flow(T_up, depth1, intrinsics, min_depth=0.01, max_depth=30.0)
            valid = valid > 0.5
            pose_list.append(pose_cnn)  # Append the estimated camera pose by the initializer cnn to the last position
            return flow_est_list, flow_rev_list, pose_list, valid

        T_up = se3_field.upsample_se3(T, mask)
        return T_up, self.invert_pose(pose.vec().squeeze())
