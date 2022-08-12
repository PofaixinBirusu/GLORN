import argparse
import os
import os.path as osp

from easydict import EasyDict as edict

from glorn.utils.common import ensure_dir


_config = edict()

# random seed
_config.seed = 7351

# dirs
_config.working_dir = osp.dirname(osp.realpath(__file__).replace("/config_kitti.py", ""))
_config.root_dir = osp.dirname(osp.dirname(_config.working_dir))

_config.exp_name = osp.basename(_config.working_dir)
_config.output_dir = osp.join(_config.working_dir, 'kitti_working_dir', 'output')
_config.snapshot_dir = osp.join(_config.output_dir, 'snapshots')
_config.log_dir = osp.join(_config.output_dir, 'logs')
_config.event_dir = osp.join(_config.output_dir, 'events')
_config.feature_dir = osp.join(_config.output_dir, 'features')

ensure_dir(_config.output_dir)
ensure_dir(_config.snapshot_dir)
ensure_dir(_config.log_dir)
ensure_dir(_config.event_dir)
ensure_dir(_config.feature_dir)

# data
_config.data = edict()
_config.data.dataset_root = "/home/zhang/KITTI"

# train data
_config.train = edict()
_config.train.batch_size = 1
_config.train.num_workers = 8
_config.train.point_limit = 30000
_config.train.use_augmentation = True
_config.train.augmentation_noise = 0.01
_config.train.augmentation_min_scale = 0.8
_config.train.augmentation_max_scale = 1.2
_config.train.augmentation_shift = 2.0
_config.train.augmentation_rotation = 1.0

# test config
_config.test = edict()
_config.test.batch_size = 1
_config.test.num_workers = 8
_config.test.point_limit = None

# eval config
_config.eval = edict()
_config.eval.acceptance_overlap = 0.0
_config.eval.acceptance_radius = 1.0
_config.eval.inlier_ratio_threshold = 0.05
_config.eval.rre_threshold = 5.0
_config.eval.rte_threshold = 2.0

# ransac
_config.ransac = edict()
_config.ransac.distance_threshold = 0.3
_config.ransac.num_points = 4
_config.ransac.num_iterations = 50000

# optim config
_config.optim = edict()
_config.optim.lr = 1e-4
_config.optim.lr_decay = 0.95
_config.optim.lr_decay_steps = 4
_config.optim.weight_decay = 1e-6
_config.optim.max_epoch = 160
_config.optim.grad_acc_steps = 1

# model - backbone
_config.backbone = edict()
_config.backbone.num_stages = 5
_config.backbone.init_voxel_size = 0.3
_config.backbone.kernel_size = 15
_config.backbone.base_radius = 4.25
_config.backbone.base_sigma = 2.0
_config.backbone.init_radius = _config.backbone.base_radius * _config.backbone.init_voxel_size
_config.backbone.init_sigma = _config.backbone.base_sigma * _config.backbone.init_voxel_size
_config.backbone.group_norm = 32
_config.backbone.input_dim = 1
_config.backbone.init_dim = 64
_config.backbone.output_dim = 256

# model - Global
_config.model = edict()
_config.model.ground_truth_matching_radius = 0.6
_config.model.num_points_in_patch = 128
_config.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_config.coarse_matching = edict()
_config.coarse_matching.num_targets = 128
_config.coarse_matching.overlap_threshold = 0.1
_config.coarse_matching.num_correspondences = 256
_config.coarse_matching.dual_normalization = True

# model - SuperPoint Processor (Geometric SelfAttention, Overlapping Factor)
_config.superpoint_processor = edict()
_config.superpoint_processor.input_dim = 2048
_config.superpoint_processor.hidden_dim = 128
_config.superpoint_processor.output_dim = 256
_config.superpoint_processor.num_heads = 4
_config.superpoint_processor.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_config.superpoint_processor.sigma_d = 4.8
_config.superpoint_processor.sigma_a = 15
_config.superpoint_processor.angle_k = 3
_config.superpoint_processor.reduction_a = 'max'

# model - Fine Matching
_config.fine_matching = edict()
_config.fine_matching.topk = 2
_config.fine_matching.acceptance_radius = 0.6
_config.fine_matching.mutual = True
_config.fine_matching.confidence_threshold = 0.05
_config.fine_matching.use_dustbin = False
_config.fine_matching.use_global_score = False
_config.fine_matching.correspondence_threshold = 3
_config.fine_matching.correspondence_limit = None
_config.fine_matching.num_refinement_steps = 5

# loss - Coarse level
_config.coarse_loss = edict()
_config.coarse_loss.positive_margin = 0.1
_config.coarse_loss.negative_margin = 1.4
_config.coarse_loss.positive_optimal = 0.1
_config.coarse_loss.negative_optimal = 1.4
_config.coarse_loss.log_scale = 40
_config.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_config.fine_loss = edict()
_config.fine_loss.positive_radius = 0.6

# loss - Overlapping + Matching + circle
_config.loss = edict()
_config.loss.weight_coarse_loss = 1.0
_config.loss.weight_fine_loss = 1.0
_config.loss.weight_overlapping = 1.0
_config.loss.weight_matching = 1.0


def make_cfg():
    return _config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
