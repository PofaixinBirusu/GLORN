import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from glorn.utils.common import ensure_dir


_config = edict()

# common
_config.seed = 7351

# dirs
_config.working_dir = osp.dirname(osp.realpath(__file__).replace("/config_3dmatch.py", ""))
_config.root_dir = osp.dirname(osp.dirname(_config.working_dir))

_config.exp_name = osp.basename(_config.working_dir)
_config.output_dir = osp.join(_config.working_dir, '3dmatch_working_dir', 'output')
_config.snapshot_dir = osp.join(_config.output_dir, 'snapshots')
_config.log_dir = osp.join(_config.output_dir, 'logs')
_config.event_dir = osp.join(_config.output_dir, 'events')
_config.feature_dir = osp.join(_config.output_dir, 'features')
_config.registration_dir = osp.join(_config.output_dir, 'registration')

ensure_dir(_config.output_dir)
ensure_dir(_config.snapshot_dir)
ensure_dir(_config.log_dir)
ensure_dir(_config.event_dir)
ensure_dir(_config.feature_dir)
ensure_dir(_config.registration_dir)

# data
_config.data = edict()
_config.data.dataset_root = "/home/zhang/3DMatch"

# train data
_config.train = edict()
_config.train.batch_size = 1
_config.train.num_workers = 8
_config.train.point_limit = 30000
_config.train.use_augmentation = True
_config.train.augmentation_noise = 0.005
_config.train.augmentation_rotation = 1.0

# test data
_config.test = edict()
_config.test.batch_size = 1
_config.test.num_workers = 8
# _C.test.point_limit = None
_config.test.point_limit = 30000

# evaluation
_config.eval = edict()
_config.eval.acceptance_overlap = 0.0
_config.eval.acceptance_radius = 0.1
_config.eval.inlier_ratio_threshold = 0.05
_config.eval.rmse_threshold = 0.2
_config.eval.rre_threshold = 15.0
_config.eval.rte_threshold = 0.3

# ransac
_config.ransac = edict()
_config.ransac.distance_threshold = 0.05
_config.ransac.num_points = 3
_config.ransac.num_iterations = 1000

# optim
_config.optim = edict()
_config.optim.lr = 1e-4
_config.optim.lr_decay = 0.95
_config.optim.lr_decay_steps = 1
_config.optim.weight_decay = 1e-6
_config.optim.max_epoch = 40
_config.optim.grad_acc_steps = 1

# model - KPConv
_config.backbone = edict()
_config.backbone.num_stages = 4
_config.backbone.init_voxel_size = 0.025
_config.backbone.kernel_size = 15
_config.backbone.base_radius = 2.5
_config.backbone.base_sigma = 2.0
_config.backbone.init_radius = _config.backbone.base_radius * _config.backbone.init_voxel_size
_config.backbone.init_sigma = _config.backbone.base_sigma * _config.backbone.init_voxel_size
_config.backbone.group_norm = 32
_config.backbone.input_dim = 1
_config.backbone.init_dim = 64
_config.backbone.output_dim = 256

# model - Global
_config.model = edict()
_config.model.ground_truth_matching_radius = 0.05
_config.model.num_points_in_patch = 64
_config.model.num_sinkhorn_iterations = 100

# model - Coarse Matching
_config.coarse_matching = edict()
_config.coarse_matching.num_targets = 128
_config.coarse_matching.overlap_threshold = 0.1
_config.coarse_matching.num_correspondences = 256
_config.coarse_matching.dual_normalization = True

# model - SuperPoint Processor (Geometric SelfAttention, Overlapping Factor)
_config.superpoint_processor = edict()
_config.superpoint_processor.input_dim = 1024
_config.superpoint_processor.hidden_dim = 256
_config.superpoint_processor.output_dim = 256
_config.superpoint_processor.num_heads = 4
_config.superpoint_processor.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_config.superpoint_processor.sigma_d = 0.2
_config.superpoint_processor.sigma_a = 15
_config.superpoint_processor.angle_k = 3
_config.superpoint_processor.reduction_a = 'max'

# model - Fine Matching
_config.fine_matching = edict()
_config.fine_matching.topk = 3
_config.fine_matching.acceptance_radius = 0.1
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
_config.coarse_loss.log_scale = 24
_config.coarse_loss.positive_overlap = 0.1

# loss - Fine level
_config.fine_loss = edict()
_config.fine_loss.positive_radius = 0.05

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
