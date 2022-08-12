import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from glorn.modules.loss import WeightedCircleLoss
from glorn.modules.ops.transformation import apply_transform
from glorn.modules.registration.metrics import isotropic_transform_error
from glorn.modules.ops.pairwise_distance import pairwise_distance

from sklearn.metrics import precision_recall_fscore_support


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']

        src_node_corr_knn_points = apply_transform(src_node_corr_knn_points, transform)
        dists = pairwise_distance(ref_node_corr_knn_points, src_node_corr_knn_points)  # (B, N, M)
        gt_masks = torch.logical_and(ref_node_corr_knn_masks.unsqueeze(2), src_node_corr_knn_masks.unsqueeze(1))
        gt_corr_map = torch.lt(dists, self.positive_radius ** 2)
        gt_corr_map = torch.logical_and(gt_corr_map, gt_masks)
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), ref_node_corr_knn_masks)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), src_node_corr_knn_masks)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }


class MetricLoss(nn.Module):
    """
    We evaluate both contrastive loss and circle loss
    """

    def __init__(self, configs, log_scale=16, pos_optimal=0.1, neg_optimal=1.4):
        super(MetricLoss, self).__init__()
        self.log_scale = log_scale
        self.pos_optimal = pos_optimal
        self.neg_optimal = neg_optimal

        self.pos_margin = configs.pos_margin
        self.neg_margin = configs.neg_margin
        self.max_points = configs.max_points

        self.safe_radius = configs.safe_radius
        self.matchability_radius = configs.matchability_radius
        self.pos_radius = configs.pos_radius  # just to take care of the numeric precision

    def get_circle_loss(self, coords_dist, feats_dist):
        """
        Modified from: https://github.com/XuyangBai/D3Feat.pytorch
        """
        pos_mask = coords_dist < self.pos_radius
        neg_mask = coords_dist > self.safe_radius

        ## get anchors that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

        # get alpha for both positive and negative pairs
        pos_weight = feats_dist - 1e5 * (~pos_mask).float()  # mask the non-positive
        pos_weight = (pos_weight - self.pos_optimal)  # mask the uninformative positive
        pos_weight = torch.max(torch.zeros_like(pos_weight), pos_weight).detach()

        neg_weight = feats_dist + 1e5 * (~neg_mask).float()  # mask the non-negative
        neg_weight = (self.neg_optimal - neg_weight)  # mask the uninformative negative
        neg_weight = torch.max(torch.zeros_like(neg_weight), neg_weight).detach()

        lse_pos_row = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-1)
        lse_pos_col = torch.logsumexp(self.log_scale * (feats_dist - self.pos_margin) * pos_weight, dim=-2)

        lse_neg_row = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-1)
        lse_neg_col = torch.logsumexp(self.log_scale * (self.neg_margin - feats_dist) * neg_weight, dim=-2)

        loss_row = F.softplus(lse_pos_row + lse_neg_row) / self.log_scale
        loss_col = F.softplus(lse_pos_col + lse_neg_col) / self.log_scale

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2

        return circle_loss

    def get_recall(self, coords_dist, feats_dist):
        """
        Get feature match recall, divided by number of true inliers
        """
        pos_mask = coords_dist < self.pos_radius
        n_gt_pos = (pos_mask.sum(-1) > 0).float().sum() + 1e-12
        _, sel_idx = torch.min(feats_dist, -1)
        sel_dist = torch.gather(coords_dist, dim=-1, index=sel_idx[:, None])[pos_mask.sum(-1) > 0]
        n_pred_pos = (sel_dist < self.pos_radius).float().sum()
        recall = n_pred_pos / n_gt_pos
        return recall

    def get_weighted_bce_loss(self, prediction, gt):
        loss = nn.BCELoss(reduction='none')

        class_loss = loss(prediction, gt)

        weights = torch.ones_like(gt)
        w_negative = gt.sum() / gt.size(0)
        w_positive = 1 - w_negative

        weights[gt >= 0.5] = w_positive
        weights[gt < 0.5] = w_negative
        w_class_loss = torch.mean(weights * class_loss)

        #######################################
        # get classification precision and recall
        predicted_labels = prediction.detach().cpu().round().numpy()
        cls_precision, cls_recall, _, _ = precision_recall_fscore_support(gt.cpu().numpy(), predicted_labels, average='binary')

        return w_class_loss, cls_precision, cls_recall

    def forward(self, src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, rot, trans, scores_overlap,
                scores_saliency):
        """
        Circle loss for metric learning, here we feed the positive pairs only
        Input:
            src_pcd:        [N, 3]
            tgt_pcd:        [M, 3]
            rot:            [3, 3]
            trans:          [3, 1]
            src_feats:      [N, C]
            tgt_feats:      [M, C]
        """
        src_pcd = (torch.matmul(rot, src_pcd.transpose(0, 1)) + trans).transpose(0, 1)
        stats = dict()

        src_idx = list(set(correspondence[:, 0].int().tolist()))
        tgt_idx = list(set(correspondence[:, 1].int().tolist()))

        #######################
        # get BCE loss for overlap, here the ground truth label is obtained from correspondence information
        src_gt = torch.zeros(src_pcd.size(0))
        src_gt[src_idx] = 1.
        tgt_gt = torch.zeros(tgt_pcd.size(0))
        tgt_gt[tgt_idx] = 1.
        gt_labels = torch.cat((src_gt, tgt_gt)).to(torch.device('cuda'))

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_overlap, gt_labels)
        stats['overlap_loss'] = class_loss
        stats['overlap_recall'] = cls_recall
        stats['overlap_precision'] = cls_precision

        #######################
        # get BCE loss for saliency part, here we only supervise points in the overlap region
        src_feats_sel, src_pcd_sel = src_feats[src_idx], src_pcd[src_idx]
        tgt_feats_sel, tgt_pcd_sel = tgt_feats[tgt_idx], tgt_pcd[tgt_idx]
        scores = torch.matmul(src_feats_sel, tgt_feats_sel.transpose(0, 1))
        _, idx = scores.max(1)
        distance_1 = torch.norm(src_pcd_sel - tgt_pcd_sel[idx], p=2, dim=1)
        _, idx = scores.max(0)
        distance_2 = torch.norm(tgt_pcd_sel - src_pcd_sel[idx], p=2, dim=1)

        gt_labels = torch.cat(
            ((distance_1 < self.matchability_radius).float(), (distance_2 < self.matchability_radius).float()))

        src_saliency_scores = scores_saliency[:src_pcd.size(0)][src_idx]
        tgt_saliency_scores = scores_saliency[src_pcd.size(0):][tgt_idx]
        scores_saliency = torch.cat((src_saliency_scores, tgt_saliency_scores))

        class_loss, cls_precision, cls_recall = self.get_weighted_bce_loss(scores_saliency, gt_labels)
        stats['saliency_loss'] = class_loss
        stats['saliency_recall'] = cls_recall
        stats['saliency_precision'] = cls_precision

        #######################################
        # filter some of correspondence as we are using different radius for "overlap" and "correspondence"
        c_dist = torch.norm(src_pcd[correspondence[:, 0].long()] - tgt_pcd[correspondence[:, 1].long()], dim=1)
        c_select = c_dist < self.pos_radius - 0.001
        correspondence = correspondence[c_select]
        if correspondence.size(0) > self.max_points:
            choice = np.random.permutation(correspondence.size(0))[:self.max_points]
            correspondence = correspondence[choice]
        src_idx = correspondence[:, 0].long()
        tgt_idx = correspondence[:, 1].long()
        src_pcd, tgt_pcd = src_pcd[src_idx], tgt_pcd[tgt_idx]
        src_feats, tgt_feats = src_feats[src_idx], tgt_feats[tgt_idx]

        #######################
        # get L2 distance between source / target 3dmatch
        coords_dist = torch.sqrt(pairwise_distance(src_pcd[None, :, :], tgt_pcd[None, :, :]).squeeze(0))
        feats_dist = torch.sqrt(pairwise_distance(src_feats[None, :, :], tgt_feats[None, :, :], normalised=True)).squeeze(0)

        ##############################
        # get FMR and circle loss
        ##############################
        recall = self.get_recall(coords_dist, feats_dist)
        circle_loss = self.get_circle_loss(coords_dist, feats_dist)

        stats['circle_loss'] = circle_loss
        stats['recall'] = recall

        return stats


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points = output_dict['ref_corr_points']
        src_corr_points = output_dict['src_corr_points']
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform = data_dict['transform']
        est_transform = output_dict['estimated_transform']
        src_points = output_dict['src_points']

        rre, rte = isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, rmse, recall

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RMSE': rmse,
            'RR': recall,
        }
