from scipy.optimize import linear_sum_assignment
from torch import nn
import gorilla
import torch
import torch.nn.functional as F


@gorilla.LOSSES.register_module()
class Criterion(nn.Module):
    def __init__(self,
                 ignore_label=-100,
                 loss_weight=[1.0, 1.0, 1.0, 1.0],
                 cost_weight=[1.0, 1.0, 1.0],
                 non_object_weight=0.1,
                 num_class=18,
                 ):
        super(Criterion, self).__init__()
        class_weight = torch.ones(num_class + 1)
        class_weight[-1] = non_object_weight
        self.register_buffer('class_weight', class_weight)
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)
        self.matcher = HungarianMatcher(cost_weight)
        self.num_class = num_class
        self.ignore_label = ignore_label

    def forward(self, pred, insts):
        """
        :param pred:
            pred_masks: List[Tensor (n,M)]
            pred_labels: (B, n, 19)
            pred_scores: (B, n, 1)
        :param insts:

        :return:
        """
        loss_out = {}
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # match
        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(
            pred_labels.shape[:2],
            self.num_class,
            dtype=torch.int64,
            device=pred_labels.device
        )  # (B, num_query)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)
        loss_out['cls_loss'] = class_loss.item()

        # score loss
        score_loss = torch.tensor([0.0], device=pred_labels.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)

        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue
            pred_score = score[idx_q]
            pred_mask = mask[idx_q]
            tgt_mask = inst.gt_spmasks[idx_gt]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)

            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())

        score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)  # 这里是对batch_size取平均吗， 后面debug到这里要注意，为什么没有对dice_loss取平均

        loss_out['score_loss'] = score_loss.item()
        loss_out['mask_bce_loss'] = mask_bce_loss.item()
        loss_out['mask_dice_loss'] = mask_dice_loss.item()

        loss = (
                self.loss_weight[0] * class_loss + self.loss_weight[1] * mask_bce_loss +
                self.loss_weight[2] * mask_dice_loss + self.loss_weight[3] * score_loss
        )

        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, insts)
                loss += loss_i
                loss_out.update(loss_out_i)

        loss_out['loss'] = loss.item()

        return loss, loss_out

    def get_layer_loss(self, layer, aux_outputs, insts):
        loss_out = {}
        pred_labels = aux_outputs['labels']
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']
        indices = self.matcher(pred_labels, pred_masks, insts)
        idx = self._get_src_permutation_idx(indices)

        # class loss
        tgt_class_o = torch.cat([inst.gt_labels[idx_gt] for inst, (_, idx_gt) in zip(insts, indices)])
        tgt_class = torch.full(pred_labels.shape[:2], self.num_class, dtype=torch.int64, device=pred_labels.device)
        tgt_class[idx] = tgt_class_o
        class_loss = F.cross_entropy(pred_labels.transpose(1, 2), tgt_class, self.class_weight)

        loss_out['cls_loss'] = class_loss

        # score loss
        score_loss = torch.tensor([0.0], device=pred_labels.device)

        # mask loss
        mask_bce_loss = torch.tensor([0.0], device=pred_labels.device)
        mask_dice_loss = torch.tensor([0.0], device=pred_labels.device)
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores, insts, indices):
            if len(inst) == 0:
                continue
            pred_score = score[idx_q]
            pred_mask = mask[idx_q]
            tgt_mask = inst.gt_spmasks[idx_gt]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss += F.mse_loss(pred_score, tgt_score)
            mask_bce_loss += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            mask_dice_loss += dice_loss(pred_mask, tgt_mask.float())
        score_loss = score_loss / len(pred_masks)
        mask_bce_loss = mask_bce_loss / len(pred_masks)
        mask_dice_loss = mask_dice_loss / len(pred_masks)

        loss_out['score_loss'] = score_loss.item()
        loss_out['mask_bce_loss'] = mask_bce_loss.item()
        loss_out['mask_dice_loss'] = mask_dice_loss.item()

        loss = (
                self.loss_weight[0] * class_loss + self.loss_weight[1] * mask_bce_loss
                + self.loss_weight[2] * mask_dice_loss + self.loss_weight[3] * score_loss
        )
        loss_out = {f'layer_{layer}_'+k: v for k,v in loss_out.items()}
        return loss, loss_out


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


@torch.jit.script
def dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    :param input: A float tensor of arbitrary shape.
    :param targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for negative class and 1 for the positive class).
    :return:
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def get_iou(inputs: torch.Tensor, targets: torch.Tensor):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_weight):
        """
        Creates the matcher
        :param cost_weight:
            cost_class:
            cost_mask:
            cost_dice:
        """
        super(HungarianMatcher, self).__init__()
        self.register_buffer('cost_weight', torch.tensor(cost_weight))

    @torch.no_grad()
    def forward(self, pred_labels, pred_masks, insts):
        """
        :param pred_masks: List[Tensor] len(p2c) == B, Tensor.shape==(n,N)
        :param pred_labels: (B, n_q, 19)
        :param insts: List[Instances3D]
        :return:
        """
        indices = []
        for pred_label, pred_mask, inst in zip(pred_labels, pred_masks, insts):
            if len(inst) == 0:
                indices.append(([], []))
                continue
            pred_label = pred_label.softmax(-1)  # (n_q,19)
            tgt_idx = inst.gt_labels  # (num_inst,)
            cost_class = -pred_label[:, tgt_idx]  # (n_q, num_inst)
            tgt_mask = inst.gt_spmasks  # (num_inst, N)
            cost_mask = batch_sigmoid_bce_loss(pred_mask, tgt_mask.float())
            cost_dice = batch_dice_loss(pred_mask, tgt_mask.float())

            C = (self.cost_weight[0] * cost_class + self.cost_weight[1] * cost_mask + self.cost_weight[2] * cost_dice)
            C = C.cpu()
            indices.append(linear_sum_assignment(C))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@torch.jit.script
def batch_sigmoid_bce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """

    :param inputs: (num_querys, N)
    :param targets: (num_inst, N)
    :return: Loss tensor
    """
    N = inputs.shape[1]
    pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum('nc,mc->nm', neg, 1 - targets)
    return loss / N


@torch.jit.script
def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    :param inputs:
    :param targets:
    :return:
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss
