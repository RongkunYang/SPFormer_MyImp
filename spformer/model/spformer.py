import gorilla
import functools

import spconv.pytorch as spconv
import torch
import torch.nn as nn
import pointgroup_ops
from torch_scatter import scatter_mean, scatter_max

from spformer.model.loss import Criterion

from spformer.model.backbone import ResidualBlock, UBlock
from spformer.model.query_decoder import QueryDecoder
from spformer.utils.mask_encoder import rle_encode
from spformer.utils.utils import cuda_cast
import torch.nn.functional as F


@gorilla.MODELS.register_module()
class SPFormer(nn.Module):
    def __init__(self,
                 input_channel: int = 6,
                 blocks: int = 5,
                 block_reps: int = 2,
                 media: int = 32,
                 normalize_before=True,
                 return_blocks=True,
                 pool='mean',
                 num_class=18,
                 decoder=None,
                 criterion=None,
                 test_cfg=None,
                 norm_eval=False,
                 fix_module=[],
                 ):
        super(SPFormer, self).__init__()

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks
        )
        self.output_layer = spconv.SparseSequential(
            norm_fn(media),
            nn.ReLU(inplace=True)
        )
        self.pool = pool
        self.num_class = num_class

        # decoder
        self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

        # criterion
        self.criterion = Criterion(**criterion, num_class=num_class)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval

        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, insts, superpoints, batch_offsets):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        out = self.decoder(sp_feats, batch_offsets)

        loss, loss_dict = self.criterion(out, insts)
        return loss, loss_dict

    def extract_feat(self, x, superpoints, v2pmap):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2pmap.long()]

        # superpoint pooling
        if self.pool == 'mean':
            x = scatter_mean(x, superpoints, dim=0)
        elif self.pool == 'max':
            x, _ = scatter_max(x, superpoints, dim=0)
        return x

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats, insts, superpoints,
                batch_offsets):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(),spatial_shape, batch_size)
        sp_feats = self.extract_feat(input, superpoints,p2v_map)
        out = self.decoder(sp_feats, batch_offsets)
        ret = self.predict_by_feat(scan_ids, out, superpoints, insts)
        return ret

    def predict_by_feat(self, scan_ids, out, superpoints, insts):
        pred_labels = out['labels']
        pred_masks = out['masks']
        pred_scores = out['scores']
        scores = F.softmax(pred_labels[0],dim=-1)[:,:-1]
        scores *= pred_scores[0]
        labels = torch.arange(self.num_class, device=scores.device).unsqueeze(0).repeat(self.decoder.num_query,1).flatten(0,1)
        scores, topk_idx = scores.flatten(0,1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
        labels += 1

        topk_idx = torch.div(topk_idx,self.num_class,rounding_mode='floor')
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        # mask_pred before sigmoid()
        mask_pred = (mask_pred>0).float()
        mask_scores = (mask_pred_sigmoid*mask_pred).sum(1)/(mask_pred.sum(1)+1e-6)
        scores = scores*mask_scores
        # get mask
        mask_pred = mask_pred[:, superpoints].int()

        # score_thr
        score_mask = scores>self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum>self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_ids[0]
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            pred['pred_mask'] = rle_encode(mask_pred[i])
            pred_instances.append(pred)

        gt_instances = insts[0].gt_instances
        return dict(scan_id=scan_ids[0], pred_instances=pred_instances, gt_instances=gt_instances)







