import torch
from torch import nn
from torch.nn import functional as F

import model.semantic_masker.DeepLabV3Plus.network as network
from model.aggregator.mlp import MyMLP
from model.backbone.dinov2 import DINOV2
from model.aggregator.gem import GeM
from model.aggregator.netvlad import NetVLAD
from model.normalization import L2Norm
from util import parser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get Training Parameters
myargs = parser.parse_arguments()


def get_semantic_extractor(args):
    if args.semantic_segnet == 'deeplabv3plus_resnet101':
        model = network.modeling.__dict__[args.semantic_segnet](num_classes=19, output_stride=16)
        model.to(device)
        checkpoint = torch.load(args.semantic_segnet_ckpt)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        del checkpoint
        return model
    elif args.semantic_segnet is None:
        return None


semantic_extractor = get_semantic_extractor(myargs)


class SemVGNet(nn.Module):
    """Semantic Fusion Visual Geo-localization Network"""

    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.semantic_net = args.semantic_segnet
        # self.semantic_extractor = get_semantic_extractor(args)
        self.aggrtype = args.aggregation
        self.fusion_type = args.fusion_type
        self.cls_token_only = args.cls_token_only
        self.aggr_cate = args.aggr_cate
        self.aggregation = get_aggregation(args)
        self.resize = args.resize
        self.L2Norm = L2Norm(dim=1)
        if myargs.backbone == "dinov2b":
            self.inputdim = 768
        elif myargs.backbone == "dinov2l":
            self.inputdim = 1024
        if self.aggr_cate:
            self.mlp = MyMLP(input_dim=self.inputdim, hidden_dim=512, output_dim=128)

    def forward(self, x):
        """Get Semantic Prior Information about the Dynamic Object in The Scene."""
        # Get semantic segmentation result.
        with torch.no_grad():
            patch_size = (14, 14)
            if self.semantic_net == "deeplabv3plus_resnet101":
                y = semantic_extractor(x).max(1)[1]
                # Statistical analysis of semantic information within the patch.
                unfolded_tensor = y.unfold(1, patch_size[0], patch_size[0]).unfold(2, patch_size[1], patch_size[1])
                patch_mode, _ = unfolded_tensor.contiguous().view(y.size(0), -1,
                                                                  patch_size[0] * patch_size[1]).mode(dim=2)

                # Filtering the Dynamic Objects & SKY & Vegetation.
                static_mask = patch_mode < 8
                static_mask = static_mask.to(torch.int)
            elif self.semantic_net is None:
                static_mask = None

        """Extract Local Features according Attention Scores and Semantic Prior Information."""
        x = self.backbone(x, static_mask)
        # x = self.L2Norm(x)
        g_fea = x[:, 0, :]
        """Idea3: SELECT VALID SEMANTIC + AGGREGATE ACCORDING TO CATEGORY"""
        if self.aggr_cate:
            aggr_l_fea = []
            l_fea = x[:, 1:, :]
            sem_assin = patch_mode.masked_fill(static_mask == 0, -1)
            l_fea = self.mlp(l_fea)
            B, N, C = l_fea.shape
            # 遍历每个类别
            num_classes = 8
            for cls in range(num_classes):
                class_mask = sem_assin == cls
                expanded_mask = class_mask.unsqueeze(-1).expand(-1, -1, C)
                class_features = l_fea * expanded_mask
                gem_output = self.aggregation(class_features)
                gem_output = gem_output.squeeze(2)
                # aggr_l_fea.append(gem_output)
                normalized_output = self.L2Norm(gem_output)
                aggr_l_fea.append(normalized_output)
            aggr_l_fea = torch.stack(aggr_l_fea)
            aggr_l_fea = aggr_l_fea.view(aggr_l_fea.shape[1], aggr_l_fea.shape[0]*aggr_l_fea.shape[2])
            final = torch.cat((g_fea, aggr_l_fea), dim=1)
            final = self.L2Norm(final)
            return final


        if self.cls_token_only:
            return g_fea
        else:
            # Aggregate key local features
            if self.aggrtype == 'gem':
                x = self.aggregation(x[:, 1:, :])
                x = x.squeeze(2)
            # Fuse global feature and aggregated key local feature
            if self.fusion_type == "orthogonal_fusion":
                x = calc_orthogonal_vector(x, g_fea)
            elif self.fusion_type == "concat_fusion":
                x = torch.cat((x, g_fea), dim=1)
            x = self.L2Norm(x)
            return x


def get_backbone(args):
    if args.backbone == 'dinov2s':
        backbone = DINOV2(model_name='dinov2_vits14')
    elif args.backbone == 'dinov2b':
        backbone = DINOV2(model_name='dinov2_vitb14')
    elif args.backbone == 'dinov2l':
        backbone = DINOV2(model_name='dinov2_vitl14')
    return backbone


def get_aggregation(args):
    if args.aggregation == 'gem':
        return GeM()
    elif args.aggregation == 'netvlad':
        return NetVLAD()





def calc_orthogonal_vector(a, b):
    # Calculate the dot product of a and b for each row
    dot_product = (a * b).sum(1, keepdim=True)

    # Calculate the magnitude squared of b for each row
    magnitude_squared_b = (b ** 2).sum(1, keepdim=True)

    # Calculate the projection of a on b for each row
    projection_a_on_b = (dot_product / magnitude_squared_b) * b

    # Calculate the orthogonal component of a relative to b
    orthogonal_component = a - projection_a_on_b

    # Add the orthogonal component to tensor b
    result = orthogonal_component + b
    return result
