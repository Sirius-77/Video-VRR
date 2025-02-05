import torch
import torch.nn as nn

from util import parser
from pathlib import Path

# DINOv2
DINOV2_MODEL = [
    'dinov2_vits14',
    'dinov2_vitb14',
    'dinov2_vitl14',
    'dinov2_vitg14'
]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get Training Parameters
myargs = parser.parse_arguments()


class DINOV2(nn.Module):
    def __init__(self, model_name='dinov2_vitl14'):
        super().__init__()
        repo_path = str(Path(__file__).resolve().parent.parent / 'dinov2')
        self.model = torch.hub.load(
            repo_or_dir=repo_path,
            model=model_name,
            source="local"
        )
        self.num_trainable_blocks = 4

    def forward(self, x, static_mask):
        """
            The forward method for the DINOv2 class

            Parameters:
                x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

            Returns:
                f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
                t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """
        x = self.model.prepare_tokens_with_masks(x)

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for i, blk in enumerate(self.model.blocks[-self.num_trainable_blocks:]):
            if i == self.num_trainable_blocks - 1:
                y = blk.norm1(x)
                B, N, C = y.shape
                qkv = blk.attn.qkv(y).reshape(B, N, 3, blk.attn.num_heads, C // blk.attn.num_heads).permute(2, 0, 3, 1,
                                                                                                            4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                att = (q @ k.transpose(-2, -1)) * blk.attn.scale
                att = att.softmax(dim=-1)
                num_heads = blk.attn.num_heads
                last_map = (att[:, :, :1, 1:].detach()).sum(dim=1).sum(dim=1) / num_heads
            x = blk(x)

        x = self.model.norm(x)

        if myargs.aggr_cate:
            return x

        if static_mask is not None:
            output = x.detach().clone()
            output = output[:, 1:]
            """Idea1: SELECT VALID SEMANTIC + HIGH ATTENTION WEIGHT (SPECIFY NUM) LOCAL FEATURES"""
            """Selecting important local features by semantic and attention scores"""
            N = 150
            # Set positions where static_mask is 0 in last_map to negative infinity
            masked_scores = last_map.masked_fill(static_mask == 0, float('-inf'))

            # Use topk to get the indices of the top N scoring features for each image
            top_scores, top_indices = torch.topk(masked_scores, N, dim=1)
            # Create an expanded index for selecting features from output
            expanded_indices = top_indices.unsqueeze(-1).expand(-1, -1, output.size(2))

            # Select features
            selected_local_features = torch.gather(output, 1, expanded_indices)

            # Calculate the number of features actually selected for each image
            valid_feature_counts = (masked_scores > 0).sum(dim=1)
            selected_feature_counts = (top_scores > float('-inf')).sum(dim=1)

            # Obtain the indices of the actually selected features in the original output
            # actual_selected_indices = top_indices * (top_scores > float('-inf'))
            actual_selected_indices = top_indices.masked_fill(top_scores == float('-inf'), -1)
            expanded_indices_pro = actual_selected_indices.unsqueeze(-1).expand(-1, -1, output.size(2))
            valid_mask = expanded_indices_pro == -1
            selected_local_features[valid_mask] = 0
            """Idea2: SELECT VALID SEMANTIC + HIGH ATTENTION WEIGHT (SPECIFY THRESHOLD) LOCAL FEATURES"""
            # # Set threshold
            # threshold = 1e-4
            # # Set positions where static_mask is 0 in last_map to negative infinity
            # masked_scores = last_map.masked_fill(static_mask == 0, float('-inf'))
            # # Use threshold to get the indices of the threshold-above scoring features for each image
            # valid_feature_counts = (masked_scores >= threshold).sum(dim=1)
            # invalid_mask = masked_scores < threshold
            # expanded_mask = invalid_mask.unsqueeze(-1).expand(-1, -1, output.size(2))
            # selected_local_features = output * expanded_mask

            t = x[:, 0]
            cls_token_expanded = t.unsqueeze(dim=1)
            fusion_features = torch.cat((cls_token_expanded, selected_local_features), dim=1)
            return fusion_features
        else:
            return x



