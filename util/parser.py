import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="SemVG for Visual Geo-localization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    """Preprocess Parameters"""
    parser.add_argument("--resize", type=int, default=[322, 322], nargs=2,
                        help="Resizing shape for images (H×W). [224, 224] for training and [322, 322] for validation.")
    """Training Parameters"""
    parser.add_argument("--epochs_num", type=int, default=30,
                        help="Number of epochs to train for.")
    parser.add_argument("--save_dir", type=str, default="../ckpt",
                        help="Save path for the checkpoint file.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training.")
    """Model Parameters"""
    parser.add_argument("--backbone", type=str, default="dinov2l",
                        choices=["dinov2b", "dinov2l"],
                        help="Backbone.")
    parser.add_argument("--semantic_segnet", type=str, default="deeplabv3plus_resnet101",
                        choices=["deeplabv3plus_resnet101", None],
                        help="Semantic Segment Network.")
    parser.add_argument("--semantic_segnet_ckpt", type=str, default="../ckpt/best"
                                                                    "_deeplabv3plus_resnet101_cityscapes_os16.pth",
                        help="Path to semantic segmentation network checkpoint.")
    parser.add_argument("--aggregation", type=str, default="gem",
                        choices=["gem", "netvlad"],
                        help="Aggregation method.")
    parser.add_argument("--fusion_type", type=str, default="orthogonal_fusion",
                        choices=["orthogonal_fusion", "concat_fusion"],
                        help="Method for global feature and aggregated local feature fusion.")
    parser.add_argument("--cls_token_only", type=bool, default=False,
                        help="If only use cls token.")
    parser.add_argument("--aggr_cate", type=bool, default=True,
                        help="If use cate aggr patch token")
    """Inference Parameters"""
    parser.add_argument("--infer_ckpt", type=str, default="../ckpt/semvgl_aggr_cate_best_model.pth",
                        help="Path to load checkpoint from, for inference.")
    args = parser.parse_args()
    return args
