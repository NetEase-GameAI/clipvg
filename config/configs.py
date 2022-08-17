import argparse
import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')


def _str2bool(x: str):
    if x.lower() == 'true':
        flag = True
    elif x.lower() == 'false':
        flag = False
    else:
        raise RuntimeError(f'Invalid input {x} for str2bool, expected True or False.')
    return flag


def _none_or_str(x: str):
    if x == 'None':
        return None
    else:
        return x


def parse_args():
    """Configurations."""
    desc = "CLIPVG."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--svg", default=os.path.join(ROOT_DIR, 'images', 'pitt.svg'),
                        help="Source SVG path.")
    parser.add_argument('--prompts', type=str, nargs='+', required=True,
                        help='Text prompts for each ROI. The first one is always associated with the whole image.'
                             'The rest are associated with the extra ROIs.')
    parser.add_argument('--extra_rois', type=int, nargs='+', default=None,
                        help='Extra ROIs besides the whole image. [x1, y1, h1, w1, x2, w2, h2, w2, ...]')
    parser.add_argument("--n_iters", type=int, default=150, help='Number of iterations.')
    parser.add_argument("--shape_lr", type=float, default=0.2, help='Learning rate for the shape parameters.')
    parser.add_argument("--color_lr", type=float, default=0.01, help='Learning rate for the color parameters.')
    parser.add_argument("--crop_ratio", type=float, default=0.8, help='Ratio of the patch size to the ROI.')
    parser.add_argument('--n_crops', type=int, default=64,
                        help='number of patches per ROI.')
    parser.add_argument("--w_roi", type=float, nargs='+', default=30, help='Weight for each ROI.')
    parser.add_argument("--w_patch", type=float, nargs='+', default=80, help='Weight for the patches for each ROI.')

    return parser.parse_args()
