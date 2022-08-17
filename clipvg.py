import pydiffvg
import argparse
import ttools.modules
import torch
import skimage.io
from roi import ROI


class CLIPVG():
    def __init__(self, args):
        self._rois = []
        self._svg = args.svg

        canvas_width, canvas_height, shapes, shape_groups = \
            pydiffvg.svg_to_scene(args.svg)

        self._canvas_width = canvas_width
        self._canvas_height = canvas_height
        self._device = pydiffvg.get_device()

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)

        self._render = pydiffvg.RenderFunction.apply
        init_img = self._render(scene_args)
        # The output image is in linear RGB space. Do Gamma correction before saving the image.
        # pydiffvg.imwrite(img.cpu(), 'results/refine_svg/init.png', gamma=1.0)

        self._rois = self._init_rois(args, init_img=init_img)

    def _render(self, scene_args):
        img = self._render(self._canvas_width,  # width
                           self._canvas_height,  # height
                           2,  # num_samples_x
                           2,  # num_samples_y
                           0,  # seed
                           None,  # bg
                           *scene_args)
        return img

    def _init_rois(self, args, init_img: torch.Tensor):
        rois = []
        areas = [0, 0, self._canvas_height, self._canvas_width]  # [x, y, h, w] for the whole image
        areas = areas.extend(args.extra_rois)
        for i, prompt in enumerate(args.prompts):
            n_patches = self._get_config_from_roi_idx(args.n_patches, i)
            crop_ratio = self._get_config_from_roi_idx(args.crop_ratio, i)
            w_roi = self._get_config_from_roi_idx(args.w_roi, i)
            w_patch = self._get_config_from_roi_idx(args.w_patch, i)
            roi = ROI(init_img=init_img, prompt=prompt, area=areas[i*4:i*4+4],
                      n_patches=n_patches, crop_ratio=crop_ratio,
                      w_roi=w_roi, w_patch=w_patch)
            rois.append(roi)
        return rois

    @staticmethod
    def _get_config_from_roi_idx(configs: list, idx: int):
        config = configs[0] if len(configs) == 1 else configs[idx]
        return config

    def run(self):
