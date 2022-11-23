import os
import time
import pydiffvg
import torch
from roi import ROI
# from data.get_content_image import get_content_image

class CLIPVG:
    def __init__(self, args):
        self._rois = []
        self._svg = args.svg
        self._device = pydiffvg.get_device()
        self._render = pydiffvg.RenderFunction.apply
        self._n_iters = args.n_iters
        self._img_save_freq = args.img_save_freq

        basename = os.path.splitext(os.path.basename(args.svg))[0]
        prompts_concat = self._normalize_name('_'.join(args.prompts))
        self._output_dir = os.path.join(args.output_dir, self._normalize_name(basename),
                                        prompts_concat + '_' + args.exp_name)
        os.makedirs(self._output_dir, exist_ok=True)

        self._parse_svg(args)
        self._rois = self._init_rois(args, init_img=self._init_img)
        self._points_optim = torch.optim.Adam(self._points_vars, lr=args.shape_lr)
        self._color_optim = torch.optim.Adam(self._color_vars, lr=args.color_lr)

    def _parse_svg(self, args):
        canvas_width, canvas_height, shapes, shape_groups = \
            pydiffvg.svg_to_scene(args.svg)

        self._canvas_width = canvas_width
        self._canvas_height = canvas_height
        self._shapes = shapes
        self._shape_groups = shape_groups

        if args.add_bg_layer:
            self._add_bg_layer(args)

        points_vars = []
        for path in self._shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)
        color_vars = {}
        for group in self._shape_groups:
            group.fill_color.requires_grad = True
            color_vars[group.fill_color.data_ptr()] = group.fill_color
        color_vars = list(color_vars.values())
        self._points_vars = points_vars
        self._color_vars = color_vars

        self._init_img = self._render_torch().detach()
        # self._init_img = get_content_image().to(self._device)
        # The output image is in linear RGB space. Do Gamma correction before saving the image.
        # pydiffvg.imwrite(img.cpu(), 'results/refine_svg/init.png', gamma=1.0)

    def _render_torch(self):
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self._canvas_width, self._canvas_height, self._shapes, self._shape_groups)
        img = self._render(self._canvas_width,  # width
                           self._canvas_height,  # height
                           2,  # num_samples_x
                           2,  # num_samples_y
                           0,  # seed
                           None,  # bg
                           *scene_args)
        # Compose img with white background, H*W*4 -> H*W*3
        img = img[:, :, 3:4] * img[:, :, :3] + \
              torch.ones(img.shape[0], img.shape[1], 3, device=self._device) * (1.0 - img[:, :, 3:4])
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img

    def _init_rois(self, args, init_img: torch.Tensor):
        rois = []
        areas = [0, 0, self._canvas_width, self._canvas_height]  # [x, y, w, h] for the whole image
        areas.extend(args.extra_rois)
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
    def _normalize_name(x: str):
        return x.strip().replace(' ', '_').replace(',', '')

    @staticmethod
    def _get_config_from_roi_idx(configs: list, idx: int):
        config = configs[0] if len(configs) == 1 else configs[idx]
        return config

    def _save_raster(self, x: torch.Tensor, name: str):
        raster = x.detach().cpu().squeeze(0).permute(1, 2, 0)  # NCHW -> CHW -> HWC
        pydiffvg.imwrite(raster, os.path.join(self._output_dir, name), gamma=1.0)

    def _save_svg(self, name: str):
        pydiffvg.save_svg(os.path.join(self._output_dir, name),
                          self._canvas_width, self._canvas_height, self._shapes, self._shape_groups)

    def _add_bg_layer(self, args):
        num_segments = 4
        points = []
        points.append((0.01, 0.01))
        points.append((0.01, 0.33))
        points.append((0.01, 0.66))
        points.append((0.01, 0.99))
        points.append((0.33, 0.99))
        points.append((0.66, 0.99))
        points.append((0.99, 0.99))
        points.append((0.99, 0.66))
        points.append((0.99, 0.33))
        points.append((0.99, 0.01))
        points.append((0.66, 0.01))
        points.append((0.33, 0.01))
        points = torch.tensor(points)
        points[:, 0] *= self._canvas_width
        points[:, 1] *= self._canvas_height
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        path = pydiffvg.Path(num_control_points=num_control_points,
                             points=points,
                             stroke_width=torch.tensor(1.0),
                             is_closed=True)

        # Insert the bg element at the beginning as the bottom layer
        self._shapes.insert(0, path)

        # Increase the shape_id of other elements by 1
        for sg in self._shape_groups:
            shape_ids = sg.shape_ids
            sg.shape_ids = shape_ids + 1

        # Add a shape group for the bg element
        assert len(args.bg_layer_rgba) == 4
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                         fill_color=torch.tensor(args.bg_layer_rgba))
        self._shape_groups.insert(0, path_group)

    def run(self):
        # self._save_raster(self._init_img, 'init.png')
        st = time.time()
        for i in range(self._n_iters):
            self._points_optim.zero_grad()
            self._color_optim.zero_grad()
            roi_loss_list = []
            total_loss = torch.tensor([0.0], requires_grad=True).to(self._device)

            # Forward
            img = self._render_torch()
            for roi_calc in self._rois:
                roi_loss = roi_calc(img)
                total_loss += roi_loss
                roi_loss_list.append(roi_loss.item())

            # Backprop
            total_loss.backward()
            self._points_optim.step()
            self._color_optim.step()
            for group in self._shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)

            print(f'Iter: {i}, Elapsed time: {time.time()-st},'
                  f'Total loss: {total_loss.item()}, ROI loss: {roi_loss_list}')

            if (i + 1) % self._img_save_freq == 0 \
                    or i == 0 \
                    or i == (self._n_iters - 1):
                self._save_raster(img, f'iter_{str(i).zfill(4)}.png')
                self._save_svg(f'{str(i).zfill(4)}.svg')

        time_cost = time.time() - st
        print(f'Time cost : {time_cost} seconds, average time cost per iter: {time_cost / self._n_iters}')

