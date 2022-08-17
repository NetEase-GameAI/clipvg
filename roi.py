import torch
from torchvision import transforms
from typing import List
from optimizer.clip_loss_dir import CLIPLossDir
from data.text_template import compose_text_with_templates


class ROI():
    def __init__(self, init_img: torch.Tensor, prompt: str, area: List[int], n_patches: int = 64, crop_ratio: float = 0.8,
                 w_roi: float = 30, w_patch: float = 80):
        self._n_patches = n_patches
        # self.crop_ratio = crop_ratio
        self._w_roi = w_roi
        self._w_patch = w_patch
        x, y, h, w = area
        self._x = x
        self._y = y
        self._h = h
        self._w = w
        patch_size = int(max(h, w) * crop_ratio)
        self._cropper_patch = transforms.RandomCrop(patch_size, pad_if_needed=True, fill=1.0, padding_mode='constant')
        # self._cropper_roi = transforms.RandomCrop(max(h, w), pad_if_needed=True, fill=1.0, padding_mode='constant')
        self._cropper_roi = transforms.CenterCrop(max(h, w))  # TODO: use white padding instead.
        self._process = transforms.Compose([
            transforms.RandomPerspective(fill=1.0, p=1.0, distortion_scale=0.3),
            transforms.RandomHorizontalFlip(p=0.3),
        ])
        init_img_roi = self._get_roi(init_img)
        self._clip_loss_func = CLIPLossDir(default_ref1=self._cropper_roi(init_img_roi),
                                           default_ref2=compose_text_with_templates('photo'),
                                           default_input2=compose_text_with_templates(prompt))

    def _get_roi(self, x: torch.Tensor):
        x_roi = x[:, :, self._y:self._y + self._h, self._x:self._x + self._w]
        return x_roi

    def __call__(self, x: torch.Tensor):
        x_roi_raw = self._get_roi(x)
        total_loss = 0.0
        if self._w_roi > 0.0:
            x_roi = self._cropper_roi(x_roi_raw)
            # x_roi = self._process(x_roi)
            roi_loss = self._clip_loss_func(x1=x_roi)
            total_loss += self._w_roi * roi_loss
        if self._n_patches > 0 and self._w_patch > 0.0:
            patches = []
            for _ in range(self._n_patches):
                patch = self._cropper_patch(x_roi_raw)
                patch = self._process(patch)
                patches.append(patch)

            patches = torch.cat(patches, dim=0)
            patch_loss = self._w_patch * self._clip_loss_func(x1=patches)
            total_loss += self._w_patch * patch_loss
        return total_loss


