import os
import sys
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VTF
from typing import Union, List
# from .antialias import AdaptAntiAlias
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.input_norm import get_input_norm

# ViTB32_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'models', 'ViT-B-32.pt')
torch.autograd.set_detect_anomaly(True)


class AdaptCLIP(nn.Module):
    def __init__(self, model_name: str = "ViT-B/32", tensor_range: Union[None, tuple] = None):
        super().__init__()
        tensor_range = tensor_range or [0.0, 1.0]
        assert (len(tensor_range) == 2 and tensor_range[1] > tensor_range[0]), \
            'tensor_range must be a 2-tuple of [min, max]!'
        range_min = tensor_range[0]
        range_max = tensor_range[1]
        self._factor = 1.0 / (range_max - range_min)
        self._bias = 0.0 - range_min * self._factor
        self._normalizer = get_input_norm(input_norm_method='clip')
        model, _ = clip.load(name=model_name)
        self._model = model
        self._input_size = 224
        # self._aa = AdaptAntiAlias()

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 4:  # image
            # convert range to [0.0, 1.0],
            # .e.g., for input range of [-1.0, 1.0], factor = 0.5, bias = 0.5
            x = x * self._factor + self._bias
            # x = VTF.resize(x, size=[self._input_size, self._input_size], interpolation=transforms.InterpolationMode.BICUBIC)
            x = F.interpolate(x, size=(224, 224), mode='area')
            # x = VTF.center_crop(x, output_size=self._input_size)
            x = self._normalizer(x)
            features = self._model.encode_image(x)
        elif len(x.shape) == 2:  # text
            # x = clip.tokenize(x)
            features = self._model.encode_text(x)
        else:
            raise TypeError(f'Invalid input with type {type(x)}!')
        # Inplace division will cause an error.
        # "one of the variables needed for gradient computation has been modified by an inplace operation"
        # features /= features.norm(dim=-1, keepdim=True)
        features = features / features.norm(dim=-1, keepdim=True)
        return features


class CLIPLossDir(nn.Module):
    def __init__(self, model_name: str = "ViT-B/32", tensor_range: Union[None, tuple] = None,
                 default_input1=None, default_input2=None, default_ref1=None, default_ref2=None, device=None):
        super().__init__()
        self._device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = nn.DataParallel(AdaptCLIP(model_name=model_name, tensor_range=tensor_range).to(device))
        with torch.no_grad():
            self._default_feat1 = None if default_input1 is None else self._avg_feat(self._model(self._preprocess(default_input1)))
            self._default_feat2 = None if default_input2 is None else self._avg_feat(self._model(self._preprocess(default_input2)))
            self._default_ref1_feat = None if default_ref1 is None else self._model(self._preprocess(default_ref1))
            self._default_ref1_feat = self._default_ref1_feat.mean(axis=0, keepdim=True)
            self._default_ref1_feat /= self._default_ref1_feat.norm(dim=-1, keepdim=True)
            self._default_ref2_feat = None if default_ref2 is None else self._model(self._preprocess(default_ref2))
            self._default_ref2_feat = self._default_ref2_feat.mean(axis=0, keepdim=True)
            self._default_ref2_feat /= self._default_ref2_feat.norm(dim=-1, keepdim=True)

    @staticmethod
    def _avg_feat(x: torch.Tensor):
        assert len(x.shape) == 2
        x = x.mean(axis=0, keepdim=True)
        x /= x.norm(dim=-1, keepdim=True)
        return x

    def _preprocess(self, x: Union[None, torch.Tensor, List[str]]):
        # Only the preprocess of text (tokenize) is done here.
        # The preprocess of image can be done with multi-GPU, and is left to the AdaptCLIP module.
        # Note: the preprocess of text can not be done in AdaptCLIP,
        # since the behavior of nn.DataParallel is weird if the input is a list of len==1.
        # The input will be copied to both GPUs in the two-GPU case,
        # and the result will be a 2*512 tensor instead of the desired 1*512.
        if isinstance(x, list):  # text
            assert isinstance(x[0], str)
            x = clip.tokenize(x)
            x = x.to(self._device)
        return x

    def forward(self,
                x1: Union[None, torch.Tensor, List[str]] = None,
                x2: Union[None, torch.Tensor, List[str]] = None):
        feat1 = self._default_feat1 if x1 is None else self._model(self._preprocess(x1))
        feat2 = self._default_feat2 if x2 is None else self._model(self._preprocess(x2))
        delta1 = feat1 - self._default_ref1_feat
        delta1 /= delta1.norm(dim=-1, keepdim=True)
        delta2 = feat2 - self._default_ref1_feat
        delta2 /= delta2.norm(dim=-1, keepdim=True)
        similarities = feat1 @ feat2.T
        # the range of cosine simialrity is [-1, 1].
        # the range of clip_loss is [0, 2]
        clip_loss = 1.0 - torch.mean(similarities)
        return clip_loss
