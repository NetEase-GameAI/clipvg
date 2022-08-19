## Environment
- CUDA 11.3
- Python 3.6
- ``pip install -r requirements.txt``
- Install diffvg:
```bash
git clone git@github.com:BachiLi/diffvg.git --recursive
cd diffvg
python setup.py build
```

## CLIPVG Optimization
- Default Setting
```bash
python main.py \
--svg images/pitt.svg \
--prompts "Joker, Heath Ledger"
```

- ROI Prompts
```bash
# The first prompt is always defined for the whole image, followed the extra ROI prompts.
# The extra ROIs are defined by x1, y1, w1, h1, x2, y2, w2, h2, ...
python main.py \
--svg images/horseriding.svg \
--prompts "Astronaut riding a metal horse" "Astronaut" "A metal horse" \
--extra_rois 185 84 185 316 20 175 480 325
```

- Shape-only manipulation
```bash
python main.py \
--svg images/athlete.svg \
--prompts "Stephen Curry" \
--shape_lr 0.4 \
--color_lr 0.0
```

- Color-only manipulation
```bash
python main.py \
--svg images/tajmahal.svg \
--prompts "Taj Mahal under the moon light" \
--shape_lr 0.0 \
--color_lr 0.02
```

## Results
- Find the results in ``./output/<image_name>/<prompt>_<exp_name>``
- The default exp_name is "exp"

## Configs
- Check the configs in ./config/configs.py

## Acknowledgement
- Our code is based on [diffvg](https://github.com/BachiLi/diffvg) and [CLIP](https://github.com/openai/CLIP)
