## Environment
- CUDA 11.3
- Python 3.6
- ``pip install -r requirements.txt``

## CLIPVG Optimization
- Default Setting
```bash
python main.py \
--svg images/pitt.svg \
--prompts "Joker, Heath Ledger" \
--exp_name default
```

- ROI Prompts
```bash
python main.py \
--svg images/horseriding.svg \
--prompts "Astronaut riding a metal horse" "Astronaut" "A metal horse" \
--extra_rois 185 84 185 316 20 175 480 325 \
--exp_name roi_prompts
```

- Shape-only manipulation
```bash
python main.py \
--svg images/athlete.svg \
--prompts "Stephen Curry" \
--shape_lr 0.4 \
--color_lr 0.0 \
--exp_name reshape
```

- Color-only manipulation
```bash
python main.py \
--svg images/tajmahal.svg \
--prompts "Taj Mahal under the moon light" \
--shape_lr 0.0 \
--color_lr 0.02 \
--exp_name recolor
```

## Results
- Find the results in ``./output/<image_name>/<prompt>_<exp_name>``

## Configs
- Check the configs in ./config/configs.py
