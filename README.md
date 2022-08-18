## Environment
- CUDA 11.3
- Python 3.6
- ``pip install -r requirements.txt``

## CLIPVG Optimization
- Default Setting
```bash
python main.py \
--svg images/peter10-30.svg
--prompts "Joker, Heath Ledger"
```

- ROI Prompts
```bash
python main.py \
--svg images/qima4.svg \
--prompts "Astronaut riding a metal horse" "Astronaut" "A metal horse" \
--extra_rois 185 84 185 316 20 175 480 325
```

- Shape-only 
```bash

```

## Results
- Find the results in ``./output/<imaage_name>/<prompt>``