from torchvision import transforms


def get_input_norm(input_norm_method: str = 'plusminus1'):
    if input_norm_method == 'plusminus1':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif input_norm_method == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif input_norm_method == 'clip':
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        raise ValueError(f'Invalid input_norm_method {input_norm_method}!')
    return transforms.Normalize(mean, std)
