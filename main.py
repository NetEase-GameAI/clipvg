from config.configs import parse_args
from clipvg import CLIPVG


def main():
    """General entry point for running GANs."""
    args = parse_args()
    framework = CLIPVG(args)
    framework.run()


if __name__ == '__main__':
    main()
