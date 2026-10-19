"""Command-line entry point.

    python -m cxr gan       --config configs/default.yaml
    python -m cxr train     --config configs/default.yaml [--synthetic]
    python -m cxr evaluate  --config configs/default.yaml [--synthetic]
    python -m cxr qsvm      --config configs/default.yaml [--synthetic]
"""

import argparse

from cxr import pipeline
from cxr.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m cxr", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["gan", "train", "evaluate", "qsvm"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="use the classifier trained with GAN-augmented data")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.stage == "gan":
        pipeline.train_gan(cfg)
    elif args.stage == "train":
        pipeline.train_classifier(cfg, args.synthetic)
    elif args.stage == "evaluate":
        pipeline.evaluate(cfg, args.synthetic)
    else:
        pipeline.qsvm(cfg, args.synthetic)


if __name__ == "__main__":
    main()
