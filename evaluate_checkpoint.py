import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.data import get_data_loaders
from src.model import MyModel
from src.optimization import get_loss
from src.train import one_epoch_test

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint on the test split"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_val_loss.pt",
        help="Path to the checkpoint file to load",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation data loader",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Validation split fraction used when creating loaders",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout rate used when instantiating the model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_loaders = get_data_loaders(
        batch_size=args.batch_size, valid_size=args.valid_size
    )
    loss = get_loss()

    model = MyModel(num_classes=50, dropout=args.dropout)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    if device == "cuda":
        model = model.cuda()

    one_epoch_test(data_loaders["test"], model, loss)


if __name__ == "__main__":
    raise SystemExit(main())
