"""Command-line sentiment prediction.

Usage::

    python -m src.predict "I love this product, it works perfectly"
    python -m src.predict --batch "Great product!" "Terrible, broke in a day"
"""

from __future__ import annotations

import argparse
import json

from src.model import load_or_train, predict_batch, predict_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment from the command line.")
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument(
        "--batch", nargs="+", default=None, help="Classify several texts at once"
    )
    args = parser.parse_args()

    if args.batch is None and args.text is None:
        parser.error("provide a text argument or --batch")

    model = load_or_train()
    if args.batch is not None:
        print(json.dumps(predict_batch(model, args.batch), indent=2))
    else:
        print(json.dumps(predict_text(model, args.text), indent=2))


if __name__ == "__main__":
    main()
