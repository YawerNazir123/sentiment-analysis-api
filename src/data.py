"""Dataset for the sentiment-analysis demo.

The built-in dataset is a labelled collection of short product-review
sentences. It is generated from curated phrase templates with a fixed
random seed, so ``generate_dataset()`` is fully reproducible and the
project works offline with no downloads.

This is a *demo* dataset: the sentences are synthetic, so the trained
model is a working proof of the full pipeline (data -> training ->
evaluation -> serving), not a production-grade sentiment model. To train
on real data, pass your own CSV with ``text`` and ``label`` columns to
``src.train --data path/to/your.csv`` -- for example the IMDB or Amazon
Polarity datasets.
"""

from __future__ import annotations

import random

import pandas as pd

LABELS = ("positive", "negative", "neutral")

ITEMS = (
    "phone",
    "laptop",
    "headphones",
    "camera",
    "smartwatch",
    "tablet",
    "speaker",
    "keyboard",
    "monitor",
    "earbuds",
    "charger",
    "mouse",
)

# Each class has openers, aspects and closers that are combined to build
# varied sentences instead of repeating a handful of fixed strings.
_POSITIVE_OPENERS = (
    "I love this {item}",
    "This {item} is fantastic",
    "Really impressed with this {item}",
    "Absolutely love my new {item}",
    "This {item} is brilliant",
    "I am so happy with this {item}",
    "Great {item}",
    "This {item} exceeded my expectations",
    "Best {item} I have ever owned",
    "I would highly recommend this {item}",
    "Delighted with this {item}",
    "This {item} is a joy to use",
)
_POSITIVE_ASPECTS = (
    "the battery lasts for days",
    "the build quality feels premium",
    "the screen is bright and sharp",
    "the sound is crystal clear",
    "it is fast and responsive",
    "setup took less than five minutes",
    "it works flawlessly out of the box",
    "the design is sleek and modern",
    "it is comfortable to use all day",
    "the performance is outstanding",
    "it charges incredibly fast",
    "the picture quality is stunning",
)
_POSITIVE_CLOSERS = (
    "worth every penny.",
    "five stars from me!",
    "I will definitely buy from this brand again.",
    "highly recommended.",
    "a great purchase overall.",
    "could not be happier.",
    "exactly what I was looking for.",
    "I use it every single day.",
)

_NEGATIVE_OPENERS = (
    "I hate this {item}",
    "This {item} is terrible",
    "Really disappointed with this {item}",
    "Worst {item} I have ever bought",
    "This {item} is a complete waste of money",
    "I regret buying this {item}",
    "Awful {item}",
    "This {item} is broken",
    "Do not buy this {item}",
    "Frustrated with this {item}",
    "This {item} is useless",
    "Very unhappy with this {item}",
)
_NEGATIVE_ASPECTS = (
    "the battery dies within an hour",
    "the build quality feels cheap and flimsy",
    "the screen flickers constantly",
    "the sound is muffled and distorted",
    "it freezes all the time",
    "setup was a nightmare",
    "it stopped working after two days",
    "the design is ugly and awkward",
    "it overheats within minutes",
    "the performance is painfully slow",
    "it arrived scratched and damaged",
    "customer support never replied",
)
_NEGATIVE_CLOSERS = (
    "total waste of money.",
    "one star, and that is generous.",
    "I am returning it immediately.",
    "do not waste your money.",
    "a terrible purchase overall.",
    "avoid at all costs.",
    "I want a refund.",
    "extremely disappointed.",
)

_NEUTRAL_OPENERS = (
    "This {item} is okay",
    "The {item} arrived on time",
    "I bought this {item} last week",
    "This {item} does what it says",
    "The {item} comes in a plain box",
    "I received this {item} yesterday",
    "This {item} is average",
    "The {item} is neither great nor bad",
    "I have been using this {item} for a month",
    "This {item} is fairly standard",
    "The {item} looks like the photos",
    "I ordered this {item} in black",
)
_NEUTRAL_ASPECTS = (
    "the battery lasts about a day",
    "the build quality is acceptable",
    "the screen is fine indoors",
    "the sound is decent for the price",
    "it works as expected",
    "setup took a while",
    "it does the job",
    "the design is simple",
    "it is comfortable enough",
    "the performance is adequate",
    "it charges at a normal speed",
    "the picture quality is fine",
)
_NEUTRAL_CLOSERS = (
    "nothing special, but it works.",
    "it is fine for everyday use.",
    "about what I expected.",
    "not amazing, not terrible.",
    "does the job, nothing more.",
    "fine for the price.",
    "I have no strong feelings about it.",
    "it is what it is.",
)

_TEMPLATES = {
    "positive": (_POSITIVE_OPENERS, _POSITIVE_ASPECTS, _POSITIVE_CLOSERS),
    "negative": (_NEGATIVE_OPENERS, _NEGATIVE_ASPECTS, _NEGATIVE_CLOSERS),
    "neutral": (_NEUTRAL_OPENERS, _NEUTRAL_ASPECTS, _NEUTRAL_CLOSERS),
}

# A few fully hand-written sentences per class, so the dataset is not
# only template combinations.
_HANDWRITTEN = {
    "positive": (
        "The delivery was fast and the product works beautifully, I am thrilled.",
        "After a week of heavy use I can honestly say this is excellent value.",
        "Sound quality blew me away, easily the best I have tried at this price.",
        "Sturdy, reliable and good looking. I have already recommended it to friends.",
        "It just works. No fuss, no problems, exactly as advertised and then some.",
        "Customer service was wonderful and the product itself is even better.",
    ),
    "negative": (
        "It broke the first time I used it and the seller refused to help.",
        "Cheap materials, loose parts and a terrible smell out of the box.",
        "Nothing about this product works as advertised, complete disappointment.",
        "I waited three weeks for delivery and it arrived faulty.",
        "The instructions are wrong, the parts do not fit and it keeps crashing.",
        "Save your money. This is the worst purchase I have made this year.",
    ),
    "neutral": (
        "It arrived in a large box with plenty of packaging and a short manual.",
        "I use it a few times a week, mostly in the evenings.",
        "The product is black, rectangular and about the size of a paperback.",
        "It was on sale when I bought it, so the price seemed reasonable.",
        "There are three buttons on the side and a small light on the front.",
        "I bought it as a gift and the recipient seems happy enough with it.",
    ),
}


def generate_dataset(n_per_class: int = 350, seed: int = 42) -> pd.DataFrame:
    """Generate the labelled review dataset.

    Parameters
    ----------
    n_per_class:
        Number of template-generated sentences per class.
    seed:
        Random seed, so the dataset is identical on every run.
    """
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for label in LABELS:
        openers, aspects, closers = _TEMPLATES[label]
        seen: set[str] = set()
        while len(seen) < n_per_class:
            item = rng.choice(ITEMS)
            opener = rng.choice(openers).format(item=item)
            aspect = rng.choice(aspects)
            closer = rng.choice(closers)
            sentence = f"{opener}, {aspect} -- {closer}"
            seen.add(sentence)
        rows.extend({"text": text, "label": label} for text in sorted(seen))
        rows.extend(
            {"text": text, "label": label} for text in _HANDWRITTEN[label]
        )

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def load_dataset(path: str = "data/reviews.csv") -> pd.DataFrame:
    """Load a labelled dataset CSV with ``text`` and ``label`` columns."""
    df = pd.read_csv(path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("dataset CSV must have 'text' and 'label' columns")
    return df.dropna(subset=["text", "label"]).reset_index(drop=True)
