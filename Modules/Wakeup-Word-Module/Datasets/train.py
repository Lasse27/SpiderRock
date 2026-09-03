"""
This file generated the dataset for model training in the subfolder "Train".
"""

import logging
import os
from pathlib import Path

import helpers

# Constants
_SPLIT: str = "Train"
_LOGGER: logging.Logger = helpers.initialize_logger(_SPLIT)


def main():
    # Initialize directory environment
    helpers.initialize_environment(_LOGGER)

    # Generate positive samples
    helpers.generate_voice_samples(
        logger=_LOGGER,
        split=_SPLIT,
        label=helpers.LABEL_POSITIVE,
        text_file=Path(os.path.join(helpers.WORDS_FOLDER, "positives_train.txt")),
        limit=50,
    )

    helpers.generate_voice_samples_with_cloning(
        logger=_LOGGER,
        split=_SPLIT,
        label=helpers.LABEL_POSITIVE,
        text_file=Path(os.path.join(helpers.WORDS_FOLDER, "positives_train.txt")),
        limit=50,
    )

    # Add augmentations to generated samples
    helpers.generate_volumed_samples(
        logger=_LOGGER,
        split=_SPLIT,
        label=helpers.LABEL_POSITIVE,
    )

    helpers.generate_noised_samples(
        logger=_LOGGER,
        split=_SPLIT,
        label=helpers.LABEL_POSITIVE,
        noise_dir=Path(os.path.join(helpers.NOISE_FOLDER, _SPLIT)),
    )

    # Generate negative samples
    helpers.generate_voice_samples(
        logger=_LOGGER,
        split=_SPLIT,
        label=helpers.LABEL_NEGATIVE,
        text_file=Path(os.path.join(helpers.WORDS_FOLDER, "negatives_train.txt")),
        limit=200,
    )

    helpers.generate_voice_samples_with_cloning(
        logger=_LOGGER,
        split=_SPLIT,
        label=helpers.LABEL_NEGATIVE,
        text_file=Path(os.path.join(helpers.WORDS_FOLDER, "negatives_train.txt")),
        limit=200,
    )

    # Add augmentations to generated samples
    helpers.generate_volumed_samples(
        logger=_LOGGER,
        split=_SPLIT,
        label=helpers.LABEL_NEGATIVE,
    )

    helpers.generate_noised_samples(
        logger=_LOGGER,
        split=_SPLIT,
        label=helpers.LABEL_NEGATIVE,
        noise_dir=Path(os.path.join(helpers.NOISE_FOLDER, _SPLIT)),
    )

    # Generate random noise
    helpers.generate_just_noise(
        logger=_LOGGER,
        split=_SPLIT,
        label=helpers.LABEL_NEGATIVE,
        noise_dir=Path(os.path.join(helpers.NOISE_FOLDER, _SPLIT)),
        samples_per_noise=25,
    )

    # Generate annotation file
    helpers.generate_computed(logger=_LOGGER, split=_SPLIT)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as ex:
        _LOGGER.warning(ex)
