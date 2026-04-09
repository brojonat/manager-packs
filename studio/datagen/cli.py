"""datagen — synthetic data generation CLI for ManagerPack content development.

Each subcommand generates a different problem-type dataset and writes
parquet + sidecar JSON containing the ground-truth parameters.
"""

import click

from datagen.problems.binary_classification import binary_classification
from datagen.problems.blobs import blobs
from datagen.problems.coin_flip import coin_flip
from datagen.problems.friedman import friedman
from datagen.problems.messy_binary import messy_binary
from datagen.problems.multiclass_classification import multiclass_classification
from datagen.problems.multilabel_classification import multilabel_classification
from datagen.problems.regression import regression


@click.group()
def main():
    """Generate synthetic datasets with known ground truth."""


main.add_command(coin_flip)
main.add_command(binary_classification)
main.add_command(multiclass_classification)
main.add_command(multilabel_classification)
main.add_command(regression)
main.add_command(friedman)
main.add_command(blobs)
main.add_command(messy_binary)
