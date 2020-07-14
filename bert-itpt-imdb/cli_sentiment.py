"""
CLI for sentiment analysis prediction on a single sentence by finetuned bert-itpt-imdb model.
"""

import click

from get_prediction import get_prediction


@click.command(
    #may have to remove this help statement
    help="Classifies a passed sequence sentence as positive sentiment or negative sentiment."
)
@click.option(
    "--sequence-string",
    "-s",
    required=True,
    help="Sequence to classify.",
)
@click.option(
    "--model_path",
    "-m",
    required=True,
    help="""Path to finetuned pytorch model file from Google Drive link.""",
)
@click.option(
    "--cuda/--cpu",
    required=True,
    help="Device to load model on.",
)
def main(
        sequence_string, model_path, cuda
):
    #pylint: disable=missing-docstring
    result = get_prediction(str(sequence_string), str(model_path), cuda)
    if result == 0:
        print("Negative sentiment")
    elif result == 1:
        print("Positive sentiment")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
