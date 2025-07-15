from transformers import HfArgumentParser

from src.absm import (
    AbsEmbedderModelArguments as EncoderOnlyEmbedderModelArguments,
    AbsEmbedderDataArguments as EncoderOnlyEmbedderDataArguments,
    AbsEmbedderTrainingArguments as EncoderOnlyEmbedderTrainingArguments,
)

from src.modeling import BiEncoderOnlyEmbedderModel
from src.trainer import EncoderOnlyEmbedderTrainer
from src.runner import EncoderOnlyEmbedderRunner

# __all__ = [
#     'EncoderOnlyEmbedderModelArguments',
#     'EncoderOnlyEmbedderDataArguments',
#     'EncoderOnlyEmbedderTrainingArguments',
#     'BiEncoderOnlyEmbedderModel',
#     'EncoderOnlyEmbedderTrainer',
#     'EncoderOnlyEmbedderRunner',
# ]



# from . import (
#     EncoderOnlyEmbedderDataArguments,
#     EncoderOnlyEmbedderTrainingArguments,
#     EncoderOnlyEmbedderModelArguments,
#     EncoderOnlyEmbedderRunner,
# )


def main():
    parser = HfArgumentParser((
        EncoderOnlyEmbedderModelArguments,
        EncoderOnlyEmbedderDataArguments,
        EncoderOnlyEmbedderTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: EncoderOnlyEmbedderModelArguments
    data_args: EncoderOnlyEmbedderDataArguments
    training_args: EncoderOnlyEmbedderTrainingArguments

    runner = EncoderOnlyEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
