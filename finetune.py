from transformers import HfArgumentParser
from huggingface_hub import HfApi
import io
from src.absm import (
    AbsEmbedderModelArguments as EncoderOnlyEmbedderModelArguments,
    AbsEmbedderDataArguments as EncoderOnlyEmbedderDataArguments,
    AbsEmbedderTrainingArguments as EncoderOnlyEmbedderTrainingArguments,
)

from src.modeling import BiEncoderOnlyEmbedderModel
from src.trainer import EncoderOnlyEmbedderTrainer
from src.runner import EncoderOnlyEmbedderRunner



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

    api = HfApi()
    args = '\n'.join(l for l in open("base_same_dataset.sh").readlines() if not l.strip().startswith('hf_hub_token'))
    api.upload_file(
        path_or_fileobj=io.BytesIO(args.encode('utf-8')),
        path_in_repo="args.txt",
        repo_id = training_args.hub_model_id,
        repo_type="model",
        token = training_args.hub_token
        )
    runner = EncoderOnlyEmbedderRunner(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args
    )
    runner.run()


if __name__ == "__main__":
    main()
