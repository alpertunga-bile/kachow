from transformers import TrainingArguments, Trainer, AutoTokenizer
from optimum.bettertransformer import BetterTransformer

from utilities.consts import (
    default_onnx_train_args,
    default_train_args,
    default_model_name,
    default_model_path,
)
from utilities.dataset import get_train_eval_datasets
from utilities.model import get_tokenizer, get_model

from optimum.onnxruntime import ORTTrainer, ORTTrainingArguments, ORTModelForCausalLM


def train_onnx(
    train_file: str,
    model_name: str = default_model_name,
    save_path: str = default_model_path,
    train_args: ORTTrainingArguments = default_onnx_train_args,
) -> None:
    model = ORTModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset, eval_dataset = get_train_eval_datasets(train_file, tokenizer)

    trainer = ORTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        feature="text-generation",
    )

    trainer.train()

    trainer.save_model(save_path)


def train(
    train_file: str,
    model_name: str = default_model_name,
    save_path: str = default_model_path,
    train_args: TrainingArguments = default_train_args,
    use_sdp=False,
) -> None:
    model = get_model(model_name, use_sdp)
    tokenizer = get_tokenizer(model_name)
    train_dataset, eval_dataset = get_train_eval_datasets(train_file, tokenizer)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    if use_sdp:
        trainer.model = BetterTransformer.reverse(trainer.model)

    trainer.save_model(save_path)
