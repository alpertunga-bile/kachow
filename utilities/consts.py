from os import environ
from transformers import TrainingArguments
from optimum.onnxruntime import ORTTrainingArguments

environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
environ["TOKENIZERS_PARALLELISM"] = "false"

cache_folder = "cache"
default_model_path = "my_model"
default_model_name = "distilgpt2"

from torch.cuda import is_available

device = "cuda" if is_available() else "cpu"

default_adafactor_train_args = TrainingArguments(
    output_dir=f"{cache_folder}/checkpoints",
    evaluation_strategy="no",
    save_strategy="no",
    save_total_limit=1,
    report_to="none",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    fp16=True,
    overwrite_output_dir=True,
    optim="adafactor",
    learning_rate=1e-3,
    weight_decay=0.0,
    adam_beta1=None,
    adam_beta2=None,
    adam_epsilon=(1e-30, 1e-3),
)

default_train_args = TrainingArguments(
    output_dir=f"{cache_folder}/checkpoints",
    evaluation_strategy="no",
    save_strategy="no",
    save_total_limit=1,
    report_to="none",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    fp16=True,
    overwrite_output_dir=True,
)

default_onnx_train_args = ORTTrainingArguments(
    output_dir=f"{cache_folder}/checkpoints",
    evaluation_strategy="no",
    save_strategy="no",
    save_total_limit=1,
    report_to="none",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    fp16=True,
    overwrite_output_dir=True,
    optim="adamw_ort_fused",
)
