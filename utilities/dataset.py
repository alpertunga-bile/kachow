from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


def tokenize(
    tokenizer: AutoTokenizer, dataset: DatasetDict, max_input_length
) -> Dataset:
    def tokenize_function(example):
        texts = example["text"]
        texts = [case + "\n" for case in texts[:]]
        return tokenizer(texts)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        load_from_cache_file=True,
    )

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        output_length = (total_length // max_input_length) * max_input_length

        if output_length == 0:
            output_length = total_length

        result = {
            k: [
                t[i : i + max_input_length]
                for i in range(0, output_length, max_input_length)
            ]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()

        return result

    tokenized_dataset = tokenized_dataset.map(
        group_texts, batched=True, load_from_cache_file=True
    )

    return tokenized_dataset


def get_train_eval_datasets(train_file: str, tokenizer, seed=42, test_size=0.3):
    dataset = (
        load_dataset(
            "text",
            data_files={"train": train_file},
            split="train",
        )
        .shuffle(seed=seed)
        .train_test_split(test_size=test_size)
    )

    max_input_length = tokenizer.model_max_length
    train_dataset = tokenize(tokenizer, dataset["train"], max_input_length).with_format(
        "torch"
    )
    eval_dataset = tokenize(tokenizer, dataset["test"], max_input_length).with_format(
        "torch"
    )

    return train_dataset, eval_dataset
