from utilities.model import (
    get_bettertransformer_pipeline,
    get_pipeline_onnx,
    get_default_pipeline,
)
from dataclasses import dataclass
from utilities.utility import get_accelerator_type, get_variable_dictionary


@dataclass
class GenerateArgs:
    num_return_sequences: int = 1
    min_length: int = 0
    max_length: int = 50
    early_stopping: bool = False
    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 0
    remove_invalid_values: bool = False
    guidance_scale: float = 1.0


def generate_text(
    input: str,
    model: str,
    args: GenerateArgs = GenerateArgs(),
    is_accelerate: bool = True,
) -> str:
    pipe = None

    if is_accelerate is False:
        pipe = get_default_pipeline(model)
    else:
        accelerator_type = get_accelerator_type(model)

        if accelerator_type == "onnx":
            pipe = pipe = get_pipeline_onnx(model_name=model)
        elif accelerator_type == "bettertransformer":
            pipe = get_bettertransformer_pipeline(model_name=model, use_sdp=False)
        else:
            raise ValueError(
                "Cant define accelerator type by folder. Can't find .onnx file for onnx, .bin for bettertransformer"
            )

    args = get_variable_dictionary(args)

    output = pipe(input, **args)

    return output[0]["generated_text"]
