from utilities.consts import default_model_name, cache_folder

from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline

from optimum.pipelines import pipeline
from optimum.bettertransformer import BetterTransformer
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import (
    OptimizationConfig,
    AutoQuantizationConfig,
)

from os import rename
from os.path import join
from shutil import rmtree


def optimize_and_quantize_onnx(model_path: str):

    # ----------------------------------------------------------------------------------------------
    # -- OPTIMIZE

    print(" OPTIMIZE ".center(150, "#"))

    model_name = model_path
    save_dir = "tmp/my_model_onnx_optimized"

    model = ORTModelForCausalLM.from_pretrained(model_name, export=True)

    optimizer = ORTOptimizer.from_pretrained(model)
    optimization_config = OptimizationConfig(
        optimization_level=2,
        enable_transformers_specific_optimizations=True,
    )

    optimizer.optimize(save_dir=save_dir, optimization_config=optimization_config)

    # ----------------------------------------------------------------------------------------------
    # -- QUANTIZE

    print(" QUANTIZE ".center(150, "#"))

    model_name = save_dir
    save_dir = f"{model_path}_onnx"

    quantizer_1 = ORTQuantizer.from_pretrained(
        model_name, file_name="decoder_model_optimized.onnx"
    )
    quantizer_2 = ORTQuantizer.from_pretrained(
        model_name, file_name="decoder_with_past_model_optimized.onnx"
    )

    quantizer = [quantizer_1, quantizer_2]

    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    for q in quantizer:
        q.quantize(save_dir=save_dir, quantization_config=dqconfig)

    # ----------------------------------------------------------------------------------------------
    # -- RENAME

    rename(
        join(f"{save_dir}", "decoder_model_optimized_quantized.onnx"),
        join(f"{save_dir}", "decoder_model.onnx"),
    )

    rename(
        join(f"{save_dir}", "decoder_with_past_model_optimized_quantized.onnx"),
        join(f"{save_dir}", "decoder_with_past_model.onnx"),
    )

    rmtree("tmp")


def get_pipeline_onnx(model_name: str = default_model_name) -> Pipeline:

    model = ORTModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, accelerator="ort"
    )

    return pipe


def get_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=f"{cache_folder}/tokenizers"
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model(model_name: str, use_sdp: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        cache_dir=f"{cache_folder}/models",
    )

    if use_sdp:
        model = BetterTransformer.transform(model)

    return model


def get_pipeline(
    model_name: str = default_model_name, use_sdp: bool = False
) -> Pipeline:
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, use_sdp=use_sdp)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        accelerator="bettertransformer",
    )

    return pipe
