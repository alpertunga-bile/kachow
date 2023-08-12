from os.path import exists
from platform import system
from os import listdir


def create_requirements_file() -> None:
    requirements_file = "requirements.txt"
    os_name = system()

    if exists(requirements_file):
        return

    packages = set()
    packages.add("transformers\n")
    packages.add("accelerate\n")
    packages.add("xformers\n")

    if os_name == "Linux":
        packages.add("triton\n")

    packages.add("optimum\n")
    packages.add("optimum[onnxruntime]\n")
    packages.add("optimum[onnxruntime-gpu]")
    packages.add("optimum[onnxruntime-training]")
    packages.add(
        "torch torchvision --index-url https://download.pytorch.org/whl/cu118\n"
    )

    with open(requirements_file, "w") as file:
        file.writelines(packages)


def get_variable_dictionary(given_class) -> dict:
    return {
        key: value
        for key, value in given_class.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }


def get_accelerator_type(path: str) -> str:
    files = listdir(path)
    accelerator_type = "none"

    for file in files:
        if file.endswith(".onnx"):
            accelerator_type = "onnx"
            break
        if file.endswith(".bin"):
            accelerator_type = "bettertransformer"
            break

    return accelerator_type
