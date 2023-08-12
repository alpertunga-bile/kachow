# kachow
Small library for using [optimum package](https://github.com/huggingface/optimum) for text generation training and inference

- Can use text file as dataset for training.

## Setup

- Clone the repository
- Run the ```launch.py```. ```VenvManager``` is going to setup the virtual environment. 
- Can use the functions in the ```main.py``` file. 
- For running the ```main.py``` file, use ```python launch.py``` command. It is going to run the ```main.py``` file in the virtual environment.
 
## Examples

### Generate

#### Generate with transformers package

```python
from utilities.generate import generate_text, GenerateArgs

default_args = GenerateArgs()
default_args.min_length = 30
default_args.max_length = 75

generate_text("Hello", args=default_args)
```

#### Generate with optimum package using BetterTransformer and ONNX models

```python
from utilities.generate import generate_text, GenerateArgs

default_args = GenerateArgs()
default_args.min_length = 30
default_args.max_length = 75

generate_text("Hello", args=default_args, is_accelerate=True)
```

### Train
#### Train with transformers package
```python
from utilities.consts import default_train_args
from utilities.train import train

train_args = default_train_args

train_args.learning_rate = 1e-3
train_args.num_train_epochs = 3
train_args.auto_find_batch_size = True
train_args.torch_compile = True

train(train_file="dataset.txt", save_path="my_model", train_args=train_args)
```

#### Train with transformers package using BetterTransformer
```python
from utilities.consts import default_train_args
from utilities.train import train

train_args = default_train_args

train_args.learning_rate = 1e-3
train_args.num_train_epochs = 3
train_args.auto_find_batch_size = True
train_args.torch_compile = True

train(train_file="dataset.txt", save_path="my_model", train_args=train_args, use_sdp=True)
```

#### Train with optimum package using ONNX
```python
from utilities.consts import default_onnx_train_args
from utilities.train import train_onnx

train_args = default_onnx_train_args

train_onnx(train_file="dataset.txt", save_path="my_model", train_args=train_args)
```

### Convert

#### transformers model to ONNX model

- ONNX model is going to appear in the same parent folder with the base model. Folder name suffix for converted models is ```_onnx```. For this example, the folder name is going to be ```my_model_onnx```.

```python
from utilities.model import optimize_and_quantize_onnx

optimize_and_quantize_onnx("my_model")
```