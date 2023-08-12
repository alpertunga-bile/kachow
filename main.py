from utilities.train import train, train_onnx
from utilities.generate import generate_text, GenerateArgs
from utilities.consts import default_train_args


def main_train():
    train_args = default_train_args

    train_args.learning_rate = 1e-3
    train_args.num_train_epochs = 3
    train_args.auto_find_batch_size = True
    train_args.torch_compile = True


def main_generate():
    default_args = GenerateArgs()
    default_args.min_length = 30
    default_args.max_length = 75


if __name__ == "__main__":
    main_train()
