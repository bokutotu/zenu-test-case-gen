# import argparse
#
# import torch
#
# from tensor_to_dict import tensor_to_dict
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='GRU')
#     parser.add_argument("--input_size", type=int, default=10)
#     parser.add_argument("--hidden_size", type=int, default=20)
#     parser.add_argument("--num_layers", type=int, default=1)
#     parser.add_argument("--is_bidirectional", type=bool, default=False)
#     parser.add_argument("--batch_size", type=int, default=True)
#     parser.add_argument("--seq_len", type=int, default=3)
#     return parser.parse_args()
#
#
# def gru(args):
#     from torch import nn
#
#     input_size = args.input_size
#     hidden_size = args.hidden_size
#     num_layers = args.num_layers
#     is_bidirectional = args.is_bidirectional
#
#     gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=is_bidirectional, batch_first=True)
#     return gru
#
#
# def main():
#     args = parse_args()
#     layer = gru(args)
#     layer.train()
#     input = torch.randn(args.batch_size, args.seq_len, args.input_size)
#     input.requires_grad = True
#     output, hidden = layer(input)
#
#     output.backward(torch.randn_like(output))
#
#     output_dict = {}
#
#     output_dict["input"] = tensor_to_dict(input)
#     output_dict["input_grad"] = tensor_to_dict(input.grad)
#
#     for (k, v) in layer.state_dict().items():
#         output_dict[k] = tensor_to_dict(v)
#         output_dict[k + "_grad"] = tensor_to_dict(v.grad)
#
#     output_dict["output"] = tensor_to_dict(output)
#     output_dict["hidden"] = tensor_to_dict(hidden)
#     output_dict["output_grad"] = tensor_to_dict(output.grad)
#
#     ## save as json
#     import json
#     with open("gru.json", "w") as f:
#         json.dump(output_dict, f, indent=2)
#
#
# if __name__ == "__main__":
#     main()
import argparse
import torch
from torch import nn
from tensor_to_dict import tensor_to_dict


def parse_args():
    parser = argparse.ArgumentParser(description='GRU')
    parser.add_argument("--input_size", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=20)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--is_bidirectional", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=3)
    return parser.parse_args()


def gru(args):

    input_size = args.input_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    is_bidirectional = args.is_bidirectional
    print("is_bidirectional", is_bidirectional)

    gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=is_bidirectional,
        batch_first=False
    )
    return gru


def main():
    args = parse_args()
    layer = gru(args)

    input = torch.randn(args.batch_size, args.seq_len, args.input_size, requires_grad=True)
    output, hidden = layer(input)
    output.retain_grad()
    hidden.retain_grad()

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)
    print("Hidden shape:", hidden.shape)
    # Define a simple loss function
    loss = output.sum()

    # Backpropagate
    loss.backward()

    output_dict = {
        "input": tensor_to_dict(input),
        "input_grad": tensor_to_dict(input.grad),
        "output": tensor_to_dict(output),
        "hidden": tensor_to_dict(hidden),
        "output_grad": tensor_to_dict(output.grad)
    }

    # Extracting gradients of GRU parameters
    for name, param in layer.named_parameters():
        name = "rnn." + name
        output_dict[name] = tensor_to_dict(param.data)
        output_dict[name + "_grad"] = tensor_to_dict(param.grad)
        print(name, param.shape)

    # Save as JSON
    import json
    with open("gru.json", "w") as f:
        json.dump(output_dict, f, indent=2)


if __name__ == "__main__":
    main()


