import torch
import torch.nn as nn
import json

from tensor_to_json import tensor_to_dict

# Conv2dレイヤー作成（バイアスなし）
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1,bias=True)

# 入力テンソル作成
x = torch.randn(1, 3, 32, 32, requires_grad=True)

# forward
y = conv(x)

# forward結果を辞書化
input_dict = tensor_to_dict(x)
output_dict = tensor_to_dict(y)
filter_dict = tensor_to_dict(conv.weight)
bias_dict = tensor_to_dict(conv.bias)

# backward用のダミー勾配を作成してbackwardを実行
dy = torch.ones_like(y)
y.backward(dy)

# backward後の勾配を辞書化
grad_input_dict = tensor_to_dict(x.grad)
grad_weight_dict = tensor_to_dict(conv.weight.grad)
grad_bias_dict = tensor_to_dict(conv.bias.grad)

# 保存用の辞書をまとめる
results = {
    "input": input_dict,
    "output": output_dict,
    "filter": filter_dict,
    "bias": bias_dict,
    "grad_input": grad_input_dict,
    "grad_weight": grad_weight_dict,
    "grad_bias": grad_bias_dict
}

# JSONとして保存
with open("conv_results.json", "w") as f:
    json.dump(results, f, indent=2)

