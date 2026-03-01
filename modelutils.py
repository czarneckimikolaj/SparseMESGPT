import torch
import torch.nn as nn

# Check for Apple Silicon GPU
if torch.backends.mps.is_available():
    DEV = torch.device("mps")
elif torch.cuda.is_available():
    DEV = torch.device("cuda")
else:
    DEV = torch.device("cpu")


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
