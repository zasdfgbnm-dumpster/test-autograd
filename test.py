import torch
import torch.utils.cpp_extension

torch.ops.load_library('build/lib.macosx-11-arm64-3.9/test.cpython-39-darwin.so')
