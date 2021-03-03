import torch
import tqdm
import torch.utils.cpp_extension

torch.ops.load_library('build/lib.macosx-11-arm64-3.9/test.cpython-39-darwin.so')

for i in tqdm.trange(100000):
    x = torch.zeros(1024*1024*512, requires_grad=True)
    y = torch.ops.test.forward(x).sum()
    g = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
    g.sum().backward()
