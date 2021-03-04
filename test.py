import torch
import pynvml
import os
import torch.utils.cpp_extension

torch.ops.load_library('build/lib.macosx-11-arm64-3.9/test.cpython-39-darwin.so')


def checkgpu(device=None):
    i = device if device else torch.cuda.current_device()
    t = torch.cuda.get_device_properties(i).total_memory
    c = torch.cuda.memory_reserved(i)
    name = torch.cuda.get_device_properties(i).name
    print('   GPU Memory Cached (pytorch) : {:7.1f}MB / {:.1f}MB ({})'.format(c / 1024 / 1024, t / 1024 / 1024, name))
    real_i = int(os.environ['CUDA_VISIBLE_DEVICES'][0]) if 'CUDA_VISIBLE_DEVICES' in os.environ else i
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(real_i)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    name = pynvml.nvmlDeviceGetName(h)
    print('   GPU Memory Used (nvidia-smi): {:7.1f}MB / {:.1f}MB ({})'.format(info.used / 1024 / 1024, info.total / 1024 / 1024, name.decode()))
    return f'{(info.used / 1024 / 1024):.1f}MB'


for i in range(10):
    x = torch.zeros(1024*1024*50, requires_grad=True, device='cuda')
    y = torch.ops.test1.forward(x)
    g = torch.autograd.grad(y, x, y, create_graph=True, retain_graph=True)[0]
    g.sum().backward()
    print(i)
    checkgpu()
    print('-' * 70)
