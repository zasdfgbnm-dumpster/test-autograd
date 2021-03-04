import torch
import tqdm
import glob

g = glob.glob("build/*/*.so")
for f in g:
    if f.endswith(".so"):
        break

torch.ops.load_library(f)

for i in tqdm.trange(100000):
    x = torch.zeros(1024*1024*512, requires_grad=True)
    y = torch.ops.test.forward(x).sum()
    g = torch.autograd.grad(y, x, create_graph=True, retain_graph=True)[0]
    g.sum().backward()
