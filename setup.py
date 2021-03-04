import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name='test',
    ext_modules=[
        CUDAExtension(
            name='test',
            pkg='test',
            sources=glob.glob('*.cu'))
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
