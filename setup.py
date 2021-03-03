import os
import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension


setup(
    name='test',
    ext_modules=[
        CppExtension(
            name='test',
            pkg='test',
            sources=glob.glob('*.cpp'))
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
