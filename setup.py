from setuptools import setup, find_packages

setup(
    name='maple',
    version='0.0.1',
    description='MAPLE: resolving spatiotemporal tissue microenvironments through spatial multi-modal integration via explicit dual-level graph modeling',
    author='Zhipeng Zhou, Shenshen Bu, Yang Zhang, Zhiming Dai',
    author_email='zhouzhp@mail2.sysu.edu.cn',
    maintainer='Zhiming Dai (Corresponding Author)',
    maintainer_email='daizhim@mail.sysu.edu.cn',
    packages=find_packages(include=['maple', 'maple.*']),
    include_package_data=True,
    package_data={'maple': ['checkpoints/*.pth']},
    python_requires='>=3.13',
)