# coding: utf-8
from setuptools import setup

setup(name='heinsen_position_embeddings',
    version='1.0.1',
    description='Implementation of "Encoding Position by Decaying and Updating Different Exponentiated States Differently" (Heinsen, 2024).',
    url='https://github.com/glassroom/heinsen_position_embeddings',
    author='Franz A. Heinsen',
    author_email='franz@glassroom.com',
    license='MIT',
    packages=['heinsen_position_embeddings'],
    install_requires='torch',
    zip_safe=False)
