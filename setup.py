from setuptools import setup, find_packages

setup(
    name='FastAudioSR',
    version='0.0.1',
    packages=find_packages(),
    author='johnwick123f',
    description='A fast speech recognition package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/johnwick123f/FastAudioSR',
    license='MIT',
    install_requires=[
        'soxr',
        'timm',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
