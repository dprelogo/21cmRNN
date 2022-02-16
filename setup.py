from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="rnn21cm",
    version="0.1dev",
    author="David Prelogović",
    author_email="david.prelogovic@gmail.com",
    description="RNNs for 21cm lightcones",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dprelogo/21cmRNN",
    packages=["rnn21cm"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "jax",
        "tensorflow",
        "numpy",
        "scipy",
        "h5py",
        "tools21cm @ git+https://github.com/dprelogo/tools21cm@master#egg=tools21cm",
    ],
    extras_require={"hvd": ["horovod"]},
)
