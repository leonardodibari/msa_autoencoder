from setuptools import setup, find_packages

setup(
    name="msa_autoencoder",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "biopython",
        "scikit-learn",
        "matplotlib",
        "tqdm"
    ],
)
