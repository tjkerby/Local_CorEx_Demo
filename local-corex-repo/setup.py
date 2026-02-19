from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="local-corex",
    version="0.1.0",
    author="TomKerbyLab",
    description="Local Correlation Explanation and utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomKerbyLab/Local_CorEx",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9,<3.13",
    install_requires=[
        "datasets>=3.4.1",
        "huggingface-hub>=0.29.3",
        "imageio>=2.37.0",
        "ipykernel>=6.29.5",
        "ipywidgets>=8.1.5",
        "numpy<2.0",
        "pandas<2.0",
        "phate>=1.0.11",
        "pytorch-lightning>=2.5.0.post0",
        "scikit-learn>=1.6.1",
        "scprep>=1.2.3",
        "seaborn>=0.13.2",
        "tensorboard>=2.19.0",
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "tqdm>=4.67.1",
        "ucimlrepo>=0.0.7",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
