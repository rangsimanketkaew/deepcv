import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepcv",
    version="2.0",
    author="Rangsiman Ketkaew",
    author_email="rangsiman.ketkaew@chem.uzh.ch",
    description="Unsupervised machine learning package for discovering collective variables",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.uzh.ch/lubergroup/deepcv",
    download_url="https://gitlab.uzh.ch/lubergroup/deepcv/-/releases",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    keywords=[
        "deep learning",
        "autoencoder",
        "chemistry",
        "computational chemistry",
        "theoretical chemistry",
        "molecular dynamics",
        "enhanced sampling",
        "metadynamics",
        "collective variables",
        "free energies",
    ],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "deepcv=src.main:main",
        ]
    },
)
