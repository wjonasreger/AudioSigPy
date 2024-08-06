from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="AudioSigPy",
    version="1.0.0",
    description="A simple audio and speech signal processing package featuring basic tools for speech processing and audio synthesis for music generation.",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wjonasreger/AudioSigPy",
    author="wjonasreger",
    author_email="wjonasreger@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy >= 1.24.3",
        "scipy >= 1.10.1",
        "matplotlib >= 3.7.1",
        "pyloudnorm >= 0.1.1"
    ],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.8",
)