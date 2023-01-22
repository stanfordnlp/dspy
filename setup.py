from setuptools import setup

setup(
    name="dsp",
    version="0.1.0",
    description="Demonstrate-Search-Predict",
    url="https://github.com/stanfordnlp/dsp",
    author="Omar Khattab",
    author_email="okhattab@stanford.edu",
    license="MIT License",
    packages=["dsp"],
    python_requires='>=3.8',
    install_requires=[
        "backoff",
        "joblib",
        "jupyter",
        "openai",
        "pandas",
        "spacy",
        "regex",
    ],
    classifiers=[
		"Development Status :: 3 - Alpha",
		"Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
