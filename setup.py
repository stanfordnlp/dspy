from setuptools import setup, find_packages	

# Read the content of the README file	
with open('README.md', 'r', encoding='utf-8') as f:	
    long_description = f.read()	

# Read the content of the requirements.txt file	
with open('requirements.txt', 'r', encoding='utf-8') as f:	
    requirements = f.read().splitlines()	

setup(	
    name="dspy-ai",	
    version="2.1.6",	
    description="DSPy",	
    long_description=long_description,	
    long_description_content_type='text/markdown',	
    url="https://github.com/stanfordnlp/dsp",	
    author="Omar Khattab",	
    author_email="okhattab@stanford.edu",	
    license="MIT License",	
    packages=find_packages(include=['dsp.*', 'dspy.*', 'dsp', 'dspy']),	
    python_requires='>=3.9',	
    install_requires=requirements,	
    extras_require={	
        "pinecone": ["pinecone-client~=2.2.4"],	
        "qdrant": ["qdrant-client~=1.6.2", "fastembed~=0.1.0"],	
        "chromadb": ["chromadb~=0.4.14"],	
        "marqo": ["marqo"],	
        "weaviate": ["weaviate-client~=3.26.1"],	
        "mongodb": ["pymongo~=3.12.0"],	
    },	
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
