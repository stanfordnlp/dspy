from setuptools import find_packages, setup

# Read the content of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read the content of the requirements.txt file
with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(	
    name="dspy-ai",	
    version="2.4.10",	
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
        "chromadb": ["chromadb~=0.4.14"],
        "qdrant": ["qdrant-client", "fastembed"],
        "marqo": ["marqo~=3.1.0"],
        "mongodb": ["pymongo~=3.12.0"],
        "pinecone": ["pinecone-client~=2.2.4"],
        "weaviate": ["weaviate-client~=3.26.1"],
        "faiss-cpu": ["sentence_transformers", "faiss-cpu"],
        "milvus": ["pymilvus~=2.3.7"],
        "google-vertex-ai": ["google-cloud-aiplatform==1.43.0"],
        "myscale":["clickhouse-connect"],
        "snowflake": ["snowflake-snowpark-python"],
        "fastembed": ["fastembed"],
        "google-vertex-ai": ["google-cloud-aiplatform==1.43.0"],
        "myscale":["clickhouse-connect"],
        "groq": ["groq~=0.8.0"],
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
