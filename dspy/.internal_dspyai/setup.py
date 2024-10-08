from setuptools import find_packages, setup

# Read the content of the README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    #replace_package_name_marker
    name="dspy-ai",
    #replace_package_version_marker
    version="2.5.5", 	
    description="DSPy",	
    long_description=long_description,	
    long_description_content_type="text/markdown",	
    url="https://github.com/stanfordnlp/dsp",	
    author="Omar Khattab",	
    author_email="okhattab@stanford.edu",	
    license="MIT License",	
    packages=find_packages(include=["dsp.*", "dspy.*", "dsp", "dspy"]),	
    python_requires=">=3.9",
    #replace_dspy_version_marker
    install_requires=["dspy==2.5.3"]
)	
