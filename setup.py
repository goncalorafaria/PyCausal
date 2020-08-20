
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
    name="pycausal",
    version="0.0.1",
    author="Gonçalo Faria",
    author_email="goncalorafaria@tecnico.ulisboa.pt",
    description="Package for defining Structural Causal Models and for Structure Identification from data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/goncalorafaria/PyCausal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    python_requires='>=3.6'
)
