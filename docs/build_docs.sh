#!/bin/bash

# Must be run from the docs directory

# Remove the old api_reference files
rm -rf docs/source/api_reference/*.rst
sphinx-apidoc -o docs/source/api_reference qtmlib

# Run the Python script to update api_reference index.rst

python docs/source/api_reference/build_index.py

# Build the docs using Sphinx
make -f docs/Makefile html

# cp -r build/html/* .