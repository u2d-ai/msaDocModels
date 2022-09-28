#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --in-place msaDocModels docs_src --exclude=__init__.py
black msaDocModels docs_src
isort msaDocModels docs_src
