#!/usr/bin/env bash

set -e
set -x

mypy msaDocModels
flake8 msaDocModels docs_src
black msaDocModels docs_src --check
isort msaDocModels docs_src scripts --check-only

