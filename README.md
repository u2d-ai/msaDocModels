<p align="center">
  <img src="http://logos.u2d.ai/msaDocModels_logo.png?raw=true" alt="msaDocModels Logo"/>
</p>

------
<p align="center">
    <em>msaDocModels - MSA Document Pydantic Models and Schemas, used to store Parser, NLP, NLU and AI results for processed documents</em>
<br>
    Optimized for use with FastAPI/Pydantic.
<br>
  <a href="https://pypi.org/project/msaDocModels" target="_blank">
      <img src="https://img.shields.io/pypi/v/msaDocModels?color=%2334D058&label=pypi%20package" alt="Package version">
  </a>
  <a href="https://pypi.org/project/msaDocModels" target="_blank">
      <img src="https://img.shields.io/pypi/pyversions/msaDocModels.svg?color=%2334D058" alt="Supported Python versions">
  </a>
</p>

------

**Documentation**: <a href="https://msaDocModels.u2d.ai/" target="_blank">msaDocModels Documentation (https://msaDocModels.u2d.ai/)</a>

------

## Features
- **Schema/Models for Document Understanding Result Data**: sdu.
- **Schema/Models for General Document Handling Data**: wdc.
- **Schema/Models for Workflow Definition and Processing Data**: wfl.
- **Schema/Models for Work With Text**: spk.
- **API Message class**: msg, allows generic API JSON message creation with capabilities to re-create original datatypes and class instances.


## Main Dependencies

- msaUtils >= 0.0.2
- Pydantic



## License Agreement

- `msaDocModels` is based on `MIT` open source and free to use, it is free for commercial use, but please show/list the copyright information about msaDocModels somewhere.


## How to create the documentation

We use mkdocs and mkdocsstring. The code reference and nav entry get's created virtually by the triggered python script /docs/gen_ref_pages.py while ``mkdocs`` ``serve`` or ``build`` is executed.

### Requirements Install for the PDF creation option:
PDF Export is using mainly weasyprint, if you get some errors here pls. check there documentation. Installation is part of the msaDocModels, so this should be fine.

We can now test and view our documentation using:

    mkdocs serve

Build static Site:

    mkdocs build


## Build and Publish
  
Build:  

    python setup.py sdist

Publish to pypi:

    twine upload dist/*
