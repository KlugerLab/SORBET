# SORBET – Spatial 'Omics Reasoning for Binary Label Tasks

SORBET is a geometric deep-learning framework that utilizes graph-convolutional neural networks (GCN) to classify tissues according to observed phenotypes, such as response to immunotherapy. SORBET classifies tissues directly from spatially-resolved omics measurements (spatial proteomics / transcriptomics). 

The model is broken into a `Learning` phase, where tissues are classified, and a `Reasoning` phase, where the inferred models are analyzed for biological insight. The code shares that structure, with sub-modules for `learning` and `reasoning`. In addition, the code includes a sub-module for `data_handling`.  

![overview_figure](assets/Overview.png)

## Requirements and Installation

SORBET has been tested on Linux 20.04.5 system with CUDA 11.7. Code was developed in Python (v3.10).

As a deep learning model, code will run more efficiently on GPUs. Models typically fit within 12 GB. 

The package is distributed with a `Dockerfile` for setting up an appropriate environment. Alternatively, the requirements may be installed from the `requirements.txt` file. Note, some packages (primarily the `torch-geometric` stack) must be manually installed after the installation of the packages in the `requirements.txt` file. Please see the `Dockerfile` for additional requirements (lines: 16-21). 

### Requirements:
In addition to the system and language requirements, the package was developed using the packages
```
numpy<2.0 scipy scikit-learn pandas setuptools ipykernel matplotlib<3.10 seaborn tqdm networkx statsmodels umap-learn torch==1.13.1 tensorboard optuna ray[tune]=2.6.2 anndata scanpy[leiden] openpyxl geovoronoi omnipath rbo pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch-geometric
``` 

A listing of the complete versions for installed packages is:
<details>
<summary>Package Versions</summary>
  
```
! pip freeze
absl-py==2.2.2
aiohappyeyeballs==2.6.1
aiohttp==3.11.16
aiosignal==1.3.2
alembic==1.15.2
anndata==0.11.4
anyio==4.9.0
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
array_api_compat==1.11.2
arrow==1.3.0
asttokens==3.0.0
async-lru==2.0.5
async-timeout==5.0.1
attrs==25.3.0
babel==2.17.0
beautifulsoup4==4.13.3
biopython==1.85
bleach==6.2.0
certifi==2025.1.31
cffi==1.17.1
chardet==5.2.0
charset-normalizer==3.4.1
click==8.1.8
colorlog==6.9.0
comm==0.2.2
contourpy==1.3.1
cycler==0.12.1
debugpy==1.8.13
decorator==5.2.1
defusedxml==0.7.1
diffexp==0.1
distlib==0.3.9
docrep==0.3.2
et_xmlfile==2.0.0
exceptiongroup==1.2.2
executing==2.2.0
fastjsonschema==2.21.1
filelock==3.18.0
fonttools==4.57.0
fqdn==1.5.1
frozenlist==1.5.0
fsspec==2025.3.2
geovoronoi==0.4.0
greenlet==3.1.1
grpcio==1.71.0
h11==0.14.0
h5py==3.13.0
httpcore==1.0.7
httpx==0.28.1
idna==3.10
igraph==0.11.8
inflect==7.5.0
iniconfig==2.1.0
ipykernel==6.29.5
ipython==8.34.0
ipywidgets==8.1.5
isoduration==20.11.0
jedi==0.19.2
Jinja2==3.1.6
joblib==1.4.2
json5==0.12.0
jsonpointer==3.0.0
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
jupyter==1.1.1
jupyter-console==6.6.3
jupyter-events==0.12.0
jupyter-lsp==2.2.5
jupyter_client==8.6.3
jupyter_core==5.7.2
jupyter_server==2.15.0
jupyter_server_terminals==0.5.3
jupyterlab==4.3.6
jupyterlab_pygments==0.3.0
jupyterlab_server==2.27.3
jupyterlab_widgets==3.0.13
kiwisolver==1.4.8
legacy-api-wrap==1.4.1
leidenalg==0.10.2
llvmlite==0.44.0
Mako==1.3.9
Markdown==3.7
MarkupSafe==3.0.2
matplotlib==3.9.4
matplotlib-inline==0.1.7
mistune==3.1.3
more-itertools==10.6.0
msgpack==1.1.0
multidict==6.3.2
natsort==8.4.0
nbclient==0.10.2
nbconvert==7.16.6
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.4.2
notebook==7.3.3
notebook_shim==0.2.4
numba==0.61.0
numpy==1.26.4
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
omnipath==1.0.9
openpyxl==3.1.5
optuna==4.2.1
overrides==7.7.0
packaging==24.2
pandas==2.2.3
pandocfilters==1.5.1
parso==0.8.4
patsy==1.0.1
pexpect==4.9.0
pillow==11.1.0
platformdirs==4.3.7
pluggy==1.5.0
prometheus_client==0.21.1
prompt_toolkit==3.0.50
propcache==0.3.1
protobuf==6.30.2
psutil==7.0.0
ptyprocess==0.7.0
pure_eval==0.2.3
pyarrow==19.0.1
pycparser==2.22
pyg-lib==0.4.0+pt113cu117
Pygments==2.19.1
pynndescent==0.5.13
pyparsing==3.2.3
pytest==8.3.5
python-dateutil==2.9.0.post0
python-json-logger==3.3.0
pytz==2025.2
PyYAML==6.0.2
pyzmq==26.4.0
ray==2.6.2
rbo==0.1.3
referencing==0.36.2
reportlab==4.3.1
requests==2.32.3
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rpds-py==0.24.0
rpy2==3.5.17
scanpy==1.11.1
scikit-learn==1.5.2
scipy==1.15.2
seaborn==0.13.2
Send2Trash==1.8.3
session-info2==0.1.2
shapely==2.1.0
six==1.17.0
sniffio==1.3.1
soupsieve==2.6
SQLAlchemy==2.0.40
stack-data==0.6.3
statsmodels==0.14.4
tensorboard==2.19.0
tensorboard-data-server==0.7.2
tensorboardX==2.6.2.2
terminado==0.18.1
texttable==1.7.0
threadpoolctl==3.6.0
tinycss2==1.4.0
tomli==2.2.1
torch==1.13.1
torch-cluster==1.6.1+pt113cu117
torch-geometric==2.6.1
torch-scatter==2.1.1+pt113cu117
torch-sparse==0.6.17+pt113cu117
torch-spline-conv==1.2.2+pt113cu117
tornado==6.4.2
tqdm==4.67.1
traitlets==5.14.3
typeguard==4.4.2
types-python-dateutil==2.9.0.20241206
typing_extensions==4.13.1
tzdata==2025.2
tzlocal==5.3.1
umap-learn==0.5.7
uri-template==1.3.0
urllib3==2.3.0
virtualenv==20.30.0
wcwidth==0.2.13
webcolors==24.11.1
webencodings==0.5.1
websocket-client==1.8.0
Werkzeug==3.1.3
widgetsnbextension==4.0.13
wrapt==1.17.2
yarl==1.19.0
```
</details>

### Docker (Preferred)
Setting up the repo to work with Docker may be accomplished in the following way. Some variables need to be set and are indicated by [text]
``` 
# From repo:
docker build --no-cache --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t [docker container name] .
docker run -it -d -p [remote port]:[remote port] --gpus all --mount "type=bind,src=$(pwd),dst=[workdir name]" --workdir [workdir name] [docker container name] /bin/bash
```
This sets up an interactive session. Next, it is feasible to start-up the Jupyter Notebooks and listen on a chosen port as follows:
```
# Start a jupyter notebook:
docker attach [x]                                         # Replace [x] with the appropriate docker name ($ docker ps)
jupyter notebook password                                 # Set password in session
jupyter notebook --ip 0.0.0.0 --port 7676 --no-browser    # Can change port
```

Build time is typically under 30 minutes. 

### Troubleshooting: 
The `requirements.txt` file contains the majority of the packages dependencies. The required packages for `torch-geometric` install with Docker required setting some CUDA dependencies. These may need to be updated, depending on the machine.  

## Getting Started 
Three Python notebooks demonstrate how to use each of the three major modules (`data_handling`, `learning`, `reasoning`). Please run these in order:
1. `Tutorial_1_Data_Handling.ipynb`: Conversion of external data format to a common format (`data_handling.OmicsGraph`) and pre-processing / subgraph extraction.
2. `Tutorial_2_Learning.ipynb`: Modeling of data using GNN's
3. `Tutorial_3_Reasoning.ipynb`: Analysis of inferred models / model reasoning.  

These notebooks reference the core functions (and their options) for using the implemented code.  

The run time for examples depends on exact machine configuration. The data handling example should take under 30 minutes. A single model in the learning example, trained on a computer with a GPU, should take minutes to finish. Hyperparameter optimization will take longer and depends on available GPUs. Model reasoning should take around 1.5 hours to run, with the major time constraint arising in the sparse CCA step.

## Reference

For additional information, please reference our pre-print:

> **SORBET: Automated cell-neighborhood analysis]{SORBET: Automated cell-neighborhood analysis of spatial transcriptomics or proteomics for interpretable sample classification via GNN** 
>
> Shay Shimonov, Joseph M Cunningham, Ronen Talmon, Lilach Aizenbud, Shruti J Desai, David Rimm, Kurt Schalper, Harriet Kluger, Yuval Kluger 
>
> _bioRxiv_ 2024 January 3. doi: [https://doi.org/10.1101/2023.12.30.573739](10.1101/2023.12.30.573739)
