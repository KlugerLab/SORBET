# SORBET – Spatial 'Omics Reasoning for Binary Label Tasks

SORBET is a geometric deep-learning framework that utilizes graph-convolutional neural networks (GCN) to classify tissues according to observed phenotypes, such as response to immunotherapy. SORBET classifies tissues directly from spatially-resolved omics measurements (spatial proteomics / transcriptomics). 

The model is broken into a `Learning` phase, where tissues are classified, and a `Reasoning` phase, where the inferred models are analyzed for biological insight. The code shares that structure, with sub-modules for `learning` and `reasoning`. In addition, the code includes a sub-module for `data_handling`.  

![overview_figure](assets/Overview.png)

## Requirements and Installation

SORBET has been tested on Linux 20.04.5 system with CUDA 11.7. Code was developed in Python (v3.10). 

The package is distributed with a `Dockerfile` for setting up an appropriate environment. Alternatively, the requirements may be installed from the `requirements.txt` file. Note, some packages (primarily the `torch-geometric` stack) must be manually installed after the installation of the packages in the `requirements.txt` file. Please see the `Dockerfile` for additional requirements (lines: 16-21). 

### Docker (Preferred)
Setting up the repo to work with Docker may be accomplished in the following way. Some variables need to be set and are indicated by [text]
``` 
# From repo:
docker build --no-cache --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t [docker container name] .
docker run -it -d -p [remote port]:[remote port] --gpus all --mount "type=bind,src=$(pwd),dst=[workdir name]" --workdir [workdir name] [docker container name] /bin/bash
```
This sets up an interactive session. Next, it is feasible to 

### Troubleshooting: 
The `requirements.txt` file contains the majority of the packages dependencies. The required packages for `torch-geometric` install with Docker required setting some CUDA dependencies. These may need to be updated, depending on the machine.  

## Getting Started 
Three Python notebooks demonstrate how to use each of the three major modules (`data_handling`, `learning`, `reasoning`). Please run these in order:
1. `Tutorial_1_Data_Handling.ipynb`: Conversion of external data format to a common format (`data_handling.OmicsGraph`) and pre-processing / subgraph extraction.
2. `Tutorial_2_Learning.ipynb`: Modeling of data using GNN's
3. `Tutorial_3_Reasoning.ipynb`: Analysis of inferred models / model reasooning.  

These notebooks reference the core functions (and their options) for using the implemented code.  

## Reference

For additional information, please reference our pre-print:

> **SORBET: Automated cell-neighborhood analysis]{SORBET: Automated cell-neighborhood analysis of spatial transcriptomics or proteomics for interpretable sample classification via GNN** 
>
> Shay Shimonov, Joseph M Cunningham, Ronen Talmon, Lilach Aizenbud, Shruti J Desai, David Rimm, Kurt Schalper, Harriet Kluger, Yuval Kluger 
>
> _bioRxiv_ 2024 January 3. doi: [https://doi.org/10.1101/2023.12.30.573739](10.1101/2023.12.30.573739)
