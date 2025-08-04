# SORBET – Spatial 'Omics Reasoning for Binary Label Tasks

SORBET is a geometric deep-learning framework that utilizes graph-convolutional neural networks (GCN) to classify tissues according to observed phenotypes, such as response to immunotherapy. SORBET classifies tissues directly from spatially-resolved omics measurements (spatial proteomics / transcriptomics). 

The model is broken into a `Learning` phase, where tissues are classified, and a `Reasoning` phase, where the inferred models are analyzed for biological insight. The code shares that structure, with sub-modules for `learning` and `reasoning`. In addition, the code includes a sub-module for `data_handling`.  

## Requirements

SORBET has been tested on Linux 20.04.5 system with CUDA 11.7. 

The package is distributed with a `Dockerfile` for setting up an appropriate environment. Alternatively, the requirements may be installed from the `requirements.txt` file. Note, some packages (primarily the `torch-geometric` stack) must be manually installed after the installation of the packages in the `requirements.txt` file. Please see the `Dockerfile` for additional requirements (lines: 16-21). 

## Reference

For additional information, please reference our pre-print:

> **SORBET: Automated cell-neighborhood analysis]{SORBET: Automated cell-neighborhood analysis of spatial transcriptomics or proteomics for interpretable sample classification via GNN** 
>
> Shay Shimonov, Joseph M Cunningham, Ronen Talmon, Lilach Aizenbud, Shruti J Desai, David Rimm, Kurt Schalper, Harriet Kluger, Yuval Kluger 
>
> _bioRxiv_ 2024 January 3. doi: []()
>

## Get Started 
### Data
Data should be stoered in the following format for Cosmx/ CODEX/ IMC
New data classes can be created with ease (See )
### Train

### Reason
