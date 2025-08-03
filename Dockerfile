FROM python:3.10
ARG USER_ID
ARG GROUP_ID

RUN apt-get update
RUN apt-get install -y --no-install-recommends
RUN apt-get install -y r-base r-base-dev r-bioc-deseq2 r-bioc-rhdf5 libcurl4-openssl-dev libxml2-dev libssl-dev 
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
WORKDIR /home/joe/spatial_omics_reimp

COPY requirements.txt .
RUN pip install --upgrade pip

# Install diffexpr package for DESeq2 access:
RUN pip install pandas tzlocal biopython ReportLab pytest rpy2
RUN git clone https://github.com/wckdouglas/diffexpr.git /opt/diffexpr 
RUN python /opt/diffexpr/setup.py install
RUN /usr/bin/Rscript -e "BiocManager::install(c('apeglm'))"
ENV PYTHONPATH "${PYTHONPATH}:/opt/diffexpr"

RUN pip install -r requirements.txt
RUN pip install --no-index pyg_lib -f https://data.pyg.org/whl/torch-1.13.1+cu117.html \
&& pip install --no-index torch_scatter -f https://data.pyg.org/whl/torch-1.13.1+cu117.html \
&& pip install --no-index torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html \
&& pip install --no-index torch_cluster -f https://data.pyg.org/whl/torch-1.13.1+cu117.html \
&& pip install --no-index torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html \
&& pip install torch-geometric

#RUN apt-get install -y texlive texstudio texlive-latex-extra texlive-fonts-recommended dvipng cm-super

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
