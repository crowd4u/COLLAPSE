# Base image
FROM python:3.10-bullseye

# Set environment variable for CmdStanPy version
ENV CMDSTANPY_VERSION=1.2.5

# Install required packages including python2 and curl
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    build-essential \
    python2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 2
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py && \
    python2 get-pip.py && \
    rm get-pip.py && \
    python2 -m pip install numpy==1.16.6 pandas==0.24.2 \
                           python-dateutil==2.9.0.post0 pytz==2025.2 six==1.17.0
 
# Install CmdStanPy and dependencies
RUN pip3 install matplotlib==3.7.1 numpy==1.24.3 scipy==1.10.1 \
                sympy==1.11.1 pandas==2.0.1 arviz==0.20.0 \
                jupyterlab==4.3.4 ipykernel==6.22.0 nest-asyncio==1.6.0

RUN pip3 install cmdstanpy[all]==$CMDSTANPY_VERSION
RUN python3 -c 'import cmdstanpy; cmdstanpy.install_cmdstan(version="2.32.2")'

RUN pip3 install crowd-kit==1.3.0.post0 ipywidgets
RUN pip3 install polars==1.19.0
RUN pip3 install streamlit==1.32.2

# Set working directory
WORKDIR /app
