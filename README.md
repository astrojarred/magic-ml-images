# MAGIC ML

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Google Colab](#colab)
- [The Data](#the_data)

## About <a name = "about"></a>

This package contains the code for reading and plotting parquet tables of images from the [MAGIC telescopes](https://magic.mpp.mpg.de/).

## Getting Started <a name = "getting_started"></a>

Clone the repository:

```
git clone https://github.com/astrojarred/magic-ml-images.git
cd magic-ml-images
```


### Prerequisites and Installation

With whatever package manager you prefer, install the following dependencies:

- **python 3.13+**
- matplotlib
- numpy
- pandas
- polars (needed to lazy-loading data)
- pyarrow
- ipython
- ipykernel

For example, using conda:

```
conda create -n magic-ml python=3.13 matplotlib numpy pandas polars pyarrow ipython ipykernel
```

or using pip:

```
pip install matplotlib numpy pandas polars pyarrow ipython ipykernel
```

or using `uv`:

```
uv init -p 3.13
uv add matplotlib numpy pandas polars pyarrow ipython ipykernel
```

## Pytorch Installation

If you want to run `pytorch.ipynb`, you will need to add pytorch in your environment. The installation method depends on your system and whether or not you have a GPU. 

Please see the [Pytorch Installation Guide](https://pytorch.org/get-started/locally/) for more details.

## Usage <a name = "usage"></a>

Open the example notebook: `example.ipynb` to explore some MAGIC data!

Open the notebook: `pytorch.ipynb` to see an example of how to load the parquet files into a PyTorch Dataset and DataLoader.

## Running on Google Colab <a name = "colab"></a>

In order to run your notebooks on Google Colab, please write me and I'll provide you with a data downloading script.

The rest of the needed packages should already be installed into the default environment, and you can just copy/paste the cells from the example notebooks that you need.

## The Data  <a name = "the_data"></a>

- You will be provided with the data in the form of parquet files.
- The most up-to-date files are the `magic-gammas-chunked.parquet` and `magic-protons-chunked.parquet` files. They contain the same events as the other files, except **the data is formatted in a way that supports streaming and lazy-loading**.
- Many more gamma events are provided than proton events. This is up to you to decide how to handle this.
- The gamma events are stored in 2 ways:
  - All the gammas in one big file: `magic-gammas.parquet`
  - The exact same gamma events split into 4 smaller files: `magic-gammas-1.parquet`, `magic-gammas-2.parquet`, `magic-gammas-3.parquet`, `magic-gammas-4.parquet`



