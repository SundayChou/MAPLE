# MAPLE: resolving tissue microenvironments through spatial multi-modal integration via dual-level graph modeling

![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![Version](https://img.shields.io/badge/version-0.0.1-success.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![DOI](https://zenodo.org/badge/1246637054.svg)](https://doi.org/10.5281/zenodo.20343752)

## 🍁 1. Method Description

MAPLE is a unified deep generative framework designed to resolve spatial tissue microenvironments by integrating spatial multi-omics and histology data.
Driven by an efficient dual-level graph architecture, MAPLE overcomes rigid spatial topologies while enforcing biologically-informed feature constraints.
By generating highly discriminative and biologically interpretable joint representations, it provides a scalable computational infrastructure for comprehensive spatial multi-modal analyses.

![png](overview.png)

## 🛠️ 2. Installation Instructions

We strongly recommend installing MAPLE within a dedicated `Conda` virtual environment to avoid dependency conflicts.
MAPLE is fully developed and tested on `Python 3.13.12`.

💡 **NOTE:** Please ensure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

### 2.1. Clone the repository

First, clone the MAPLE repository to your local machine and navigate into the project root directory:

```bash
git clone https://github.com/SundayChou/MAPLE.git
cd MAPLE
```

### 2.2. Option A: Install via environment.yml (Recommended)

The easiest way to set up the environment is to use the provided `environment.yml` file.
This will automatically create an environment named `maple_env` with `Python 3.13.12` and install all required dependencies.
Finally, install MAPLE in editable mode.

```bash
conda env create -f environment.yml -y
conda activate maple_env
pip install -e .
```

💡 **NOTE:** The `pip install -e .` command installs MAPLE in "editable" mode.
This maps the local source code to your Python environment, allowing you to use `import maple` globally without moving the underlying files.

### 2.3. Option B: Install via requirements.txt (Alternative)

If you prefer to configure the environment manually using `pip`, you can create a fresh Conda environment, install the dependencies from `requirements.txt`, and then install MAPLE:

```bash
conda create -n maple_env python=3.13.12 -y
conda activate maple_env
pip install -r requirements.txt
pip install -e .
```

### 2.4. Verification

Once the installation is complete, you can verify it by running a quick import test in your terminal:

```bash
python -c 'import maple; print(f"maple v{maple.__version__} installed successfully!")'
```

## 📄 3. Tutorial Documents

We provide step-by-step Jupyter Notebook tutorials to demonstrate how to run MAPLE to obtain multi-modal joint embeddings across different integration scenarios.

All tutorials are located in the `tutorials/` directory:

* `1_sim_turorial.ipynb`: Integration of simulated spatial tri-omics dataset.
* `2_hln_tutorial.ipynb`: Integration of spatial transcriptome-proteome human lymph node dataset.
* `3_mbc_tutorial.ipynb`: Integration of spatial epigenome-transcriptome mouse brain coronal dataset.
* `4_ht_tutorial.ipynb`: Integration of spatial transcriptome-proteome with histology image human tonsil dataset.

## 📂 4. File Acquisition

All required files for running MAPLE (including benchmarking datasets, biological priors, and pre-trained models) are publicly available.
You can **[CLICK HERE](https://drive.google.com/drive/folders/1mTk-NJCZUnqpKzqtyrS-Gasldj-4Bv1Y?usp=drive_link)** to download all processed files.

💡 **NOTE:** The cloned repository already provides the empty folder structure.
To ensure the notebooks in the `tutorials/` folder run seamlessly, please place the downloaded files into their corresponding subdirectories within the `data/` folder.
The final directory structure should look as follows:

```text
data/
├── feature_prior/
│   ├── go_graphs/
│   ├── tf_graphs/
│   ├── go-basic.obo
│   ├── mm10.fa
│   ├── peak2gene.csv
│   └── pro2gene.csv
├── hipt_model/
│   ├── vit4k_xs_dino.pth
│   └── vit256_small_dino.pth
├── human_lymph_node/
│   ├── adata_pro.h5ad
│   └── adata_rna.h5ad
├── human_tonsil/
│   ├── adata_pro.h5ad
│   ├── adata_rna.h5ad
│   ├── he_raw.tif
│   ├── scalefactors_json.json
│   └── tissue_positions_list.csv
├── mouse_brain_coronal/
│   ├── adata_atac.h5ad
│   └── adata_rna.h5ad
└── simulated/
    ├── adata_atac.h5ad
    ├── adata_pro.h5ad
    └── adata_rna.h5ad
```

## 📬 5. Contact Information

Please contact us if you have any questions:
- Zhipeng Zhou (zhouzhp23@mail2.sysu.edu.cn);
- Shenshen Bu (bushsh@alumni.sysu.edu.cn);
- Yang Zhang (zhangy2569@mail2.sysu.edu.cn);
- Zhiming Dai **(Corresponding Author)** (daizhim@mail.sysu.edu.cn).

## ⚖️ 6. Copyright Information

Please see the `LICENSE` file for the copyright information.
