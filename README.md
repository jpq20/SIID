# SIID: Spatial Integration for Imputation and Deconvolution
SIID integrates two Spatially Resolved Transcriptomics (SRT) datasets with different gene panel and spatial resolution, to simultaneously impute missing genes and infer admixed cell types. SIID is based on a joint Nonnegative Matrix Factorization (NMF) with spatially matched spots.

## Basic Usage
We provide wrapper functions at ``src/lib_helper.py``.

For imputation, ``lib_helper.impute_genes(vi, xe, holdouts, hdim)`` takes the coordinate-aligned Visium and Xenium AnnData objects, a list of holdout genes (must be a subset of Visium genes) and number of hidden dimensions, and returns an AnnData object with imputed gene expression for each Xenium cell.

For deconvolution, ``lib_helper.infer_latent_types(vi, xe, hdim)`` takes the coordinate-aligned Visium and Xenium AnnData objects and number of hidden dimensions, and returns two AnnData objects representing inferred cell type compositions for (deconvolved) Visium and Xenium spots.

For both use cases, an extra parameter ``R`` (a 3-by-3 numpy array) can be passed to transform Visium coordiantes (in homogeneous coordinate system), and ``device`` can be used to specify where the model runs. We strongly suggest using a GPU.

## Advanced Usage

The ``lib_helper`` functions contain steps for preprocessing, as well as generation of $\Gamma$, the spatial mapping matrix.
The core model code lies in ``nmf_v01.py`` and class ``LowDimWithScaling``, which contains more parameters for tuning. 

## Examples

See ``notebook/BRCA-data.ipynb`` for example usage for imputing holdout genes in a paired Xenium-Visium breast cancer slice. 