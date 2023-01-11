# Mid-level Harmonic Audio Features for Musical Style Classification

This repository contains the implementation of the musical style classification system proposed for the ISMIR 2022 submission "Mid-level Harmonic Audio Features for Musical Style Classification" (F. Almeida, G. Bernardes and C. Weiß).

## Getting started

Install the required system dependencies:

- Python 3.8
- Pip package manager
- GNU Make
- Git LFS

Clone this repository and setup Git LFS. Please make sure the `--recursive` flag is passed so that all submodules are also cloned.

```bash
git clone git@github.com:fcfalmeida/style-ident.git --recursive
cd style-ident
git lfs install
```

Install project dependencies by running:

```bash
make install
```

## Usage

Interaction with the system is done through the commands defined in the project's `Makefile`, which includes documentation on how to use them. For example, to compute TIV Basic features using the multiple resolution segmentation approach on the Cross-Era Piano dataset, run the following:

```bash
make res_feats dataset=crossera_piano pipeline=tis_basic_segmented
```

A *pipeline* represents the computation of a feature group using a given segmentation approach. Currently available pipelines are:

**Harmonic Segmentation**
- *tis_complexity_segmented*: TIV Complexity
- *tis_basic_segmented*: TIV Basic
- `*harm_rhythm*: Harmonic Rhythm

**Multiple Resolutions Segmentation**
- *tis_complexity_res*: TIV Complexity
- *tis_basic_res*: TIV Basic
- *complexity*: Tonal Complexity
- *template_based*: Template-based

**Local Resolution Segmentation**
- *tis_complexity_local_res*: TIV Complexity
- *tis_basic_local_res*: TIV Basic

Unprocessed datasets are placed under *data/external* and are provided in this repository for easier reproducibility of research results. The conducted study uses the following datasets:

- *crossera_piano*
- *crossera_orchestra*
- *crossera_full*
- *crosscomp5*
- *crosscomp11*
