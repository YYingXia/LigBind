# LigBind
LigBind is a relation-aware framework with graph-level pre-training to enhance the ligand-specific binding residue predictions for 1159 ligands, which can effectively cover the ligands with a few known binding proteins.

We also release a ligand-general method LigBind-G for query ligand or general ligand binding residue prediction.

## Preparation
LigBind is built on Python3.
We recommend to use a virtual environment for the installation of GraphBind and its dependencies.
A virtual environment can be created and (de)activated as follows by using conda:

    # create
    $ conda create -n LigBind_env python=3.6
    # activate
    $ source activate GraphBind_env

When you want to quit the virtual environment, just:

    $ source deactivate

Download the source code of LigBind from GitHub:

    $ git clone https://github.com/yingx/LigBind.git

Download all of the trained models from [url](http://www.csbio.sjtu.edu.cn/bioinf/LigBind/files/LigBind_pth.tar.gz).

    $ tar zxvf LigBind_pth.tar.gz
    $ cp -r LigBind_pth LigBind/checkpoints

Instead, you can download models for specific ligand in [url](http://www.csbio.sjtu.edu.cn/bioinf/LigBind/download_checkpoints.html). Taking ZN as an example:
    
    $ mkdir LigBind/checkpoints/LigBind_pth
    $ tar zxvf ZN.tar.gz
    $ cp -r ZN LigBind/checkpoints/LigBind_pth


Install the dependencies as following:

    $ pip install biopython
    $ pip install torch==1.4.0
    $ pip install torch-scatter==2.0.3
    $ pip install torch-cluster==1.5.2
    $ pip install torch-sparse==0.5.1
    $ pip install torch-spline-conv==1.2.0
    $ pip install torch-geometric==1.7.0
    $ conda install -c conda-forge rdkit   # version 2020.09.1
    $ pip install git+https://github.com/bp-kelley/descriptastorus
     

Install the bioinformatics tools:

(1) Install DSSP (version: 2.0.4) for extracting SS (Secondary structure) profiles
    
    $ cd LigBind
    $ chmod +x ./dssp

(2) Install HHblits for extracting HMM profiles

To install HHblits (version: 3.3.0) and download uniclust30_2018_08 for HHblits, please refer to [hh-suite](https://github.com/soedinglab/hh-suite).
Set the absolute paths of HHblits and uniclust30_2018_08 databases in the script "./prediction.py".



## Usage

### Method 1. LigBind
Prediction for ligands in the ligand-specific datasets with fine-tuned models.

If the target ligand is included in [1159 ligands](https://github.com/YYingXia/LigBind/blob/main/dataset/ligand-specific_dataset.csv), you can select the ligand type and use ligand-specific LigBind for prediction:

You can search ligand ID by ligand name from 
[Ligand information](https://zhanggroup.org//BioLiP/ligand.html).

If you want to make prediction for mmCIF files instead of PDB files, please install [Open Babel](http://openbabel.org/) and convert mmCIF into PDB with: 

    $ obabel output/example/7bv2.cif -icif -opdb -O output/example/7bv2.pdb

(Option) Since the most time-consuming step of prediction pipeline is generating MSAs with HHblits and computing the secondary structures with DSSP, users can generate them before prediction:
    
    $ python extract_protein_features.py --querypath output/example  --protein_filename 7bv2.pdb --chainid A --method LigBind --ligands ADP,ATP --cpu 20

Make prediction:

    $ python prediction.py --querypath output/example  --protein_filename 7bv2.pdb --chainid A --method LigBind --ligands ADP,ATP --cpu 20

Results are saved in output/example/results. The predicted probabilities and binary results are saved in csv file. We provide an annotated pdb for per query ligand with b-factor replaced by an indicator of results, so that the results could be easily displayed in molecular viewers such as pymol.


### Method 2. LigBind-G
If the target ligand isn't included in [1159 ligands](https://github.com/YYingXia/LigBind/blob/main/dataset/ligand-specific_dataset.csv), you can input the ligand SMILES and use ligand-general LigBind-G for prediction:

    $ python prediction.py --querypath output/example  --protein_filename 7bv2.pdb --chainid A --method LigBind-G --ligand_filename ligand_smiles.txt --cpu 20

### Method 3. LigBind-G without ligand
If you want to predict general ligand-binding residues without ligand information, you can choose this method:

    $ python prediction.py --querypath output/example  --protein_filename 7bv2.pdb --chainid A --method LigBind-G-nolig --cpu 20

### Option: Construct your own dataset with low sequence identity.
First, [MMseqs2](https://github.com/soedinglab/MMseqs2) is applied for removing proteins of dataset1 with over 30% sequence identity to any proteins of dataset2:
    
    $ ./mmseqs easy-search dataset1 dataset2 out tmp --min-seq-id 0.3

Then, [CD-HIT](https://github.com/weizhongli/cdhit) (version: cd-hit-v4.8.1-2019-0228) is applied for reducing sequence identity of dataset1 itself. Following the CD-HIT user's guide, we reduce sequence identity of dataset1 to 30% with three steps:
    
    $ ./cd-hit -i dataset1 -o dataset1_80 -c 0.8 -n 5 -d 0 -M 16000 -T 16
    $ ./cd-hit -i dataset1_80 -o dataset1_60 -c 0.6 -n 4 -d 0 -M 16000 -T 16
    $ ./cd-hit -i dataset1_60 -o dataset1_30 -c 0.3



## License

Our project is under [GPL v3.0](https://github.com/YYingXia/LigBind/blob/main/LICENSE).
The parameters are made availabe under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). 

## Online service
Online retrieval service and benchmark datasets are in [here](http://www.csbio.sjtu.edu.cn/bioinf/LigBind/).
