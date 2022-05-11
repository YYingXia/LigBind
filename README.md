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

Install the dependencies as following:

    $ pip install biopython
    $ pip install torch==1.4.0
    $ pip install torch-scatter==2.0.3
    $ pip install torch-cluster==1.5.2
    $ pip install torch-sparse==0.5.1
    $ pip install torch-spline-conv==1.2.0
    $ pip install torch-geometric==1.7.0
    $ conda install -c conda-forge rdkit
    $ pip install git+https://github.com/bp-kelley/descriptastorus


Install the bioinformatics tools:

(1) Install DSSP for extracting SS (Secondary structure) profiles
    
    $ cd LigBind
    $ chmod +x ./dssp

(2) Install HHblits for extracting HMM profiles
To install HHblits and download uniclust30_2018_08 for HHblits, please refer to hh-suite.
Set the absolute paths of HHblits and uniclust30_2018_08 databases in the script "./prediction.py".

## Usage
### Method 1. LigBind
Prediction for ligands in the ligand-specific datasets with fine-tuned models.

If the target ligand is included in [1159 ligands](https://github.com/yingx/LigBind/dataset/ligand-specific_dataset.csv), you can select the ligand type and use ligand-specific LigBind for prediction:

    $ python prediction.py --querypath output/example  --protein_filename 7bv2.pdb --chainid A --method LigBind --ligands ADP,ATP --cpu 20


### Method 2. LigBind-G
If the target ligand isn't included in [1159 ligands](https://github.com/yingx/LigBind/dataset/ligand-specific_dataset.csv), you can input the ligand SMILES and use ligand-general LigBind-G for prediction:

    $ python prediction.py --querypath output/example  --protein_filename 7bv2.pdb --chainid A --method LigBind-G --ligand_filename ligand_smiles.txt --cpu 20

### Method 3. LigBind-G without ligand
If you want to predict general ligand-binding residues without ligand information, you can choose this method:

    $ python prediction.py --querypath output/example  --protein_filename 7bv2.pdb --chainid A --method LigBind-G-nolig --cpu 20


## License

Our project is under 
[Apache License](https://github.com/yingx/LigBind/blob/master/LICENSE). 

## Online service
We also provide online retrieval service [here](http://www.csbio.sjtu.edu.cn/bioinf/LigBind/).
Our website follows a 'filter and refine' paradigm, which means it can provide more accurate result.
