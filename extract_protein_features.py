import sys
import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

sys.path.append(os.path.abspath(''))
LigBind = os.path.abspath('.')
sys.path.append(LigBind)
import argparse
import sys
import os
import collections
import numpy as np
import torch
import sklearn
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data, Dataset
from model_architecture import MetaBind_MultiEdges, Pair_model_spe, Pair_model_gen
from torch import nn
from torch_geometric.data import DataLoader
import pickle
from descriptastorus import descriptors
from sklearn.cluster import MeanShift
from Bio.PDB.PDBParser import PDBParser
import subprocess
warnings.filterwarnings("ignore", category=UserWarning)

# Set the absolute paths of blast+, HHBlits and their databases in here.
HHblits = '/data/xiaying/tool/hh-suite/bin/hhblits'
HHblits_DB = '/data/zhangweixun/database/uniclust30_2018_08/uniclust30_2018_08'

# DSSP is contained in "scripts/dssp", and it should be given executable permission by commend line "chmod +x scripts/dssp".
DSSP = './dssp'

res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M', 'TRP': 'W','CYS': 'C',
            'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K','ARG': 'R'}


def cal_fasta(query_path):
    resid_res_dict = collections.OrderedDict()
    with open('{}/chain.dssp'.format(query_path),'r') as f:
        text = f.readlines()
    for i in range(25, len(text)):
        line = text[i]
        if line[13] not in res_dict.values() or line[9] == ' ':
            continue
        res_id = int(line[6:11])
        res = line[13]
        resid_res_dict[res_id] = res

    sequence = ''
    resid_list = []
    for res_id, res in resid_res_dict.items():
        sequence += res
        resid_list.append(res_id)
    with open('{}/chain.seq'.format(query_path),'w') as f:
        f.write('>chain\n{}'.format(sequence))
    return sequence, resid_res_dict, resid_list

def SaveChainPDB(protein_filename, chain_id, query_path):
    with open('{}/{}'.format(query_path, protein_filename), 'r') as f:
        text = f.readlines()
        pdb_text = []
        for line in text:
            if line[21] == chain_id:
                if line.startswith('ATOM'):
                        pdb_text.append(line)
                elif line.startswith('TER'):
                    pdb_text.append(line)
                    break
        with open('{}/chain.pdb'.format(query_path),'w') as f:
            f.writelines(pdb_text)
    return

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--querypath", dest="query_path", help="The path of query structure")
    parser.add_argument("--protein_filename", dest="protein_filename", help="The file name of the query structure which should be in PDB format.")
    parser.add_argument("--chainid", dest="chain_id", default = '', help="The query chain id(case sensitive). If there is only one chain in your query structure, you can leave it blank.")
    parser.add_argument("--method", dest='model_type', help='method types. You can choose from LigBind, LigBind-G, and LigBind-G-nolig.')
    parser.add_argument("--ligands", dest='ligands', help='Ligand types. Multiple ligands should be separated by commas. You can choose from DNA,RNA,CA,MG,MN,ATP,HEME.')
    parser.add_argument("--ligand_filename", dest='ligand_filename', help='SMILES of the unseen Ligand.')
    parser.add_argument("--cpu", dest="fea_num_threads",default = '1', help="The number of CPUs used for calculating PSSM and HMM profile.")
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()

    if args.query_path is None:
        print('ERROR: please input --querypath!')
        raise ValueError
    if args.protein_filename is None:
        print('ERROR: please input --protein_filename!')
        raise ValueError

    fea_num_threads = args.fea_num_threads
    chain_id = args.chain_id
    query_path = args.query_path.rstrip('/')
    protein_filename = args.protein_filename
    model_type = args.model_type

    if model_type == 'LigBind':
        if args.ligands is None:
            print('ERROR: please input --ligands for LigBind')
            raise ValueError
        ligand_list = args.ligands.split(',')
        ligand_filename = 'None'
    elif model_type == 'LigBind-G-nolig':
        ligand_list = []
        ligand_filename = 'None'
    elif model_type == 'LigBind-G':
        ligand_list = []
        ligand_filename = args.ligand_filename

        if ligand_filename== None:
            print('ERROR: Please input --ligand_filename')
            raise ValueError

        if not os.path.exists('{}/{}'.format(query_path, ligand_filename)):
            print('ERROR: Your query ligand structure "{}/{}" is not found!'.format(query_path, ligand_filename))
            raise ValueError
        else:
            with open('{}/{}'.format(query_path, ligand_filename), 'r') as f:
                smi = f.readlines()[0].strip()
            generator = descriptors.RDKit2DNormalized()
            mol_fea = generator.process(smi)
            if mol_fea is None:
                print('ERROR: Your query ligand SMILES {} is not correctly!'.format(smi))
    else:
        print('--method should be one of "LigBind", "LigBind-G" or "LigBind-G-nolig"')
        raise ValueError

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LigBind_threshold_csv = pd.read_csv('./checkpoints/LigBind_threshold.csv', keep_default_na=False)
    specific_ligand_list = LigBind_threshold_csv['ligand'].to_list()
    general_ligand_model = './checkpoints/LigBind-G.pth'

    # check input
    if not os.path.exists('{}/{}'.format(query_path,protein_filename)):
        print('ERROR: Your query protein structure "{}/{}" is not found!'.format(query_path,protein_filename))
        raise ValueError

    if model_type == 'LigBind':
        for ligand_i in ligand_list:
            if ligand_i not in specific_ligand_list:
                print('ERROR: ligand "{}" is not included in ligand-specific LigBind, please use ligand-general LigBind-G!'.format(ligand_i))
                raise ValueError

    p1 = PDBParser(PERMISSIVE=1)
    try:
        structure = p1.get_structure('chain', '{}/{}'.format(query_path,protein_filename))
    except:
        print('ERROR: The query protein structure "{}/{}" is not in correct PDB format, please check the structure!'.format(query_path,protein_filename))
        raise ValueError

    SaveChainPDB(protein_filename, chain_id, query_path)
    with open('{}/chain.pdb'.format(query_path),'r') as f:
        chain_atom_num = len(f.readlines())
        if chain_atom_num == 0:
            print('ERROR: Your query protein chain id "{}" is not in the uploaded protein structure, please check the chain ID!'.format(chain_id))
            raise ValueError

    DSSP_code = subprocess.call([DSSP, '-i', '{}/chain.pdb'.format(query_path), '-o', '{}/chain.dssp'.format(query_path)])
    if not os.path.exists('{}/chain.dssp'.format(query_path)):
        print("ERROR: The protein structure of chain {} is not complete, please check the structure!")
        raise ValueError

    sequence, resid_res_dict, resid_list = cal_fasta(query_path)

    HHblits_code = subprocess.call([HHblits, '-d', HHblits_DB, '-cpu', str(fea_num_threads),
                                    '-i', '{}/chain.seq'.format(query_path), '-ohhm', '{}/chain.hhm'.format(query_path)])

    print('done!')






