import sys
import os
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

sys.path.append(os.path.abspath(''))
LigBind = os.path.abspath('.')
sys.path.append(LigBind)
import argparse
import sys
sys.path.append('../')
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
from tqdm import tqdm
from rdkit import Chem
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

Relative_atomic_mass = {'H': 1, 'C': 12, 'O': 16, 'N': 14, 'S': 32, 'FE': 56, 'P': 31, 'BR': 80, 'F': 19, 'CO': 59,
                        'V': 51,
                        'I': 127, 'CL': 35.5, 'CA': 40, 'B': 10.8, 'ZN': 65.5, 'MG': 24.3, 'NA': 23, 'HG': 200.6,
                        'MN': 55,
                        'K': 39.1, 'AP': 31, 'AC': 227, 'AL': 27, 'W': 183.9, 'SE': 79, 'NI': 58.7}



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

def cal_HMM(query_path, sequence):
    with open('{}/chain.hhm'.format(query_path), 'r') as fin:
        fin_data = fin.readlines()
    hhm_begin_line = 0
    hhm_end_line = 0
    for i in range(len(fin_data)):
        if '#' in fin_data[i]:
            hhm_begin_line = i+5
        elif '//' in fin_data[i]:
            hhm_end_line = i
    feature = np.zeros([int((hhm_end_line-hhm_begin_line)/3),30])
    axis_x = 0
    sequence_hhm = ''
    for i in range(hhm_begin_line,hhm_end_line,3):
        sequence_hhm+=fin_data[i].split()[0]
        line1 = fin_data[i].split()[2:-1]
        line2 = fin_data[i+1].split()
        axis_y = 0
        for j in line1:
            if j == '*':
                feature[axis_x][axis_y]=9999/10000.0
            else:
                feature[axis_x][axis_y]=float(j)/10000.0
            axis_y+=1
        for j in line2:
            if j == '*':
                feature[axis_x][axis_y]=9999/10000.0
            else:
                feature[axis_x][axis_y]=float(j)/10000.0
            axis_y+=1
        axis_x+=1
    if sequence_hhm != sequence:
        print('ERROR: the length of HMM features is not equal to sequence length!')
        raise ValueError
    feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
    return feature

def cal_DSSP(query_path, sequence):

    maxASA = {'G':188,'A':198,'V':220,'I':233,'L':304,'F':272,'P':203,'M':262,'W':317,'C':201,
              'S':234,'T':215,'N':254,'Q':259,'Y':304,'H':258,'D':236,'E':262,'K':317,'R':319}
    map_ss_8 = {' ':[1,0,0,0,0,0,0,0],'S':[0,1,0,0,0,0,0,0],'T':[0,0,1,0,0,0,0,0],'H':[0,0,0,1,0,0,0,0],
                'G':[0,0,0,0,1,0,0,0],'I':[0,0,0,0,0,1,0,0],'E':[0,0,0,0,0,0,1,0],'B':[0,0,0,0,0,0,0,1]}
    with open('{}/chain.dssp'.format(query_path), 'r') as fin:
        fin_data = fin.readlines()
    dssp_sequence = ''
    seq_feature = []
    for i in range(25, len(fin_data)):
        line = fin_data[i]
        if line[13] not in maxASA.keys() or line[9]==' ':
            continue
        res_id = float(line[5:10])
        feature = np.zeros([14])
        feature[:8] = map_ss_8[line[16]]
        feature[8] = min(float(line[35:38]) / maxASA[line[13]], 1)
        feature[9] = (float(line[85:91]) + 1) / 2
        feature[10] = min(1, float(line[91:97]) / 180)
        feature[11] = min(1, (float(line[97:103]) + 180) / 360)
        feature[12] = min(1, (float(line[103:109]) + 180) / 360)
        feature[13] = min(1, (float(line[109:115]) + 180) / 360)
        seq_feature.append(feature)
        dssp_sequence += line[13]
    if dssp_sequence != sequence:
        print('ERROR: the length of dssp features is not equal to sequence length!')
        raise ValueError
    seq_feature = np.array(seq_feature)
    return seq_feature

def cal_Psepos(query_path):
    resid_res_dict = collections.OrderedDict()
    with open('{}/chain.dssp'.format(query_path), 'r') as f:
        text = f.readlines()
    for i in range(25, len(text)):
        line = text[i]
        if line[13] not in res_dict.values() or line[9] == ' ':
            continue
        res_id = line[6:11].strip()
        res = line[13]
        resid_res_dict[res_id] = res

    res_centroid = []
    res_psepos = {}
    res_mass = {}
    with open('{}/chain.pdb'.format(query_path),'r') as f:
        text = f.readlines()
    for line in text:
        if line.startswith('ATOM'):
            res_pdb_id = line[22:27].strip()
            if res_pdb_id not in res_psepos.keys():
                res_psepos[res_pdb_id] = []
                res_mass[res_pdb_id] = []
            atom_type = line[76:78].strip()
            if atom_type in Relative_atomic_mass.keys():
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                res_psepos[res_pdb_id].append(xyz)
                res_mass[res_pdb_id].append(Relative_atomic_mass[atom_type])

    for res_id in resid_res_dict.keys():
        xyz = np.array(res_psepos[res_id])
        masses = np.array(res_mass[res_id]).reshape(-1,1)
        centroid = np.sum(masses*xyz,axis=0)/np.sum(masses)
        res_centroid.append(centroid)
    res_centroid = np.array(res_centroid)

    return res_centroid

class NeighResidue3DPoint_pred(InMemoryDataset):
    def __init__(self, root, fea, pos):
        self.pos = pos
        self.fea = fea
        self.dist = 15
        super(NeighResidue3DPoint_pred, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_dir + '/chain_data.pt')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'chain_data.pt'

    def _download(self):
        pass

    def process(self):
        with open('{}/chain.seq'.format(self.root),'r') as f:
            sequence = f.readlines()[1].strip()
        data_list = []
        for i in range(len(sequence)):
            res_psepos = self.pos[i]
            res_dist = np.sqrt(np.sum((self.pos - res_psepos) ** 2, axis=1))
            neigh_index = np.where(res_dist < self.dist)[0]
            res_pos = self.pos[neigh_index] - res_psepos
            res_feas = self.fea[neigh_index]

            node_feas = torch.tensor(res_feas, dtype=torch.float32)
            node_pos = torch.tensor(res_pos, dtype=torch.float32)

            data = Data(x=node_feas, pos=node_pos)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save([data, slices], self.processed_dir + '/chain_data.pt')

def predict(device,model, model_path_list, test_data,mol_emb):
    Numres = test_data.slices['x'].shape[0]-1
    test_batchsize = 128
    while True:
        if (Numres - 1) % test_batchsize == 0:
            test_batchsize += 1
            print('test_batchsize=', test_batchsize)
        else:
            break

    all_test_probs = []

    for model_path in model_path_list:
        if not os.path.exists(model_path):
            print('Can not find ', model_path)
            raise ValueError
        model_params = torch.load(model_path)
        model.load_state_dict(model_params)

        model.to(device)
        model.eval()

        test_dataloader = DataLoader(test_data, batch_size=test_batchsize, shuffle=False, num_workers=0,pin_memory=True)

        test_probs = []
        with torch.no_grad():
            for ii, data in enumerate(test_dataloader):
                data = data.to(device)
                logit = model(data, mol_emb)
                score = torch.sigmoid(logit.float())
                test_probs += score.tolist()
        test_probs = np.array(test_probs)
        all_test_probs.append(test_probs.reshape(-1,1))

    avg_test_probs = np.concatenate(all_test_probs,axis=1)
    avg_test_probs = np.average(avg_test_probs,axis=1)

    return avg_test_probs

def ligand_specific_model(ligand_list):
    ligand_model_dict = {}
    for ligand in ligand_list:
        ligand_model_dict[ligand] = []
        files = os.listdir('./checkpoints/LigBind/{}'.format(ligand))
        for file in files:
            if file.endswith('.pth'):
                ligand_model_dict[ligand].append('./checkpoints/LigBind/{}/{}'.format(ligand, file))
    return ligand_model_dict

def Bfactor2score(input_pth, pdbid, resid_res_dict, score, pred_site=None, ligand=None):
    score = score.tolist()
    map_res_score = {}

    for i, resid in enumerate(resid_res_dict.keys()):
        map_res_score[int(resid)] = score[i]

    with open("{}/{}.pdb".format(input_pth, pdbid),'r') as f:
        text = f.readlines()
    with open('{}/results/{}_{}_predscore.pdb'.format(input_pth, pdbid, ligand),'w') as f:
        print('{}/results/{}_{}_predscore.pdb'.format(input_pth, pdbid, ligand))
        for i in range(len(text)):
            if not text[i].startswith('ATOM'):
                f.write(text[i])
                continue
            res_id = int(text[i][22:26])

            if res_id not in map_res_score.keys():
                score = 0
            else:
                score = map_res_score[res_id]

            line = text[i][:60]
            line +='  {:.2f}'.format(score)
            line += text[i][66:]
            f.write(line)

    if pred_site is not None:
        map_res_site = {}

        for i, resid in enumerate(resid_res_dict.keys()):
            map_res_site[int(resid)] = pred_site[i]

        with open('{}/results/{}_{}_predsite.pdb'.format(input_pth, pdbid, ligand), 'w') as f:
            for i in range(len(text)):
                if not text[i].startswith('ATOM'):
                    f.write(text[i])
                    continue
                res_id = int(text[i][22:26])

                if res_id not in map_res_site.keys():
                    site = 0
                else:
                    site = map_res_site[res_id]

                line = text[i][:60]
                line += '  {:.2f}'.format(site)
                line += text[i][66:]
                f.write(line)

    return

def MeanShift_clusting(pos_position, T=None):
    if T is None:
        ms = MeanShift()
    else:
        ms = MeanShift(bandwidth=T)
    ms.fit(pos_position)
    labels_pred = ms.labels_
    return labels_pred

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

    print('1. data process...')
    sequence, resid_res_dict, resid_list = cal_fasta(query_path)

    HHblits_code = subprocess.call([HHblits, '-d', HHblits_DB, '-cpu', str(fea_num_threads),
                                    '-i', '{}/chain.seq'.format(query_path), '-ohhm', '{}/chain.hhm'.format(query_path)])

    hhm_fea = cal_HMM(query_path, sequence)
    dssp_fea = cal_DSSP(query_path, sequence)
    fea = np.concatenate([hhm_fea, dssp_fea], axis=1)
    psepos = cal_Psepos(query_path)
    test_data = NeighResidue3DPoint_pred(query_path, fea, psepos)

    if not os.path.exists('{}/results'.format(query_path)):
        os.mkdir('{}/results'.format(query_path))

    if model_type == 'LigBind':
        print('2. run LigBind...')
        protein_model = MetaBind_MultiEdges(fea_ext_fixed=False, gru_steps=4, x_ind=45, edge_ind=2,
                                            x_hs=128, e_hs=128, u_hs=128, dropratio=0.5, bias=True,
                                            edge_method='radius', r=10,
                                            aggr='mean', dist=15, max_nn=100, stack_method='GRU', apply_edgeattr=True,
                                            apply_nodeposemb=True)
        model = Pair_model_spe(protein_model, 128 * (4 + 1), 200, 128, 10)

        specific_ligand_model_threshold_dict = dict(zip(LigBind_threshold_csv['ligand'].to_list(), LigBind_threshold_csv['threshold'].to_list()))
        specific_ligand_model_dict = ligand_specific_model(specific_ligand_list)

        with open('./checkpoints/ligand_emb.pkl','rb') as f:
            ligand_emb_dict = pickle.load(f)
        MeanShift_threshold_csv = pd.read_csv('./checkpoints/MeanShift_threshold.csv', keep_default_na=False)
        meanshift_threshold_dict = dict(zip(MeanShift_threshold_csv['Ligand'].tolist(), MeanShift_threshold_csv['Threshold'].tolist()))

        for ligand in ligand_list:
            print(ligand)
            threshold = specific_ligand_model_threshold_dict[ligand]
            model_path_list = specific_ligand_model_dict[ligand]
            mol_emb = torch.from_numpy(ligand_emb_dict[ligand]).float().to(device)
            pred = predict(device, model, model_path_list, test_data, mol_emb)
            pred_bi = np.abs(np.ceil(pred - threshold))
            pred_site = np.zeros(len(pred_bi))
            if sum(pred_bi) > 0:
                pred_site[pred_bi==1] = MeanShift_clusting(psepos[pred_bi==1], meanshift_threshold_dict[ligand]) + 1
            site_num = len(np.unique(pred_site))-1
            print('LigsBind: ligand={} pos_res_num={} site_num={}'.format(ligand, sum(pred_bi), site_num))
            result_df = pd.DataFrame({'Residue_ID':resid_list, 'Residue':list(sequence),'Probability':pred,'Binary':pred_bi,'Site':pred_site})
            result_df.to_csv('{}/results/{}-binding_result(LigBind).csv'.format(query_path, ligand))
            print('Results are saved in {}/results/{}-binding_result(LigBind).csv'.format(query_path, ligand))



    elif model_type == 'LigBind-G':
        print('2. run LigBind-G...')

        # mol = Chem.SDMolSupplier('{}/{}'.format(query_path, ligand_filename))[0]
        # smi = Chem.MolToSmiles(mol)
        with open('{}/{}'.format(query_path, ligand_filename), 'r') as f:
            smi = f.readlines()[0].strip()
        generator = descriptors.RDKit2DNormalized()
        mol_fea = np.array(generator.process(smi)[1:])
        mol_fea[np.isnan(mol_fea)] = 0
        mol_emb = torch.from_numpy(mol_fea).float().to(device)

        protein_model = MetaBind_MultiEdges(fea_ext_fixed=False, gru_steps=4, x_ind=45, edge_ind=2,
                                            x_hs=128, e_hs=128, u_hs=128, dropratio=0.5, bias=True,
                                            edge_method='radius', r=10,
                                            aggr='mean', dist=15, max_nn=100, stack_method='GRU', apply_edgeattr=True,
                                            apply_nodeposemb=True)
        model = Pair_model_gen(protein_model, 128 * (4 + 1), 200, 128)
        model_path_list = [general_ligand_model]
        pred = predict(device, model, model_path_list, test_data, mol_emb)
        result_df = pd.DataFrame(
            {'Residue_ID': resid_list, 'Residue': list(sequence), 'Probability': pred})
        result_df.to_csv('{}/results/binding_result(LigBind-G).csv'.format(query_path))
        print('Results are saved in {}/results/binding_result(LigBind-G).csv'.format(query_path))

    elif model_type == 'LigBind-G-nolig':
        with open('./checkpoints/ligand_emb.pkl', 'rb') as f:
            ligand_emb_dict = pickle.load(f)
        avg_ligand_emb = list(ligand_emb_dict.values())
        avg_ligand_emb = np.array(avg_ligand_emb)
        avg_ligand_emb = np.average(avg_ligand_emb, axis=0)

        avg_ligand_emb[np.isnan(avg_ligand_emb)] = 0
        mol_emb = torch.from_numpy(avg_ligand_emb).float().to(device)

        protein_model = MetaBind_MultiEdges(fea_ext_fixed=False, gru_steps=4, x_ind=45, edge_ind=2,
                                            x_hs=128, e_hs=128, u_hs=128, dropratio=0.5, bias=True,
                                            edge_method='radius', r=10,
                                            aggr='mean', dist=15, max_nn=100, stack_method='GRU', apply_edgeattr=True,
                                            apply_nodeposemb=True)
        model = Pair_model_gen(protein_model, 128 * (4 + 1), 200, 128)
        model_path_list = [general_ligand_model]
        pred = predict(device, model, model_path_list, test_data, mol_emb)
        result_df = pd.DataFrame(
            {'Residue_ID': resid_list, 'Residue': list(sequence), 'Probability': pred})
        result_df.to_csv('{}/results/binding_result(LigBind-G-nolig).csv'.format(query_path))
        print('Results are saved in {}/results/binding_result(LigBind-G-nolig).csv'.format(query_path))


    print('Done!')






