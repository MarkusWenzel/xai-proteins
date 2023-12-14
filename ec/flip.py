# attributionfile=./data/ec50_level1/ig_outputs/embedding_attribution.csv
# python flip.py $attributionfile

import sys, os, re, glob, json, csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def processing(attributionfile):
    """ Create active|binding|motif|transmembrane datasets and iterate through the class,level folders in data/ and create test,train,valid.json files from their .npy files """

    # create the test.json from the .npy files
    test_path = "/data/"
    preprocess_dataset(test_path)
                    
    # create test files where n most relevant residues are flipped (test_n.json).
    for n_flipped in [1,2,4,8,16,32,64]:
        create_flipped_data(test_path, attributionfile, n_flipped)

def preprocess_dataset(test_path, max_len=1000):
    """ Convert *.npy files to *.json lines. For instance, preprocess_dataset('data/ec50_level1/') """

    ids = np.load(Path(test_path, "ID.npy"),allow_pickle=True)
    tok = np.load(Path(test_path, "tok.npy"),allow_pickle=True)
    tok_itos = np.load(Path(test_path, "tok_itos.npy"),allow_pickle=True)
    label = np.load(Path(test_path, "label.npy"),allow_pickle=True)
    test = np.load(Path(test_path, "test_IDs.npy"),allow_pickle=True)
    a = []
    for i in test:
        b = {'sequence': "".join(tok_itos[tok[i]][1:(max_len-2+1)]),
            'name': ids[i],
            'label': float(label[i])}
        a.append(b)
    df_test = pd.DataFrame(a)
    df_test.to_json(Path(test_path, "test.json"), orient='records')

def create_flipped_data(test_path: str, attributionfile, n_flipped):
    """
    Parameter:
        test_path: Is the folder where the test.json is stored
        attribution_path: path to the embedding_attribution.csv with attributions to most relevant residues
    """
        
    data_test = pd.read_json(f"{test_path}test.json")
    
    with open(attributionfile, 'r') as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=' ', quotechar='|')
        header = next(csv_reader) # ['protein_name', 'protein_label', 'attribution']
        data_test_flipped_rel=[]
        data_test_flipped_rnd=[]
        for row in tqdm(csv_reader):
            b = row # list of strings
            prot_name = b[0] # str
            if not(prot_name in data_test.name.values):
                continue
            prot_label = int(data_test.loc[data_test.name == prot_name]["label"])
            prot_seq = data_test.loc[data_test.name == prot_name]["sequence"].iloc[0]
            relevances_str_list = b[2] # str '[0.0,..,0.0]'
            relevances_list_of_strings = relevances_str_list[1:-1].split(',')
            relevances = [float(x) for x in relevances_list_of_strings]
            # not 1st and last, then sequence matches
            relevances = np.array(relevances[1:-1])
            
            # sort residues according to attribution and replace most relevant residues with alanine            
            flip=np.full((len(relevances),), False, dtype=bool)
            # np.argsort sorts ascending, we need to invert sign for descending order
            flip[(-relevances).argsort()[:n_flipped]]=True
                        
            prot_seq_flipped_rel = ''.join(['A' if f else r for r,f in zip(prot_seq, flip)]) # A=Alanine
            data_test_flipped_rel.append({'sequence': prot_seq_flipped_rel, 'name': prot_name, 'label': float(prot_label)})

            # flip random residues
            for i in range(0,10): # save several random repetitions for this random baseline
                np.random.shuffle(relevances)            
                flip=np.full((len(relevances),), False, dtype=bool)
                # np.argsort sorts ascending, we need to invert sign for descending order
                flip[(-relevances).argsort()[:n_flipped]]=True
                prot_seq_flipped_rnd = ''.join(['A' if f else r for r,f in zip(prot_seq, flip)]) # A=Alanine            
                data_test_flipped_rnd.append({'sequence': prot_seq_flipped_rnd, 'name': prot_name, 'label': float(prot_label)})
                
            
        data_test_flipped_rel = pd.DataFrame(data_test_flipped_rel)
        data_test_flipped_rel.to_json(Path(test_path, f"test_flip_rel_{n_flipped}.json"), orient='records')

        data_test_flipped_rnd = pd.DataFrame(data_test_flipped_rnd)
        data_test_flipped_rnd.to_json(Path(test_path, f"test_flip_rnd_{n_flipped}.json"), orient='records')
        
if __name__ == "__main__":
    attributionfile=sys.argv[1]
    processing(attributionfile)
