import os, re, glob, json
import numpy as np
import pandas as pd
from tqdm import tqdm
import sites_helper
from pathlib import Path

def processing():
    """ Create active|binding|motif|transmembrane datasets and iterate through the class,level folders in data/ and create test,train,valid.json files from their .npy files """

    """
    # create folder for models
    Path("models/ec50_level0/").mkdir(parents=True, exist_ok=True)
    Path("models/ec50_level1/").mkdir(parents=True, exist_ok=True)
    Path("models/ec50_level2/").mkdir(parents=True, exist_ok=True)
    Path("models/ec40_level0/").mkdir(parents=True, exist_ok=True)
    Path("models/ec40_level1/").mkdir(parents=True, exist_ok=True)
    Path("models/ec40_level2/").mkdir(parents=True, exist_ok=True)
    """
    # create data/motif/binding/active_site.pkl from .xml
    sites_helper.create_datasets("data/uniprot_sprot_2017_03.xml", parse_features=["active site", "binding site", "short sequence motif", "transmembrane region"])
    remove_annotation_sites_duplicates("data/active_site.pkl","data/active_site_clean.pkl")
    remove_annotation_sites_duplicates("data/binding_site.pkl","data/binding_site_clean.pkl")
    remove_annotation_sites_duplicates("data/motif_site.pkl","data/motif_site_clean.pkl")
    remove_annotation_sites_duplicates("data/transmembrane_site.pkl","data/transmembrane_site_clean.pkl")
    
    # create the test.json and train.json from the .npy files
    folders = sorted(glob.glob("data/ec*/"))
    for test_path in folders:
        preprocess_dataset(test_path)
                    
        # create motif_test.json which are all the test samples that have motifs within 1000 tokens.
        process_annotation_data(test_path, annotation_path="data/active_site_clean.pkl")
        process_annotation_data(test_path, annotation_path="data/binding_site_clean.pkl")
        process_annotation_data(test_path, annotation_path="data/motif_site_clean.pkl")
        process_annotation_data(test_path, annotation_path="data/transmembrane_site_clean.pkl")

def remove_annotation_sites_duplicates(annotation_site_filepath:str, output_filepath:str):
    """ From n with duplicates to m without duplicates """
    df_annotation = pd.read_pickle(annotation_site_filepath) 
    unique_names = np.unique(df_annotation.name.values)

    D = pd.DataFrame()
    for name in tqdm(unique_names):
        df_n = df_annotation[df_annotation.name == name]
        if len(df_n)==1:
            df_n.iloc[0]["location"] = df_n.iloc[0]["location"]
        else:
            # combine duplicates
            locs = []
            desc = []
            for l in range(len(df_n)):
                rowx = df_n.iloc[l]
                locs.append(rowx["location"][0])
                desc.append(rowx["description"])
            df_n.iloc[0]["location"] = locs
            df_n.iloc[0]["description"] = desc
        D = D.append(df_n.iloc[0])

    # add label column
    D["label"] = D["EC-Label"].apply(lambda x:int(x[0][0])-1)
    D = D[["name","EC-Label","label","type","description", "location","evidence","primary"]]
    D.to_pickle(output_filepath)

def preprocess_dataset(test_path, max_len=1000):
    """ Convert *.npy files to *.json lines. For instance, preprocess_dataset('data/ec50_level1/') """

    ids = np.load(Path(test_path, "ID.npy"),allow_pickle=True)
    tok = np.load(Path(test_path, "tok.npy"),allow_pickle=True)
    tok_itos = np.load(Path(test_path, "tok_itos.npy"),allow_pickle=True)
    label = np.load(Path(test_path, "label.npy"),allow_pickle=True)

    test = np.load(Path(test_path, "test_IDs.npy"),allow_pickle=True)
    
    # Test split
    a = []
    for i in test:
        b = {'sequence': "".join(tok_itos[tok[i]][1:(max_len-2+1)]),
            'name': ids[i],
            'label': float(label[i])}
        a.append(b)
    df_test = pd.DataFrame(a)
    df_test.to_json(Path(test_path, "test.json"), orient='records')

def process_annotation_data(test_path: str, annotation_path, max_seq_len=1000):
    """
    Create a json file of the intersection of proteins from the motif/active/binding and the test set.

    Parameter:
        test_path: Is the folder where the test.json is stored
        annotation_path: path to the motif_site.pkl (or active_site.pkl or binding_site.pkl) file with annotations extracted from xml
        max_seq_len: The maximum sequence length to be put into the output for a single protein.
    """
    
    data_test = pd.read_json(f"{test_path}test.json")
    data_annotation = pd.read_pickle(annotation_path)
        
    test_names = data_test['name'].to_list()
    annotation_name_list = data_annotation['name'].to_list()
    test_annotation = [annotation_name in test_names for annotation_name in annotation_name_list]
    data_test_annotation = data_annotation.loc[test_annotation]

    # drop the proteins with annotations above max seq length
    c = []
    for i in range(len(data_test_annotation)):
        prot = data_test_annotation.iloc[i]
        above_max_len = any(
            [any(np.asarray(x).flatten() > max_seq_len) for x in prot['location']])
        if above_max_len:
            continue
        d = {'sequence': prot.primary[:max_seq_len-2],
            'name': prot['name'],
            'label': float(prot['label']),
            'location': prot['location']}
        c.append(d)

    tag=re.split('_',str(Path(annotation_path).name))[0] # motif or active or binding
    with open(f'{test_path}{tag}_test.json', 'w') as f:
        json.dump(c, f)

if __name__ == "__main__":
    processing()
