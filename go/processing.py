import os, re, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sites_helper
from prosite_parser import parse_prosite_full
from obtain_go_terms import Ontology

SELECTED_GO_TERM = json.load(open('parameters.json'))["SELECTED_GO_TERM"]

def processing():
    ''' Create active|binding|motif|transmembrane and prosite annotation datasets and test.json for each GO dataset'''

    # create the test.json from the .npy files
    test_path = "data/clas_go_deepgoplus_temporalsplit/"
    preprocess_dataset(test_path)

    # create data/active|binding|motif|transmembrane_site.pkl from .xml
    sites_helper.create_datasets("data/uniprot_sprot_2017_03.xml", parse_features = ["active site", "binding site", "short sequence motif", "transmembrane region"])
    
    # create prosite annotation datasets
    create_prosite_dataset(test_path)
        
    remove_annotation_sites_duplicates("data/active_site.pkl","data/active_site_clean.pkl")    
    remove_annotation_sites_duplicates("data/binding_site.pkl","data/binding_site_clean.pkl")
    remove_annotation_sites_duplicates("data/motif_site.pkl","data/motif_site_clean.pkl")
    remove_annotation_sites_duplicates("data/transmembrane_site.pkl","data/transmembrane_site_clean.pkl")    
    remove_annotation_sites_duplicates("data/prosite.pkl","data/prosite_clean.pkl")
    
    # create active|binding|motif|transmembrane_site.json which are all the test samples that have annotations within 1000 tokens.
    process_annotation_data(test_path, annotation_path="data/active_site_clean.pkl")
    process_annotation_data(test_path, annotation_path="data/binding_site_clean.pkl")
    process_annotation_data(test_path, annotation_path="data/motif_site_clean.pkl")
    process_annotation_data(test_path, annotation_path="data/transmembrane_site_clean.pkl")

    # create prosite_test.json which are all the test samples that have prosite annotations within 1000 tokens.
    process_prosite_data(test_path, annotation_path="data/prosite_clean.pkl")

def preprocess_dataset(test_path):
    """Convert *.npy files to *.json lines. For instance, preprocess_dataset("data/clas_go_deepgoplus_temporalsplit/")"""

    ids = np.load(Path(test_path, "ID.npy"),allow_pickle=True)
    tok = np.load(Path(test_path, "tok.npy"),allow_pickle=True)
    tok_itos = np.load(Path(test_path, "tok_itos.npy"),allow_pickle=True)
    label = np.load(Path(test_path, "label.npy"),allow_pickle=True)

    if test_path=='data/clas_go_deepgoplus_cafa/':
        test = np.load(Path(test_path, "test_IDs.npy"),allow_pickle=True) # CAFA3: three splits including test
    elif test_path=='data/clas_go_deepgoplus_temporalsplit/':
        test = np.load(Path(test_path, "val_IDs.npy"),allow_pickle=True) # Temporalsplit: test=val! only two splits available

    # Test split
    a = []
    for i in test:        
        b = {'sequence': "".join(tok_itos[tok[i]][1:]),
            'name': ids[i],
            'label': "".join([str(s) for s in label[i]])}
        a.append(b)
    df_test = pd.DataFrame(a)
    df_test.to_json(Path(test_path, "test.json"), orient='records')
    
def remove_annotation_sites_duplicates(annotation_site_filepath:str, output_filepath:str):
    """ From m with duplicates to n without duplicates """
    
    df_annotation = pd.read_pickle(annotation_site_filepath)
    df_annotation = df_annotation[["name","location","primary"]]
    unique_names = np.unique(df_annotation.name.values)
    D = pd.DataFrame(index=unique_names, columns=["name","location","primary"])
    for name in tqdm(unique_names):
        df_n = df_annotation[df_annotation.name == name]
        if len(df_n)==1:
            locs = df_n["location"].iloc[0]
        else: # combine duplicates
            locs = []
            for l in range(len(df_n)):
                rowx = df_n.iloc[l]                
                locs.append(rowx["location"][0])
        D["location"].loc[name] = locs
        D["name"].loc[name] = name
        D["primary"].loc[name] = df_n["primary"].iloc[0]

    D.reset_index(inplace=True,drop=True)
    D.to_pickle(output_filepath)

def process_annotation_data(test_path: str, annotation_path, max_seq_len=1000):
    """ 
    Create json file of intersection of proteins from the motif/active/binding/transmembrane and the test set. Merge on protein name.
    Parameter:
        test_path: folder where test.json is stored
        annotation_path: path to the motif_site.pkl (or active_site.pkl or binding_site.pkl) file with annotations extracted from xml
        max_seq_len: maximum sequence length to be put into the output for a single protein. 
    """
    
    data = []
    with open(f"{test_path}test.json") as f:
        for line in f:
            data.append(json.loads(line))
    data_test = pd.DataFrame(data[0], columns=["sequence","name","label"])

    # Merge data_annotation with data_test via protein name, adding all extra columns from data_test to data_annotation   
    data_annotation = pd.read_pickle(annotation_path)
    
    data_test_annotation = data_annotation.merge(data_test, on = 'name')
    
    # Keep only proteins where amino acid sequence of annotation dataset and actual test set exactly match
    data_test_annotation = data_test_annotation.loc[(data_test_annotation.sequence == data_test_annotation.primary)]
    
    # drop proteins with annotations above max seq length
    c = []
    for i in range(len(data_test_annotation)):
        prot = data_test_annotation.iloc[i]
        above_max_len = any([any(np.asarray(x).flatten() > max_seq_len) for x in prot['location']])
        if above_max_len:
            continue
        d = {'sequence': prot.primary[:max_seq_len], #-2],
            'name': prot['name'],
            'label': prot['label'],
            'location': prot['location']}
        c.append(d)

    print(pd.DataFrame(c))
    
    tag=re.split('_',str(Path(annotation_path).name))[0] # motif or active or binding
    with open(f'{test_path}{tag}_test.json', 'w') as f:
        json.dump(c, f)


def create_prosite_dataset(test_path):
    """ Create dataset of prosite annotations that are related to the selected GO-term and its children """

    # parse prosite annotations    
    if not Path("prosite_alignments.tar.gz").exists():
        os.system("wget https://ftp.expasy.org/databases/prosite/prosite_alignments.tar.gz") # (size  29M) or https://ftp.expasy.org/databases/prosite/old_releases/prosite2017_03.tar.bz2 (size 8.8M) analogue to SwissProt 2017 03 ?
    if not Path("prosite_alignments").exists():
        os.system("tar xvzf prosite_alignments.tar.gz")
        # tar xvjf prosite2017_03.tar.bz2 but format not compatible to prosite_parser
    df = parse_prosite_full(folder="prosite_alignments")
    df = df.rename({'protein_name': 'name'}, axis='columns')
    df["location"] = df.apply(lambda x: [[x["start_site"], x["end_site"]]], axis=1)
    df["primary"] = ""

    # Add information about corresponding GO-term to the prosite patterns
    df_prosite2go = pd.read_csv("http://current.geneontology.org/ontology/external2go/prosite2go", sep="; ", names=["Info","GO"], skiprows=6, engine="python")
    df_prosite2go["prosite_id"] = df_prosite2go["Info"].str.split(" ", expand=True)[0].str.split(":", expand=True)[1]
    df = df.merge(df_prosite2go)
    
    # obtain all children Gene-Ontology-terms (with 'is_a' child-parent-relation if from molecular_function namespace; with 'is_a' or 'part_of' child-parent-relation if from cellular_component namespace) from selected ancestor GO-term    
    if (test_path=="data/clas_go_deepgoplus_temporalsplit/"): go_path = Path("data-2016")
    elif (test_path=="data/clas_go_deepgoplus_cafa/"): go_path = Path("data-cafa")
    go_rels = Ontology(str(go_path/"go.obo"), with_rels=True)
    (term_set) = go_rels.get_term_set(SELECTED_GO_TERM)

    # filter for prosite patterns related to the selected GO-term (and its children)
    print(f"Filter for prosite patterns related to https://www.ebi.ac.uk/QuickGO/term/{SELECTED_GO_TERM} and children")
    df = df.loc[df.apply(lambda x: x["GO"] in term_set, axis=1)]
    df.to_pickle("data/prosite.pkl")
    
def process_prosite_data(test_path: str, annotation_path, max_seq_len=1000):
    """ 
    Create json file of intersection of proteins from prosite and test set. Merge on protein name.
    Parameter:
        test_path: folder where test.json is stored
        annotation_path: path to the prosite.pkl file with annotations extracted from xml
        max_seq_len: maximum sequence length to be put into the output for a single protein. 
    """
    data = []
    with open(f"{test_path}test.json") as f:
        for line in f:
            data.append(json.loads(line))
    data_test = pd.DataFrame(data[0], columns=["sequence","name","label"])

    # Merge data_annotation with data_test via name, adding all extra columns from data_test to data_annotation   
    data_annotation = pd.read_pickle(annotation_path)
    data_test_annotation = data_annotation.merge(data_test, on = "name")
    
    # drop proteins with annotations above max seq length
    c = []
    for i in range(len(data_test_annotation)):
        prot = data_test_annotation.iloc[i]
        above_max_len = any([any(np.asarray(x).flatten() > max_seq_len) for x in prot['location']])
        if above_max_len:
            continue
        d = {'sequence': prot.sequence[:max_seq_len], #-2],
            'name': prot['name'],
            'label': prot['label'],
            'location': prot['location']}
        c.append(d)

    print(pd.DataFrame(c))
    
    tag=re.split('_',str(Path(annotation_path).name))[0] # tag = prosite
    with open(f'{test_path}{tag}_test.json', 'w') as f:
        json.dump(c, f)

if __name__ == "__main__":
    processing()
