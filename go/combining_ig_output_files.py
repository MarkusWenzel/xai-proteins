import os, json, re, glob
import pathlib
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

def combine_files(folder_path):
    """ Iterates over the files in a folder and calls combine on it """

    folder_path = folder_path
    logging.info(f"Executing file_combining() for all proteins in {folder_path}.")
        
    name_list_in_folder = [filename[len(folder_path):-4].split("_")[0] for filename in glob.glob(f"{folder_path}*0[]].csv")]
    # CSVs in data/clas_go_deepgoplus_temporalsplit/ig_outputs/ contain two underscores "_" and are "protein name" plus species, e.g. ABHD2_HUMAN_[10].csv, and therefore we need to split at the second underscore "_["
    name_list_in_folder = [filename[len(folder_path):-4].split("_[")[0] for filename in glob.glob(f"{folder_path}*0[]].csv")]
    
    print(f"{len(name_list_in_folder)} proteins found in {folder_path}")
    assert len(name_list_in_folder)!=0, f"No Proteins were found in {folder_path}, please check your path to have the format= .../"
    
    for protein_name in name_list_in_folder:
        try:
            combine(protein_name, folder_path)
        except pd.errors.EmptyDataError as err:
            print(f"{protein_name} has an empty layer file.")

def combine(protein_name:str, folder_path:str):
    """
    Combines the layer files of a protein into a single file.
    Args:
        protein_name (str): name of the protein to be combined.
        folder_path (str): Location where the files are stored.
    """
    
    protein_name = protein_name
    output_file_name = f"{folder_path}{protein_name}.csv"
    
    file_list = sorted(glob.glob(f"{folder_path}{protein_name}_*.csv"))

    assert len(file_list)==30, f"Only {len(file_list)}/30 files found for protein: {protein_name}."

    df = pd.read_csv(file_list[0])
    for file in file_list[1:]:
        try:
            df_layer = pd.read_csv(file)
        except pd.errors.EmptyDataError as err:
            print(f"{file} is empty.",err)
            raise(err)
        df = pd.concat([df,df_layer])
    df = df.reset_index(drop=True)

    df.to_csv(output_file_name, index=False)
    
    print(f"{protein_name} layers combined.")


def move_combined_files(from_dir:str="data/clas_go_deepgoplus_temporalsplit/ig_outputs/", to_dir:str="data/clas_go_deepgoplus_temporalsplit/ig_outputs_combined/", test_set_path:str="data/clas_go_deepgoplus_temporalsplit/test.json"):
    """Moves the combined files to a new directory

    Args:
        from_dir (str, optional): [description]. Defaults to "data/clas_go_deepgoplus_temporalsplit/ig_outputs/".
        to_dir (str, optional): [description]. Defaults to "data/clas_go_deepgoplus_temporalsplit/ig_outputs_combined/".
        test_set_path (str, optional): [description]. Defaults to "data/clas_go_deepgoplus_temporalsplit/test.json".
    """
    if not os.path.isdir(to_dir):
        os.system(f"mkdir {to_dir}")

    test_df = pd.read_json(test_set_path)
    test_name_list = test_df.name.values
    # If we would have selected a subset of all proteins from the test set:
    # test_name_list = [filename[len(from_dir):-4].split("_[")[0] for filename in glob.glob(f"{from_dir}*0[]].csv")] #MW
    
    for prot_name in test_name_list:
        from_path = f"{from_dir}{prot_name}.csv"
        to_path = f"{to_dir}{prot_name}.csv"
        os.system(f"mv {from_path} {to_path}")
    

def create_single_dataframe(folder_path:str, test_set_path:str):
    """
    Creates a single dataframe with columns sequence,name,label,relevances in 
    the parent folder of folder_path parameter as test_rel.json

    Args:
        folder_path (str): [description]
        test_set_path (str): [description]
    """
    parent_folder_path = str(pathlib.Path(folder_path).parent)

    data = []
    with open(test_set_path) as f:
        for line in f:
            data.append(json.loads(line))
    test_df=pd.DataFrame(data[0], columns=["sequence","name","label"])

    test_df['sequence'] = test_df.sequence.apply(lambda x: re.sub(r"[UZOB]", "X", x)) # replace rare aminoacids
    #test_df['sequence'] = test_df.sequence.apply(lambda x: [" ".join(list(x))]) # spacing the sequences # Not needed here.
    test_df['label'] = test_df['label'].apply(lambda x: np.array(list(x)).astype(int)) # label style for prepare_sample function
    
    # create new column and merge
    X = pd.DataFrame()
    # for prot_name in tqdm(test_df.name.values): # MW
    name_list_in_folder = [filename[len(folder_path):-4].split("_[")[0] for filename in glob.glob(f"{folder_path}*.csv")] #MW
    for prot_name in tqdm(name_list_in_folder): # MW
        df_prot = pd.read_csv(folder_path+prot_name+".csv")
        pro_rel_vec = df_prot.sum(axis=1).values # ndarray size=(480,)
        X = X.append({'name': prot_name, 'relevances': pro_rel_vec},ignore_index=True)    
    
    df_result = pd.merge(test_df, X, on="name")
    
    print("Length of df_result versus length of test_df: %i vs. %i" % (len(df_result), len(test_df)))
    #assert len(df_result)==len(test_df), "Some relevances are missing. please check ig_outputs_combine to have all the files."
    
    df_result.to_json(pathlib.Path(parent_folder_path, "test_rel.json"), orient='records')
    print(f"Created {parent_folder_path}/test_rel.json")
    

if __name__ == "__main__":
    """ Combines layer files in ig_outputs and move the combined ones to new dir. """
    
    combine_files(folder_path = "data/clas_go_deepgoplus_temporalsplit/ig_outputs/")
    move_combined_files(from_dir="data/clas_go_deepgoplus_temporalsplit/ig_outputs/", to_dir="data/clas_go_deepgoplus_temporalsplit/ig_outputs_combined/", test_set_path="data/clas_go_deepgoplus_temporalsplit/test.json")
    create_single_dataframe(folder_path = "data/clas_go_deepgoplus_temporalsplit/ig_outputs_combined/", test_set_path="data/clas_go_deepgoplus_temporalsplit/test.json")
