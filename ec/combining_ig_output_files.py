import os
import glob
import pathlib
import logging
import pandas as pd
from tqdm import tqdm

def combine(protein_name:str, folder_path:str):
    """Combines the layer files of a protein into a single file.
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
    
def combine_files(folder_path):
    """Iterates over the files in a folder and calls combine on it"""

    folder_path = folder_path
    logging.info(f"Executing file_combining() for all proteins in {folder_path}.")
        
    name_list_in_folder = [filename[len(folder_path):-4].split("_")[0] for filename in glob.glob(f"{folder_path}*0[]].csv")]
    print(f"{len(name_list_in_folder)} proteins found in {folder_path}")
    assert len(name_list_in_folder)!=0, f"No Proteins were found in {folder_path}, please check your path to have the format= .../"
    
    for protein_name in name_list_in_folder:
        try:
            combine(protein_name, folder_path)
        except pd.errors.EmptyDataError as err:
            print(f"{protein_name} has an empty layer file.")

def create_single_dataframe(folder_path:str, test_set_path:str):
    """
    Creates a single dataframe with columns sequence,name,label,relevances in 
    the parent folder of folder_path parameter as test_rel.pkl

    Args:
        folder_path (str): [description]
        test_set_path (str): [description]
    """
    parent_folder_path = str(pathlib.Path(folder_path).parent)
    if os.path.isfile(parent_folder_path+"/"+"test_rel.pkl"):
        print(parent_folder_path+"/"+"test_rel.pkl"," exist already.")
        return None

    test_df = pd.read_json(test_set_path)

    # create new column and merge
    X = pd.DataFrame()
    for prot_name in tqdm(test_df.name.values):
        df_prot = pd.read_csv(folder_path+prot_name+".csv")
        pro_rel_vec = df_prot.sum(axis=1).values # ndarray size=(480,)
        X = X.append({'name': prot_name, 'relevances': pro_rel_vec},ignore_index=True)

    df_result = pd.merge(test_df, X, on="name")
    assert len(df_result)==len(test_df), "Some relevances are missing. please check ig_outputs_combine to have all the files."
    
    df_result.to_pickle(path = parent_folder_path+"/"+"test_rel.pkl", protocol=4)
    print(f"Created {parent_folder_path}/test_rel.pkl")

def move_combined_files(from_dir:str="data/ec50_level1/ig_outputs/", to_dir:str="data/ec50_level1/ig_outputs_combined/", test_set_path:str="data/ec50_level1/test.json"):
    """Moves the combined files to a new directory

    Args:
        from_dir (str, optional): [description]. Defaults to "data/ec50_level1/ig_outputs/".
        to_dir (str, optional): [description]. Defaults to "data/ec50_level1/ig_outputs_combined/".
        test_set_path (str, optional): [description]. Defaults to "data/ec50_level1/test.json".
    """
    if not os.path.isdir(to_dir):
        os.system(f"mkdir {to_dir}")

    test_df = pd.read_json(test_set_path)
    test_name_list = test_df.name.values
    for prot_name in test_name_list:
        from_path = f"{from_dir}{prot_name}.csv"
        to_path = f"{to_dir}{prot_name}.csv"
        os.system(f"mv {from_path} {to_path}")

if __name__ == "__main__":
    '''
    Combines layer files in ig_outputs and move the combined ones to new dir.
    '''
    combine_files(folder_path = "data/ec50_level1/ig_outputs/")
    move_combined_files(from_dir="data/ec50_level1/ig_outputs/", to_dir="data/ec50_level1/ig_outputs_combined/", test_set_path="data/ec50_level1/test.json")
    create_single_dataframe(folder_path = "data/ec50_level1/ig_outputs_combined/", test_set_path="data/ec50_level1/test.json")
