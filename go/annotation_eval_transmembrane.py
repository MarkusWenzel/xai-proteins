import os, re, json, csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.stats import normaltest,wilcoxon
from statsmodels.stats.multitest import multipletests
from scipy.stats import pointbiserialr
from shutil import copyfile

SELECTED_GO_TERM = json.load(open('parameters.json'))["SELECTED_GO_TERM"]
GO_TERMS = {"GO:0003824":"catalytic-activity", "GO:0016020":"membrane", "GO:0005488":"binding"}

def get_ground_truth_vec(gt_length, loc) -> np.array:
    '''
    Create a [0,1] vector of the size of the relevance (CSL+Seq+SEP) with the annotation locations as 1's and the rest as 0's

    Parameter:
        loc : protein.location i.e. [[10,12]]
        prim: protein.primary  i.e. "ABCDEFGHIJKL"
    Output:
        ground_truth_array i.e. array([0,0,0,0,0,0,0,0,0,0,1,1,1,0]) accounting for CLS,Sep token
    '''
    loc_dims = len(loc)
    arr = np.zeros(gt_length)  # 0's array of size protein sequence
    for dim in range(loc_dims):

        # original locations array count from 1 therefore start-1
        start, stop = loc[dim][0], loc[dim][-1]
        start -= 1  # stop value can stay the same because of slicing exclusion
        arr[start:stop] = 1

    annot_location_arr = np.append(arr, 0)
    # adding zero to account for CLS,SEP token
    annot_location_arr = np.append(0, annot_location_arr)

    return annot_location_arr


def embedding_calc(df_test_annot, data_path, tag:str):
    """ Correlation of annotation and relevances on embedding level """

    embedding_path = data_path/"ig_outputs/embedding_attribution.csv"    
    scores = {"name":[], "label":[], "pbr":[]}
    
    with open(embedding_path, 'r') as read_obj:
        csv_reader = csv.reader(read_obj, delimiter=' ', quotechar='|')
        header = next(csv_reader) # ['protein_name', 'protein_label', 'attribution']
        for row in tqdm(csv_reader):
            b = row # list of strings
            prot_name = b[0] # str
            if not(prot_name in df_test_annot.name.values):
                continue

            scores["name"].append(prot_name)
            prot_label = int(df_test_annot.loc[df_test_annot.name == prot_name]["label"])
            scores["label"].append(prot_label)            
            
            relevances_str_list = b[2] # str '[0.0,..,0.0]'
            relevances_list_of_strings = relevances_str_list[1:-1].split(',')
            relevances = [float(x) for x in relevances_list_of_strings]

            prot_info = df_test_annot[df_test_annot.name == prot_name]                                   
            locations = prot_info.location.iloc[0]
            # Point biserial correlation a.k.a. Pearson correlation
            annot_location_arr = get_ground_truth_vec(len(relevances)-2, locations)

            # Transmembrane, hydrophobic, positive charge
            # `hydrophobic (F, M, W, I, V, L, P, A)' [https://doi.org/10.1529/biophysj.106.098004]
            # `positively charged residues (lysine and arginine)' (K, R) [https://doi.org/10.1186/s12915-017-0404-4 ]
            if tag=="transmembrane_hydrophobic":                
                seq = prot_info.sequence.iloc[0]            
                hydrophobic_pos = [pos for pos, char in enumerate(seq) if char in ['F', 'M', 'W', 'I', 'V', 'L', 'P', 'A']]
                hydrophobic = np.zeros_like(annot_location_arr)
                hydrophobic[hydrophobic_pos] = 1.0
                print(f"{tag} share: {np.sum(np.multiply(annot_location_arr, hydrophobic))/np.sum(annot_location_arr)}")
                annot_location_arr = np.multiply(annot_location_arr, hydrophobic)                
            if tag=="transmembrane_positive":                
                seq = prot_info.sequence.iloc[0]            
                positive_charged_pos = [pos for pos, char in enumerate(seq) if char in ['K', 'R']]
                positive_charged = np.zeros_like(annot_location_arr)
                positive_charged[positive_charged_pos] = 1.0
                print(f"{tag} share: {np.sum(np.multiply(annot_location_arr, positive_charged))/np.sum(annot_location_arr)}")
                annot_location_arr = np.multiply(annot_location_arr, positive_charged)
            if tag=="hydrophobic":                
                seq = prot_info.sequence.iloc[0]            
                hydrophobic_pos = [pos for pos, char in enumerate(seq) if char in ['F', 'M', 'W', 'I', 'V', 'L', 'P', 'A']]
                hydrophobic = np.zeros_like(annot_location_arr)
                hydrophobic[hydrophobic_pos] = 1.0                
                annot_location_arr = hydrophobic
            if tag=="positive":                
                seq = prot_info.sequence.iloc[0]            
                positive_charged_pos = [pos for pos, char in enumerate(seq) if char in ['K', 'R']]
                positive_charged = np.zeros_like(annot_location_arr)
                positive_charged[positive_charged_pos] = 1.0
                annot_location_arr = positive_charged
            if tag=="hydrophobic_and_positive_test":                
                seq = prot_info.sequence.iloc[0]            
                hydrophobic_pos = [pos for pos, char in enumerate(seq) if char in ['F', 'M', 'W', 'I', 'V', 'L', 'P', 'A']]
                hydrophobic = np.zeros_like(annot_location_arr)
                hydrophobic[hydrophobic_pos] = 1.0
                positive_charged_pos = [pos for pos, char in enumerate(seq) if char in ['K', 'R']]
                positive_charged = np.zeros_like(annot_location_arr)
                positive_charged[positive_charged_pos] = 1.0
                annot_location_arr = np.add(hydrophobic, positive_charged)

            res = pointbiserialr(annot_location_arr, relevances)            
            scores["pbr"].append(res.correlation)
                                    
    scores = pd.DataFrame(scores)
    median_correlation_per_class = scores.groupby("label").median()

    print(f"Wilcoxon signed rank test (across all proteins) if correlation coeffs between relevances in embedding layer and {tag} sites is larger than 0.")
    res = {}
    r = {}
    classes = sorted(scores["label"].unique())
    for i in classes:
        try:
            (statistic, pvalue) = normaltest(scores.pbr[scores.label==i])
            print("Class %i, normality test, pvalue < 0.05? %s" % (i, (pvalue<0.05)))                
        except:
            print("normaltest not possible")
        res[i] = wilcoxon(scores.pbr[scores.label==i], alternative="greater").pvalue
        print("Class %i: p=%.3f" % (i, res[i]))
        r[i] = median_correlation_per_class.loc[i].pbr
        print("Class %i: r=%.3f" % (i, r[i]))    

    return pd.DataFrame({"Class": classes, "p":res.values(), "r": r.values()})

    
if __name__ == "__main__":
    """ Correlate integrated gradient relevances and annotations on sequence level. Calculate pointwise biserial correlation coefficients (PBR) and relevance rank accuracy (RRA) scores between the relevances and active|binding|transmembrane sites or motif or prosite annotations of the aminoacid sequences; on the embedding level and for each of the 480 heads/layers. Compute summary statistic over correlation coefficients across all annotated proteins.
    
    Output: Figures in /results folder
    """

    print(f"https://www.ebi.ac.uk/QuickGO/term/{SELECTED_GO_TERM} ({GO_TERMS[SELECTED_GO_TERM]})")

    # Folder for results and figures
    Path("results").mkdir(exist_ok=True)

    # Select either GO-temporalsplit or GO-CAFA3 dataset
    if 1: 
        data_path = Path("data/clas_go_deepgoplus_temporalsplit/")
        go_path = Path("data-2016") # clas_go_deepgoplus_temporalsplit
    else: 
        data_path = Path("data/clas_go_deepgoplus_cafa/")
        go_path = Path("data-cafa") # clas_go_deepgoplus_cafa
    print(data_path)
    
    relevance_dir = data_path/"ig_outputs_combined/" # relevance csv's location
        
    # Select label dimensions in multilabel GO label vector: (GO-term with all its children GO-terms)=1 vs. rest =0
    label_itos = np.load(data_path/"label_itos.npy")
    selected_label_dim = np.where(label_itos == SELECTED_GO_TERM)[0][0]

    n_annotated = {}  # samples in test split for each type of annotation (for the selected GO-term)
    
    # p-values of t-tests over correlation coefficients of annotation and relevances in embedding layer for each type of annotation (for the selected GO-term)
    p_embedding = {}
    # median correlation coefficients r) of annotation and relevances in embedding layer for each type of annotation (for the selected GO-term)
    r_embedding = {}

    copyfile(data_path/f"transmembrane_test.json", data_path/f"hydrophobic_test.json")
    copyfile(data_path/f"transmembrane_test.json", data_path/f"positive_test.json")
    copyfile(data_path/f"transmembrane_test.json", data_path/f"hydrophobic_and_positive_test.json")
    copyfile(data_path/f"transmembrane_test.json", data_path/f"transmembrane_hydrophobic_test.json")
    copyfile(data_path/f"transmembrane_test.json", data_path/f"transmembrane_positive_test.json")

    conditions = ["transmembrane", "hydrophobic", "positive", "transmembrane_hydrophobic", "transmembrane_positive"] #"hydrophobic_and_positive", "active", "binding", "transmembrane", "motif", "prosite"]
    for tag in conditions:
        print("\nAnnotation: %s" % tag)     
                
        # Load data
        annotation_test_path = data_path/f"{tag}_test.json"
        data = []
        with open(annotation_test_path) as f:
            for line in f:
                data.append(json.loads(line))
        df_test_annot = pd.DataFrame(data[0], columns=["sequence", "name", "label", "location"])
        df_test_annot = df_test_annot.drop_duplicates(subset=["sequence", "name", "label"])
        
        # Choose selected GO-term as class
        df_test_annot['label'] = df_test_annot['label'].apply(lambda x: int(x[selected_label_dim]))

        # Keep only proteins annotated with the selected GO-term (positive class)
        df_test_annot = df_test_annot.loc[ df_test_annot['label'] == 1] # Required here in annotation_eval.py, even if already done during IG computation in integrated_gradient_helper.py
        
        # Number of samples per class
        print("Annotated samples", len(df_test_annot) )
        a = 1 # positive class only (proteins annotated with the selected GO-term)
        n_annotated[tag] = len(pd.DataFrame(df_test_annot[df_test_annot["label"]==a]))
        print("Number of samples for class %i: %i" % (a, n_annotated[tag]))
            
        # Compute correlation of relevances with annotations for the embedding layer
        df_embedding = embedding_calc(df_test_annot, data_path, tag)
        df_embedding = df_embedding[df_embedding["Class"]==1]
        
        if tag=="prosite": tag=tag.upper() # for the figure below
        if tag=="transmembrane": tag="trans-\nmembrane" # for the figure below
        if tag=="transmembrane_hydrophobic": tag="trans-\nmembrane\nhydrophobic" # for the figure below
        if tag=="transmembrane_positive": tag="trans-\nmembrane\npositive" # for the figure below
        if tag=="hydrophobic_and_positive": tag="hydrophobic\nand\npositive" # for the figure below
        p_embedding[tag] = df_embedding["p"].iloc[0]
        r_embedding[tag] = df_embedding["r"].iloc[0]
        if tag=="PROSITE": tag=tag.lower()
        if tag=="trans-\nmembrane": tag="transmembrane"
        if tag=="trans-\nmembrane\nhydrophobic": tag="transmembrane_hydrophobic"
        if tag=="trans-\nmembrane\npositive": tag="transmembrane_positive"
        if tag=="hydrophobic\nand\npositive": tag="hydrophobic_and_positive"

    
    # Barplot of p-values of t-tests over correlation coefficients of annotation and relevances in embedding layer for each type of annotation (for the selected GO-term)
    plt.figure(figsize=(len(conditions)+1,4))
    sns.barplot(x=list(p_embedding.keys()), y=-np.log(list(p_embedding.values())), width=0.4, saturation=0.7)
    xlimits = plt.gca().get_xlim()
    plt.plot(xlimits, -np.log([0.05, 0.05]), 'b', lw=1)
    plt.xlim(xlimits)
    plt.ylim([0, 35])    
    plt.xlabel("Annotation")
    plt.ylabel("-log(p)")
    plt.savefig(Path("results", f"{GO_TERMS[SELECTED_GO_TERM]}-annot-rel-corr-embedding-p-transmembrane.png"), bbox_inches="tight")
    plt.close()
