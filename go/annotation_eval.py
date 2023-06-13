import os, re, json, csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.stats import ttest_1samp
from metrics import pbr, rra
from statsmodels.stats.multitest import multipletests

SELECTED_GO_TERM = json.load(open('parameters.json'))["SELECTED_GO_TERM"]
GO_TERMS = {"GO:0003824":"catalytic-activity", "GO:0016020":"membrane", "GO:0005488":"binding"}

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

            prot_label = int(df_test_annot.loc[df_test_annot.name == prot_name]["label"])
            
            relevances_str_list = b[2] # str '[0.0,..,0.0]'
            relevances_list_of_strings = relevances_str_list[1:-1].split(',')
            relevances = [float(x) for x in relevances_list_of_strings]

            prot_info = df_test_annot[df_test_annot.name == prot_name]
            locations = prot_info.location.iloc[0]
            scores["name"].append(prot_name)
            scores["label"].append(prot_label)
            score = pbr(relevances, locations)
            scores["pbr"].append(score)
                
    scores = pd.DataFrame(scores)
    median_correlation_per_class = scores.groupby("label").median()

    print(f"T-test (across all proteins) if correlation coeffs between relevances in embedding layer and {tag} sites is larger than 0.")
    res = {}
    r = {}
    classes = sorted(scores["label"].unique())
    for i in classes:
        res[i] = ttest_1samp(scores.pbr[scores.label==i], 0.0, alternative="greater").pvalue
        print("Class %i: p=%.3f" % (i, res[i]))
        r[i] = median_correlation_per_class.loc[i].pbr
        print("Class %i: r=%.3f" % (i, r[i]))    

    return pd.DataFrame({"Class": classes, "p":res.values(), "r": r.values()})
    
def create_metric_score_dataframes(df_test_annot, relevance_dir, tag:str):
    """ Calculating annotation-relevance correlations on all annotated proteins. """
    
    D = {'pbr':{},'rra':{}}
    for idx, prot in tqdm(df_test_annot.iterrows()):
        prot_name = prot["name"]
        prot_label = prot['label']
        prot_locations = prot['location']
        
        prot_rel_df = pd.read_csv(relevance_dir/f"{prot_name}.csv")

        for metric_name, metric_func in [("pbr",pbr),("rra",rra)]:
            scores = []
            for head_idx in range(prot_rel_df.shape[0]):
                prot_rel = prot_rel_df.iloc[head_idx].values
                try:
                    score = metric_func(prot_rel,prot_locations)
                except TypeError as err:
                    print(prot_name,prot_label,prot_locations)
                    raise(err)
                scores.append(score)
            D[metric_name][prot_name] = scores
        
    for metric_name,val in D.items():
        df = pd.DataFrame.from_dict(val,orient='index')
        df.to_json(Path("results", f"{metric_name}_{tag}.json"))


def plot_summary_of_correlations(metric_name:str, tag:str):
    """Plot summary of annotation-relevance-correlations as heatmaps. T-test for point-biserial correlation (and average per class for relevance rank accuracy) visualize 480 element vector as heatmap and store it in results/.
    Args:
        metric_name (str): 'pbr' (point biserial correlation coefficient) or 'rra' (relevance rank accuracy)
        tag: type of sequence level annotation ("active", "binding", "transmembrane", "motif", "prosite")        
    Output: heatmaps in results/
    """
    df_metric_scores = pd.read_json(Path("results", f"{metric_name}_{tag}.json")) # shape (n_samples, 480)

    if metric_name=="pbr": # point-biserial-r
        cmap_label = "-log(p)"
        res = ttest_1samp(df_metric_scores, 0.0, alternative='greater') # t-test over all correl. coefficients (one per sample)            
        (reject_t, pvals_corrected, alphacSidak, alphacBonf) = multipletests(res.pvalue, alpha=0.05, method="fdr_bh")
        # Corrected for multiple comparisons; reject_t used as significance threshold by "mask" parameter of sns.heatmap        
        res = -np.log(pvals_corrected)
        
    elif metric_name=="rra":
        cmap_label = "Rank accuracy"
        res = df_metric_scores.mean(axis=0).to_numpy()

    plt.figure(figsize=(4.5,4))
    sns.heatmap(data=res.reshape(30,16), mask=(~reject_t).reshape(30,16), vmin=0, cmap=sns.color_palette("flare", as_cmap=True), cbar_kws={'label': cmap_label})
    plt.xlabel("Heads")
    plt.ylabel("Layers")

    plt.savefig(Path("results", f"{GO_TERMS[SELECTED_GO_TERM]}-annot-rel-corr-{metric_name}-{tag}.png"), bbox_inches="tight")
    plt.close()

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
    
    for tag in ["active", "binding", "transmembrane", "motif", "prosite"]:
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
        p_embedding[tag] = df_embedding["p"].iloc[0]
        r_embedding[tag] = df_embedding["r"].iloc[0]
        if tag=="PROSITE": tag=tag.lower()
        if tag=="trans-\nmembrane": tag="transmembrane"
        
        # Compute correlation of relevances with annotations for each head in every layer
        create_metric_score_dataframes(df_test_annot, relevance_dir, tag)
        
        # Summarise correlation results (pbr: p-value of t-test over correlation coefficients; rra: mean relevance rank accuracy)
        plot_summary_of_correlations("pbr", tag)

    
    # Barplot of p-values of t-tests over correlation coefficients of annotation and relevances in embedding layer for each type of annotation (for the selected GO-term)
    plt.figure(figsize=(4.5,4))
    sns.barplot(x=list(p_embedding.keys()), y=list(p_embedding.values()), width=0.4, saturation=0.7)
    xlimits = plt.gca().get_xlim()
    plt.plot(xlimits, [0.05, 0.05], 'b', lw=1)
    for i, p in enumerate(list(p_embedding.values())):
        if p < 0.05: # smaller than significance threshold 0.05 
            plt.text(i, 0.06, "*", fontdict = {"fontsize":14})
    plt.xlim(xlimits)
    plt.ylim([0, 1])
    plt.xlabel("Annotation")
    plt.ylabel("p")
    # Add number of annotated samples for each type of annotation in brackets to the xtick labels
    new_xticklabels = ["%s \n(%i)" % (a,b) for (a,b) in zip(list(p_embedding.keys()), list(n_annotated.values()))]
    plt.gca().set_xticklabels(new_xticklabels)
    # Add significance level 0.05 as additional y-tick
    new_yticks = sorted(np.append(plt.gca().get_yticks(), 0.05))
    plt.gca().set_yticks(new_yticks)   
    plt.savefig(Path("results", f"{GO_TERMS[SELECTED_GO_TERM]}-annot-rel-corr-embedding-p.png"), bbox_inches="tight")
    plt.close()

