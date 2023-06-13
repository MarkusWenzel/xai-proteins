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


def plot_summary_of_correlations(metric_name:str, tag:str, labels):
    """Plot summary of annotation-relevance-correlations as heatmaps. T-test for point-biserial correlation (and average per class for relevance rank accuracy) visualize 480 element vector as heatmap and store it in results/.
    Args:
        metric_name (str): 'pbr' (point biserial correlation coefficient) or 'rra' (relevance rank accuracy)
        tag: type of sequence level annotation ("active", "binding", "transmembrane", "motif")        
    Output: heatmaps in results/
    """
    
    fig, axes = plt.subplots(1, 6, figsize=(27, 4))
    for i in range(0,6):
        df_metric_scores = pd.read_json(Path("results", f"{metric_name}_{tag}.json"))
        df_metric_scores = df_metric_scores[(labels == i).values] # select EC class

        if metric_name=="pbr": # point-biserial-r
            cmap_label = "-log(p)"
            res = ttest_1samp(df_metric_scores, 0.0, alternative='greater') # t-test over all correl. coefficients (one per sample)            
            (reject_t, pvals_corrected, alphacSidak, alphacBonf) = multipletests(res.pvalue, alpha=0.05, method="fdr_bh")            
            # Corrected for multiple comparisons; reject_t used as significance threshold by "mask" parameter of sns.heatmap
            res = -np.log(pvals_corrected)
            
        elif metric_name=="rra":
            cmap_label = "Rank accuracy"
            res = df_metric_scores.mean(axis=0).to_numpy()

        sns.heatmap(data=res.reshape(30,16), ax=axes[i], mask=(~reject_t).reshape(30,16), vmin=0, cmap=sns.color_palette("flare", as_cmap=True), cbar_kws={'label': cmap_label})
        axes[i].set_xlabel("Heads")
        axes[i].set_ylabel("Layers")
        axes[i].set_title(f"EC%i: corr({tag}, relev.)" % i)

    plt.savefig(Path("results", f"ec-annot-rel-corr-{metric_name}-{tag}.png"), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    """ Correlate integrated gradient relevances and annotations on sequence level. Calculate pointwise biserial correlation coefficients (PBR) and relevance rank accuracy (RRA) scores between the relevances and active|binding sites, motifs or transmembrane regions of the aminoacid sequences; on the embedding level and for each of the 480 heads/layers. Compute summary statistic over correlation coefficients across all annotated proteins.
    
    Output: Figures in /results folder
    """

    # Folder for results and figures
    Path("results").mkdir(exist_ok=True)

    data_path = Path("data/ec50_level1/")
    print(data_path)
    
    relevance_dir = data_path/"ig_outputs_combined/" # relevance csv's location
        
    # Labels (6 EC classes)
    label_itos = np.load(data_path/"label_itos.npy")

    n_annotated = {}  # samples in test split for each type of annotation
    
    # p-values of t-tests over correlation coefficients of annotation and relevances in embedding layer for each type of annotation 
    p_embedding = {}
    
    for tag in ["active", "binding", "transmembrane", "motif"]:
        print("\nAnnotation: %s" % tag)     

        df_test_annot = pd.read_json(data_path/f"{tag}_test.json")
        df_test_annot = df_test_annot.drop_duplicates(subset=["sequence", "name", "label"])
        
        # Number of samples per class
        print("Annotated samples", len(df_test_annot) )
        c = []
        for a in sorted(df_test_annot["label"].unique()):
            b = len(pd.DataFrame(df_test_annot[df_test_annot["label"]==a]))
            print("Number of samples for class %i: %i" % (a, b))
            c.append(b)
        
        # Compute correlation of relevances with annotations for the embedding layer
        df_embedding = embedding_calc(df_test_annot, data_path, tag)
        if tag=="transmembrane": tag="trans-\nmembrane" # for the figure below
        n_annotated[tag] = c
        df_embedding["tag"] = tag
        p_embedding[tag] = df_embedding
        if tag=="trans-\nmembrane": tag="transmembrane"
           
        # Compute correlation of relevances with annotations for each head in every layer
        create_metric_score_dataframes(df_test_annot, relevance_dir, tag)
        
        # Summarise correlation results (pbr: p-value of t-test over correlation coefficients; rra: mean relevance rank accuracy)
        plot_summary_of_correlations("pbr", tag, df_test_annot["label"])

    
    # Barplot of p-values of t-tests over correlation coefficients of annotation and relevances in embedding layer for each type of annotation
    if tag=="transmembrane": tag="trans-\nmembrane"
    df = p_embedding["active"].append(p_embedding["binding"]).append(p_embedding["trans-\nmembrane"]).append(p_embedding["motif"])
    df = df.rename(columns = {"tag":"Annotation", "Class":"EC"})
    df["EC"] += 1 # Renaming EC0-EC5 (Python-style indexing) to EC1-EC6 for consistency with <six> EC classes
    plt.figure(figsize=(4.5,4))
    sns.barplot(data=df, x="Annotation", y="p", hue="EC", width=0.8, saturation=0.7) 
    # Add significance level 0.05 as additional line and y-tick
    new_yticks = sorted(np.append(plt.gca().get_yticks(), 0.05))
    plt.gca().set_yticks(new_yticks)   
    xlimits = plt.gca().get_xlim()
    plt.plot(xlimits, [0.05, 0.05], 'b', lw=1)
    plt.xlim(xlimits)
    plt.ylim([0,1])
    plt.savefig(Path("results", f"ec-annot-rel-corr-embedding-p.png"), bbox_inches="tight")
    plt.close()    
    
    # Number of annotated samples per EC class and annotation
    df = pd.DataFrame(n_annotated).stack()
    df = df.reset_index().rename(columns = {"level_0":"EC", "level_1":"Annotation", 0:"n"})
    df["EC"] += 1 # Renaming EC0-EC5 (Python-style indexing) to EC1-EC6 for consistency with <six> EC classes
    plt.figure(figsize=(4.5,4))
    sns.barplot(data=df, x="Annotation", y="n", hue="EC", width=0.8, saturation=0.7)
    plt.savefig(Path("results", f"ec-nr-test-samples.png"), bbox_inches="tight")
    plt.close()
