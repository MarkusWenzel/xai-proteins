import os, re, json, csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from scipy.stats import normaltest,ttest_1samp,wilcoxon
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

            prot_label = float(df_test_annot.loc[df_test_annot.name == prot_name]["label"])
            
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
    for i,j in enumerate(classes):
        try:
            (statistic, pvalue) = normaltest(scores.pbr[scores.label==i])
            print("Class %i, normality test, pvalue < 0.05? %s" % (i, (pvalue<0.05)))
        except:
            print("normaltest not possible")
        res[i] = wilcoxon(scores.pbr[scores.label==j], alternative="greater").pvalue
        print("Class %s: p=%.3f" % (j, res[i]))
        r[i] = median_correlation_per_class.loc[j].pbr
        print("Class %s: r=%.3f" % (j, r[i]))    

    return pd.DataFrame({"Class": classes, "p":res.values(), "r": r.values()})    


if __name__ == "__main__":
    """ Correlate integrated gradient relevances and annotations on sequence level. Calculate pointwise biserial correlation coefficients (PBR) and relevance rank accuracy (RRA) scores between the relevances and active|binding sites, motifs or transmembrane regions of the aminoacid sequences; on the embedding level and for each of the 480 heads/layers. Compute summary statistic over correlation coefficients across all annotated proteins.
    
    Output: Figures in /results folder
    """

    # Folder for results and figures
    Path("results").mkdir(exist_ok=True)

    for EC in [40,50]:
        for L in [1,2]:
            data_path = Path(f"data/ec{EC}_level{L}/")
            if not (data_path/"ig_outputs/embedding_attribution.csv").exists():
                print(str(data_path/"ig_outputs/embedding_attribution.csv"), "does not exist.")
                continue
            print(data_path)            
                
            # Labels (6 or 65 EC classes)
            label_itos = np.load(data_path/"label_itos.npy")

            n_annotated = {}  # samples in test split for each type of annotation
            
            # p-values of Wilcoxon signed rank tests over correlation coefficients of annotation and relevances in embedding layer for each type of annotation 
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
            
            # Barplot of p-values of t-tests over correlation coefficients of annotation and relevances in embedding layer for each type of annotation
            if tag=="transmembrane": tag="trans-\nmembrane"
            df = p_embedding["active"].append(p_embedding["binding"]).append(p_embedding["trans-\nmembrane"]).append(p_embedding["motif"])
            df = df.rename(columns = {"tag":"Annotation", "Class":"EC"})
            df["-log(p)"]=-np.log(df["p"])
            # Renaming EC0-EC5 (Python-style indexing) to EC1-EC6 for consistency with <six> EC classes
            if L==1: df["EC"] = df["EC"].apply(lambda x: int(x)+1)                 

            plt.figure(figsize=(4.5,4))           
            sns.barplot(data=df, x="Annotation", y="-log(p)", hue="EC", width=0.8, saturation=0.7)
            plt.ylim([0, 30])
            # remove legend for EC level 2 with many classes
            if L==2: plt.gca().get_legend().set_visible(False)
            # Add significance level 0.05 as additional line and y-tick
            new_yticks = sorted(np.append(plt.gca().get_yticks(), -np.log(0.05)))
            plt.gca().set_yticks(new_yticks)   
            xlimits = plt.gca().get_xlim()
            plt.plot(xlimits, -np.log([0.05, 0.05]), 'b', lw=1)
            plt.xlim(xlimits)
            plt.savefig(Path("results", f"ec{EC}-L{L}-annot-rel-corr-embedding-p.png"), bbox_inches="tight")           
            plt.close()
            
            print(n_annotated)

