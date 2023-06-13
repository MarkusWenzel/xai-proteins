import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from pathlib import Path
from scipy.stats import shapiro, wilcoxon, ttest_1samp
from matplotlib.colors import LogNorm
from statsmodels.stats.multitest import multipletests

def plot_overlay(reject_wil, metric_name:str, tag:str, ec:int, df_test_rel):
    """Plot overlay of ProtBert heads with positive relevance with ProtBert heads with positive annotation-relevance-correlations.
    Args:
    reject_wil: Vector of booleans for 480 heads. True where relevance in head significantly >0, such that Wilcoxon null hypothesis rejected.
    metric_name (str): 'pbr' (point biserial correlation coefficient) or 'rra' (relevance rank accuracy)
    tag: type of sequence level annotation ("active", "binding", "transmembrane", "motif")
    Output: heatmaps in results/
    """
    df_metric_scores = pd.read_json(Path("results", f"{metric_name}_{tag}.json")) # shape (n_samples, 480)
    df_metric_scores.reset_index(inplace=True)
    df_metric_scores = df_metric_scores.rename(columns = {"index":"name"})
    df_metric_scores = df_metric_scores.merge(df_test_rel.name)
    df_metric_scores = df_metric_scores.set_index ("name")

    if metric_name=="pbr": # point-biserial-r
        cmap_label = "-log(p)"
        res = ttest_1samp(df_metric_scores, 0.0, alternative='greater') # t-test over all correl. coefficients (one per sample)            
        (reject_t, pvals_corrected, alphacSidak, alphacBonf) = multipletests(res.pvalue, alpha=0.05, method="fdr_bh")
        # Corrected for multiple comparisons; reject_t used as significance threshold by "mask" parameter of sns.heatmap        
        res = -np.log(pvals_corrected)
        
    elif metric_name=="rra":
        cmap_label = "Rank accuracy"
        res = df_metric_scores.mean(axis=0).to_numpy()

    # Overlay ProtBert heads with positive relevance with ProtBert heads with positive annotation-relevance-correlations
    plt.figure(figsize=(3.15,2.8))
    sns.heatmap(data=res.reshape(30,16), mask=~(reject_t & reject_wil).reshape(30,16), vmin=0, cmap=sns.color_palette("flare", 20), cbar_kws={"label": cmap_label})
    plt.xlabel("Heads")
    plt.ylabel("Layers")
    plt.savefig(Path("results", f"ec{ec}-overlay-pos-rel-with-annot-rel-corr-{metric_name}-{tag}.png"), bbox_inches="tight")
    plt.close()


def shapiro_calc(df_test_rel):
    """ Shapiro-Wilk test for normality tests null hypothesis that data are drawn from normal distribution. """
    relevance_matrix = np.asarray([row for row in df_test_rel.relevances]) # number of samples x 480 (layers*heads)
    normal_dist_heads = []
    for head_idx in range(relevance_matrix.shape[1]):
        ma = relevance_matrix[:, head_idx]
        shapiro_results = shapiro(ma)
        if shapiro_results.pvalue > 0.05:
            normal_dist_heads.append((head_idx, shapiro_results.pvalue))

    print(f"Shapiro-Wilk test for normality: Relevances in {len(normal_dist_heads)}/{relevance_matrix.shape[1]} heads drawn from normal distribution, because p_value > 0.05.")


if __name__ == "__main__":
    """ Plot mean relevance of class "selected GO-term with children" per head and per layer. Wilcoxon rank-sum test if relevances significicantly larger than zero (as non-parametric alternative to Student's t-test, because assumptions are not met). Plot -log(p) of significantly positive relevances per class (per head/layer). """

    # Folder for images
    Path('results').mkdir(exist_ok=True)

    data_path = Path("data/ec50_level1/")
    print(data_path)

    # Choose all or only annotated proteins
    for tag in ["active", "binding", "transmembrane", "motif"]: # "all"
        print("\nAll or filtered for annotated proteins: %s" % tag)

        for ec in range(0,6):        
            df_test_rel = pd.read_pickle(data_path/'test_rel.pkl')

            # Choose selected EC class
            df_test_rel = df_test_rel.loc[df_test_rel["label"] == ec]
            
            if tag != "all":
                # filter for annotated proteins
                annotation_test_path = data_path/f"{tag}_test.json"
                df_test_annot = pd.read_json(annotation_test_path)
                df_test_annot = df_test_annot.drop_duplicates(subset=["sequence", "name", "label"])
                df_test_annot = df_test_annot["name"]    
                df_test_rel = df_test_rel.merge(df_test_annot)

            # Shapiro test if requirements for ANOVA, t-test are met ("relevances in n of 480 (16 heads x 30 layers) drawn from normal distribution, because p_value > 0.05, thus fullfill ANOVA and Student's t-test requirements").
            shapiro_calc(df_test_rel)
            
            # Wilcoxon signed-rank test if relevance larger than zero. Plot -log(p) of significantly pos. relevances (per head/layer)
            head_rel = pd.DataFrame(df_test_rel["relevances"].tolist())
            res_wil = np.ones(head_rel.shape[1])
            for i in range(0, len(head_rel.columns)):
                res_wil[i] = wilcoxon(head_rel[i], alternative="greater").pvalue

            (reject_wil, pvals_corrected, alphacSidak, alphacBonf) = multipletests(res_wil, alpha=0.05, method="fdr_bh")
            # Corrected for multiple comparisons; reject_wil used as significance threshold by "mask" parameter of sns.heatmap
            res_wil = -np.log(pvals_corrected)

            plt.figure(figsize=(4.5,4))
            sns.heatmap(data=res_wil.reshape(30,16), mask=(~reject_wil).reshape(30,16), vmin=0, cmap=sns.color_palette("Blues", as_cmap=True), cbar_kws={"label": "-log(p)"})
            plt.xlabel("Heads")
            plt.ylabel("Layers")
            plt.savefig(Path("results", f"ec{ec}-stat-wilcoxon-signed-rank-relevance-{tag}.png"), bbox_inches="tight")
            plt.close()
            
            # Overlay ProtBert heads <where relevance significantly correlates with annotations> with ProtBert heads <where relevance is significantly positive>
            if tag != "all":
                metric_name = "pbr" # point biserial r
                plot_overlay(reject_wil, metric_name, tag, ec, df_test_rel)

