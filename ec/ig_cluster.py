import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def t_sne(ig_outputs_path: str, test_json_path: str):
    """
    Non-linear dimensionality reduction on ig_outputs with t-distributed stochastic neighbor embedding (t-SNE)
    Find low-dimensional representation of data in which distances respect distances in original high-dimensional space.
    Principal component analysis (PCA) prior to t-SNE, as recommended on
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    Args:
        ig_outputs_path (str): path to ig_outputs/ containing .csv of each protein with dimensions 480xN
        test_json_path (str): path to test.json file
    Output: Create .png file in results folder.
    """

    relevances, targets = load_data(ig_outputs_path, test_json_path)    
    print("relevances:", relevances.shape)
    print("targets:", targets.shape)
    print("Number of classes: %i" % len(np.unique(targets)))

    # principal component analysis first
    pca = PCA(n_components=50)
    relevances = pca.fit_transform(relevances)

    # t-SNE with default parameters
    tsne = TSNE(random_state=123)
    relevances = tsne.fit_transform(relevances)

    plot_scatter(relevances, targets)

def load_data(ig_outputs_path: str, test_json_path: str):
    """
    Get targets and relevance head vectors for each protein in ig_outputs_path
    Args:
        ig_outputs_path (str): path to ig_outputs/ containing the csv files
        test_json_path(str): path to test.json file
    Returns:
        relevances (np.array): array of relevance vectors of size (480,)
        targets (np.array): array of targets
    """        
    df_test = pd.read_json(test_json_path)
    relevances = []    
    targets = []    
    for index, row in df_test.iterrows():
        prot_file_path = ig_outputs_path+row["name"]+".csv"
        if os.path.isfile(prot_file_path):
            df_prot = pd.read_csv(prot_file_path)
            head_rel_vec = df_prot.sum(axis=1).to_numpy()
            relevances.append(head_rel_vec)
            targets.append(row["label"])        
        else:
            print(row["name"], "not found")

    return np.asarray(relevances), np.asarray(targets)

def plot_scatter(x, targets):
    """
    Visualize PCA and t-SNE outputs
    Parameter:
        x: ndarray or coordinates per point. shape(5579,2)
        targets: pd.series consisting of target values
    """
    f = plt.figure(figsize=(9, 9))
    ax = plt.subplot(aspect='equal')    
    palette = np.array(sns.color_palette("hls", max(targets)+1))
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, c=palette[targets.astype(int)])   
    
    # Show legend in case of default test_json_path="data/ec50_level1/test.json" (otherwise too many elements shown in legend)
    if (np.unique(targets)<6).all():
        patches=[]
        for i in np.unique(targets):
            patch = mpatches.Patch(color=palette[i,:], label=f'Class {i+1}') # add 1 to start with "1" not "0"
            patches.append(patch)
        plt.legend(handles=patches)
    
    plt.xlabel("t-SNE X")
    plt.ylabel("t-SNE Y")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(f"{results_dir}/tsne.png")
    

if __name__ == "__main__":

    t_sne(ig_outputs_path="data/ec50_level1/ig_outputs_combined/", test_json_path="data/ec50_level1/test.json")

