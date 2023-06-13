"""
Evaluate Gene Ontology predictions

The code in this file was adapted from https://github.com/nstrodt/UDSMProt/blob/master/code/utils/evaluate_deepgoplus.py 
which was adapted from https://github.com/bio-ontology-research-group/deepgoplus/blob/master/evaluate_deepgoplus.py
published under the BSD 3-Clause "New" or "Revised" License: https://github.com/bio-ontology-research-group/deepgoplus/blob/master/LICENSE
We cite the aforementioned license below the code.
"""

import numpy as np
import pandas as pd
import math
import os
import torch
import yaml
from collections import deque, Counter

from pathlib import Path

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'

FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}

class Ontology(object):

    def __init__(self, filename='./data-2016/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None

    def has_term(self, term_id):
        return term_id in self.ont

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])
            self.ic[go_id] = math.log(min_n / n, 2)
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
        if obj is not None:
            ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
        return ont


    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

def evaluate_deepgoplus(train_data_file, test_data_file, terms_file,
         diamond_scores_file, gofile, ont, preds=None, export=False,evaluate=True,verbose=False):

    go_rels = Ontology(gofile, with_rels=True)
    if(isinstance(terms_file,list) or isinstance(terms_file,np.ndarray)):
        terms = terms_file
    else:
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    annotations = train_df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i

    
    # BLAST Similarity (Diamond)
    diamond_scores = {}
    with open(diamond_scores_file) as f:
        for line in f:
            it = line.strip().split()
            if it[0] not in diamond_scores:
                diamond_scores[it[0]] = {}
            diamond_scores[it[0]][it[1]] = float(it[2])

    blast_preds = []
    for i, row in enumerate(test_df.itertuples()):
        annots = {}
        prot_id = row.proteins
        # BlastKNN
        if prot_id in diamond_scores:
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= annotations[prot_index[p_id]]
                total_score += score
            allgos = list(sorted(allgos))
            sim = np.zeros(len(allgos), dtype=np.float32)
            for j, go_id in enumerate(allgos):
                s = 0.0
                for p_id, score in sim_prots.items():
                    if go_id in annotations[prot_index[p_id]]:
                        s += score
                sim[j] = s / total_score
            ind = np.argsort(-sim)
            for go_id, score in zip(allgos, sim):
                annots[go_id] = score
        blast_preds.append(annots)
        
    # DeepGOPlus
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    labels = test_df['annotations'].values
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    # print(len(go_set))
    deep_preds = []
    alphas = {NAMESPACES['mf']: 0.55, NAMESPACES['bp']: 0.59, NAMESPACES['cc']: 0.46}
    for i, row in enumerate(test_df.itertuples()):
        annots_dict = blast_preds[i].copy()
        for go_id in annots_dict:
            annots_dict[go_id] *= alphas[go_rels.get_namespace(go_id)]
        for j, score in enumerate(row.preds if preds is None else preds[i]):
            go_id = terms[j]
            score *= 1 - alphas[go_rels.get_namespace(go_id)]
            if go_id in annots_dict:
                annots_dict[go_id] += score
            else:
                annots_dict[go_id] = score
        deep_preds.append(annots_dict)
    
    if(export):
        export_cafa(test_df,deep_preds,"DeepGOPlus_1_all.txt")
    if(evaluate):
        print("Evaluating scores")
        compute_prmetrics(labels,deep_preds,go_rels,ont=ont,verbose=verbose)
        #aucs = compute_roc(labels,deep_preds)
        #print("aucs:",aucs)
        #print("mean aucs(predicted):",np.mean(aucs))
        #print("mean aucs(all):",(np.sum(aucs)+(len(test_annotations)-len(aucs))*0.5)/len(test_annotations))

def evaluate(train_data_file, test_data_file, terms_file,
         gofile, ont, preds=None, propagate_scores=False,export=False,evaluate=True,verbose=False):
    '''
    train_data_file: path to train_data.pkl
    test_data_file: path to test_data.pkl
    terms_file: path to terms.pkl or just a list or nparray of labels
    '''
    go_rels = Ontology(gofile, with_rels=True)
    
    if(isinstance(terms_file,list) or isinstance(terms_file,np.ndarray)):
        terms = terms_file
    else:
        terms_df = pd.read_pickle(terms_file)
        terms = terms_df['terms'].values.flatten()
    terms_dict = {v: i for i, v in enumerate(terms)}

    train_df = pd.read_pickle(train_data_file)
    test_df = pd.read_pickle(test_data_file)
    annotations = train_df['annotations'].values
    annotations = list(map(lambda x: set(x), annotations))
    test_annotations = test_df['annotations'].values
    test_annotations = list(map(lambda x: set(x), test_annotations))
    go_rels.calculate_ic(annotations + test_annotations)

    # Print IC values of terms
    ics = {}
    for term in terms:
        ics[term] = go_rels.get_ic(term)

    prot_index = {}
    for i, row in enumerate(train_df.itertuples()):
        prot_index[row.proteins] = i
    
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    
    labels = test_df['annotations'].values        
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels)) 
    
    if(preds is None):
        deep_preds = []
        for i, row in enumerate(test_df.itertuples()):
            annots_dict = {}
            for j, score in enumerate(row.preds):
                go_id = terms[j]
                annots_dict[go_id] = score
            deep_preds.append(annots_dict)
    else:
        deep_preds = [{ terms[i] : y for i,y in enumerate(x)} for x in preds]
            
    # Propagate scores (a la deepgo)
    if(propagate_scores):
        print("Propagating scores a la deepgo")
        deepgo_preds = []
        for annots_dict in deep_preds:
            annots = {}
            for go_id, score in annots_dict.items():
                for a_id in go_rels.get_anchestors(go_id):
                    if a_id in annots:
                        annots[a_id] = max(annots[a_id], score)
                    else:
                        annots[a_id] = score
            deepgo_preds.append(annots)
        deep_preds = deepgo_preds        
    
    # compute PR metrics
    if(evaluate):
        print("Evaluating scores")
        compute_prmetrics(labels,deep_preds,go_rels,ont=ont,verbose=False)

def compute_prmetrics(labels,deep_preds,go_rels,ont="mf",verbose=False):
    go_set = go_rels.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    for t in range(0, 101):
        threshold = t / 100.0
        preds = []
        for i in range(len(deep_preds)):
            annots = set()
            for go_id, score in deep_preds[i].items():
                if score >= threshold:
                    annots.add(go_id)

            new_annots = set()
            for go_id in annots:
                new_annots |= go_rels.get_anchestors(go_id)
            preds.append(new_annots)
            
        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
                    
        fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(go_rels, labels, preds)
        avg_fp = sum(map(lambda x: len(x), fps)) / len(fps)
        avg_ic = sum(map(lambda x: sum(map(lambda go_id: go_rels.get_ic(go_id), x)), fps)) / len(fps)
        if(verbose):
            print(f'{avg_fp} {avg_ic}')
        precisions.append(prec)
        recalls.append(rec)
        print(f'Fscore: {fscore}, Precision: {prec}, Recall: {rec} S: {s}, RU: {ru}, MI: {mi} threshold: {threshold}')
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s
    print(f'Fmax: {fmax:0.3f}, Smin: {smin:0.3f}, threshold: {tmax}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')

    df = pd.DataFrame({'precisions': precisions, 'recalls': recalls})
    df.to_pickle(f'PR_{ont}.pkl')

def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns


if __name__ == "__main__":

    # Current working directory
    cwd = os.getcwd()
    os.chdir("models/")

    # Select GPU-cluster job
    print("Please download job output zz_*.tar from GPU-cluster and place it in the models/ directory.")     
    job_id = input("Enter job ID (integer in zz_*.tar) and press <ENTER>.\n")

    # Job output contains: y_hat_all.pt with predictions, y_all.pt with labels, label_itos.npy with mapping from numbers to GO terms, hparams.yaml with info if temporalsplit or CAFA3
    outputfiles = ["y_hat_all.pt", "y_all.pt", "label_itos.npy", "hparams.yaml"]
    
    # Remove extracted output files from potential previous runs    
    for of in outputfiles: 
        if Path(of).exists(): 
            Path(of).unlink()            
            
    # Extract outputfiles from job output zz_*.tar of GPU-cluster
    os.system("tar --strip-components=1 -xf zz_%s.tar" % job_id)
    os.system("tar --strip-components=3 -xf zz_%s.tar job_results/lightning_logs/0/hparams.yaml" % job_id)
    
    # Assert that all outputfiles are available
    for of in outputfiles:
        assert(Path(of).exists())

    # Find out if job dataset was temporalsplit or CAFA3
    with open('hparams.yaml', 'r') as file:
        hparams = yaml.safe_load(file)
    datasetfile = hparams["datasetfile"]

    if datasetfile == "clas_go_deepgoplus_temporalsplit.tar.gz":
        path = Path("data-2016")
        label_dims = 5101
    elif datasetfile == "clas_go_deepgoplus_cafa.tar.gz":
        path = Path("data-cafa")
        label_dims = 5220
             
    # Download files needed for the evaluation from the DeepGO website
    if not path.exists():
        if not Path(str(path) + ".tar.gz").exists():
            os.system("wget https://deepgo.cbrc.kaust.edu.sa/data/" + str(path) + ".tar.gz")
        os.system("mkdir " + str(path))
        os.system("tar xvzf " + str(path) + ".tar.gz -C " + str(path))

    # Load predictions
    y_hat_all = torch.load("y_hat_all.pt", map_location=torch.device('cpu'))    
    preds = torch.reshape(y_hat_all.squeeze(),(-1,label_dims)).numpy()

    # Load corresponding labels
    y_all = torch.load("y_all.pt", map_location=torch.device('cpu'))
    labels = torch.reshape(y_all.squeeze(),(-1,label_dims)).numpy()

    # Load mapping from numbers to GO terms
    label_itos = np.load("label_itos.npy")

    # Some check
    assert len(labels) == len( pd.read_pickle(path/"predictions.pkl")["labels"] )
        
    # Assert that the labels corresponding to our predictions are the same like the labels in predictions.pkl from DeepGO
    # We train on 1 or more GPU, but always test on 1 GPU only, because multi-GPU shuffles the order of the test samples which prohibits the evaluation.
    for i, y in enumerate(pd.read_pickle(path/"predictions.pkl")["labels"]):
        a = label_itos[labels[i]==1]
        b = pd.read_pickle(path/"terms.pkl").terms[y==1]
        assert( set(a) == set(b) )
    
    # Based on model only
    for ont in ["mf", "bp", "cc"]: 
        print(ont)
        evaluate(str(path/"train_data.pkl"), str(path/"test_data.pkl"), label_itos, str(path/"go.obo"), ont, preds)    

        #print("\n\n\nOriginal evaluation with the predictions from https://deepgo.cbrc.kaust.edu.sa/data/data-2016.tar.gz. \n" 
        #    "Expected results for "mf": Fmax: 0.409, Smin: 11.296, threshold: 0.12. AUPR: 0.350:\n")
        #evaluate(str(path/"train_data.pkl"), str(path/"predictions.pkl"), str(path/"terms.pkl"), str(path/"go.obo"), ont) 
    
    # Based on model combined with DiamondScores (which uses multiple sequence alignment)
    print(10*"\n"+"With DiamondScores:"+3*"\n")        
    for ont in ["mf", "bp", "cc"]: 
        print(ont)
        evaluate_deepgoplus(str(path/"train_data.pkl"), str(path/"test_data.pkl"), label_itos, path/"test_diamond.res", str(path/"go.obo"), ont, preds)

    os.chdir(cwd)


"""
BSD 3-Clause License

Copyright (c) 2019, Bio-Ontology Research Group
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
