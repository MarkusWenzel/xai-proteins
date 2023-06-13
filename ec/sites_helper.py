"""This file contains code modified from https://github.com/nstrodt/UDSMProt/blob/master/code/utils/proteomics_utils.py which was published under the BSD License: https://github.com/nstrodt/UDSMProt/blob/master/LICENSE
We cite the aforementioned license below the code.
"""

import os,re,sys,inspect
import numpy as np
import pandas as pd
from lxml import etree
from tqdm import tqdm

def create_sites_datasets():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)
    parse_features=["active site", "binding site", "short sequence motif", "transmembrane region"]
    print("\nInfo: Creating datasets from uniprot.xml for protein types:",parse_features)
    create_datasets("data/uniprot_sprot_2017_03.xml",parse_features)

def create_datasets(script_dir, parse_features=["active site", "binding site", "short sequence motif", "transmembrane region"]):
    """ Create .pkl dataframe for each feature listed. """
    
    print("Parsing xml and storing as xml_df.pkl ")
    # parse xml into dataframe for all listed features
    df = sprot_setup(script_dir, parse_features) 
    df.to_pickle("data/xml_df.pkl",protocol=4)

    # filter by features and create pkl output
    for feature in parse_features:
        print(f"Creating {feature} dataset")
        df_filtered = filter_df_by_feature(df, feature)
        if feature=="short sequence motif":
            out_dir = "motif_site"
        elif feature=="transmembrane region":
            out_dir = "transmembrane_site" 
        else:
            out_dir = feature.replace(" ","_")
        df_filtered.to_pickle("data/"+out_dir+".pkl")
    print("Info: Site Datasets creation complete")

def filter_df_by_feature(df, feature_name):
    '''
    Filter the general dataframe for specific feature names and modify columns,index
    '''
    df_filtered = df.loc[df.type==feature_name]
    df_filtered = df_filtered[["name","primary","EC-Label","type","location","description","evidence"]]
    df_filtered['name'] = df_filtered.index.values
    df_filtered['location'] = df_filtered.location.apply(lambda x:[x])
    df_filtered = df_filtered.set_index(np.arange(len(df_filtered)))

    return df_filtered    

#set max_entries to 0 for full sprot
def sprot_setup(script_dir, parse_features):
    '''
    Create a dataframe with the proteins that have the given featurename by parsing
    the uniprot xml.
    other possible features to parse "metal ion-binding site","region of interest"..
    
    Parameter:
        script_dir: location of the xml file.
    Output:
        df: Dataframe containing proteins with the featurename
    '''
    max_entries=0
    df = parse_uniprot_xml(script_dir, max_entries, parse_features=parse_features)
    df["accession"]=df.index # store accession as explicit column
    df = df[df.ecs.apply(lambda x:len(x)>0)] # select rows with ec values
    df=df.explode("features") # separate entry for every motif/binding site
    df=df[~df.features.isna()] # select only non-zero entries
    df=df[df.features.apply(lambda x:len(x)>0)] # remove empty features

    # re-format into separate columns
    df["type"]=df["features"].apply(lambda x:x[0])
    df["description"]=df["features"].apply(lambda x:x[1])
    df["location"]=df["features"].apply(lambda x:x[2])
    df["evidence"]=df["features"].apply(lambda x:x[3]) 
    df.drop(["features","dataset"],axis=1,inplace=True) # remove the duplicates
    df = df.rename(columns={"sequence": "primary", "ecs": "EC-Label"})

    return df
        
def parse_uniprot_xml(filename,max_entries=0,parse_features=[]):
    '''parse uniprot xml file, which contains the full uniprot information (e.g. ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz)
    using custom low-level https://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    c.f. for full format https://www.uniprot.org/docs/uniprot.xsd

    Parameters:
        parse_features: a list of strings specifying the kind of features to be parsed such as "modified residue" for phosphorylation sites etc. (see https://www.uniprot.org/help/mod_res)
        (see the xsd file for all possible entries)
    '''
    context = etree.iterparse(str(filename), 
                              events=["end"], 
                              tag="{http://uniprot.org/uniprot}entry")
    context = iter(context)
    rows =[]
    
    for _, elem in tqdm(context):
        parse_func_uniprot(elem,rows,parse_features=parse_features)
        elem.clear()
        while elem.getprevious() is not None: # Returns the preceding sibling of this element or None.
            del elem.getparent()[0]
        if(max_entries > 0 and len(rows)==max_entries):
            break
    
    df=pd.DataFrame(rows).set_index("ID")
    df['name'] = df.name.astype(str)
    df['dataset'] = df.dataset.astype('category')
    df['organism'] = df.organism.astype('category')
    df['sequence'] = df.sequence.astype(str)

    return df

def parse_func_uniprot(elem, rows, parse_features=[]):
    '''
    extracting a single record from uniprot xml and place in rows variable.
    single example: https://www.uniprot.org/uniprot/O36015.xml
    Parameter:
        elem:
        rows: output list to append protein dictionaries to.
        parse_features: list of features to parse. aka the "type" 
    '''
    # retrieve sequence element 
    seqs = elem.findall("{http://uniprot.org/uniprot}sequence")
    sequence=""
    
    # Sequence & fragment from seqs element.
    sequence=""
    fragment_map = {"single":1, "multiple":2}
    fragment = 0
    seqs = elem.findall("{http://uniprot.org/uniprot}sequence")
    for s in seqs:
        if 'fragment' in s.attrib:
            fragment = fragment_map[s.attrib["fragment"]]
        sequence=s.text
        if sequence != "":
            break

    # dataset element
    dataset = elem.attrib["dataset"]

    # accession
    accession = ""
    accessions = elem.findall("{http://uniprot.org/uniprot}accession")
    for a in accessions:
        accession=a.text
        if accession !="":# primary accession! https://www.uniprot.org/help/accession_numbers!!!
            break

    # protein existence (PE in plain text)
    proteinexistence_map = {"evidence at protein level":5,"evidence at transcript level":4,"inferred from homology":3,"predicted":2,"uncertain":1}
    proteinexistence = -1
    accessions = elem.findall("{http://uniprot.org/uniprot}proteinExistence")
    for a in accessions:
        proteinexistence=proteinexistence_map[a.attrib["type"]]
        break

    # name
    name = ""
    names = elem.findall("{http://uniprot.org/uniprot}name")
    for n in names:
        name=n.text
        break

    # organism
    organism = ""
    organisms = elem.findall("{http://uniprot.org/uniprot}organism")
    for s in organisms:
        s1=s.findall("{http://uniprot.org/uniprot}name")
        for s2 in s1:
            if(s2.attrib["type"]=='scientific'):
                organism=s2.text
                break
        if organism !="":
            break

    # dbReference: PMP,GO,Pfam, EC
    ids = elem.findall("{http://uniprot.org/uniprot}dbReference")
    pfams = []
    gos =[]
    ecs = []
    pdbs =[]
    for i in ids:
        #cf. http://geneontology.org/external2go/uniprotkb_kw2go for Uniprot Keyword<->GO mapping
        #http://geneontology.org/ontology/go-basic.obo for List of go terms
        #https://www.uniprot.org/help/keywords_vs_go keywords vs. go
        if(i.attrib["type"]=="GO"):
            tmp1 = i.attrib["id"]
            for i2 in i:
                if i2.attrib["type"]=="evidence":
                    tmp2= i2.attrib["value"]
            gos.append([int(tmp1[3:]),int(tmp2[4:])]) # first value is go code, second eco evidence ID (see mapping below)
        elif(i.attrib["type"]=="Pfam"):
            pfams.append(i.attrib["id"])
        elif(i.attrib["type"]=="EC"):
            ecs.append(i.attrib["id"])
        elif(i.attrib["type"]=="PDB"):
            pdbs.append(i.attrib["id"])

    # keyword
    keywords = elem.findall("{http://uniprot.org/uniprot}keyword")
    keywords_lst = []
    for k in keywords:
        keywords_lst.append(int(k.attrib["id"][-4:]))#remove the KW-

    # parse features from protein element.
    if len(parse_features)>0:
        ptms=[]
        features = elem.findall("{http://uniprot.org/uniprot}feature")
        for f in features:
            if(f.attrib["type"] in parse_features):#only add features of the requested type
                locs=[]
                for l in f[0]:
                    #locs.append(int(l.attrib["position"]))
                    try: # <--
                        locs.append(int(l.attrib["position"]))
                    except: locs.append(999999) # <-- If position not available set to large number. Annotations above max_len=1000 amino acids will later be dropped                    
                ptms.append([f.attrib["type"],f.attrib["description"] if 'description' in f.attrib else "NaN",locs, f.attrib['evidence'] if 'evidence' in f.attrib else "NaN"])

    data_dict={"ID": accession, "name": name, "dataset":dataset, "proteinexistence":proteinexistence, "fragment":fragment, "organism":organism, "ecs": ecs, "pdbs": pdbs, "pfams" : pfams, "keywords": keywords_lst, "gos": gos,  "sequence": sequence}

    if len(parse_features)>0:
        data_dict["features"]=ptms
    rows.append(data_dict)


if __name__ == "__main__":
    create_sites_datasets()

    
"""
COPYRIGHT

The copyright in this software is being made available under the BSD
License, included below. This software is subject to other contributor rights,
including patent rights, and no such rights are granted under this license.

Copyright (c) 2019, Nils Strodthoff
All rights reserved.


LICENSE

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
