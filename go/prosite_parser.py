import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

def parse_prosite(filename_msa):
    filename_msa = Path(filename_msa)
    with open(filename_msa) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    result=[]
    for line in lines:
        if(line.startswith(">")):#skip the pattern part
            tmp1 = line
            protein_accession=tmp1.split("/")[0].split("|")[1]
            protein_name=acc=tmp1.split("/")[0].split("|")[0][1:]
            fromto = [int(x) for x in tmp1.split("/")[1].split(":")[0].split("-")]
            protein_start_site= fromto[0]
            protein_end_site=fromto[1]
            prosite_name=tmp1.split(":")[1].strip()
            result.append({"protein_accession":protein_accession, "protein_name":protein_name, "start_site":protein_start_site, "end_site":protein_end_site, "prosite_name":prosite_name, "prosite_id":filename_msa.stem})#, "pattern":prosite_pattern})
    return pd.DataFrame(result)

def parse_prosite_full(folder="."):
    '''parse content of https://ftp.expasy.org/databases/prosite/prosite_alignments.tar.gz
    c.f. http://current.geneontology.org/ontology/external2go/prosite2go for prosite GO mapping'''
    res = []
    for p in tqdm(list(Path(folder).glob("*.msa"))):
        res.append(parse_prosite(p))
    return pd.concat(res)
