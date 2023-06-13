"""
Obtain all children Gene-Ontology-terms from selected ancestor GO term (e.g., from 'catalytic activity' https://www.ebi.ac.uk/QuickGO/term/GO:0003824, 'membrane' https://www.ebi.ac.uk/QuickGO/term/GO:0016020, or 'binding' https://www.ebi.ac.uk/QuickGO/term/GO:0005488 )

* Append child term if root, parent and child terms in molecular_function name space (not cellular component or biological process) and if child "is_a" parent (not "regulates" or "part_of")
* Append child term  if root, parent and child term in cellular_component name space (not molecular_function or biological process) and if child "is_a" or "part_of" parent (not "regulates")
* Not designed for biological_process GO.

Contains code from https://github.com/nstrodt/UDSMProt/blob/master/code/utils/evaluate_deepgoplus.py 
which was adapted from https://github.com/bio-ontology-research-group/deepgoplus/blob/master/evaluate_deepgoplus.py
published under the BSD 3-Clause "New" or "Revised" License: https://github.com/bio-ontology-research-group/deepgoplus/blob/master/LICENSE
We cite the aforementioned license below the code.
"""

import os, random, json
from collections import deque
from pathlib import Path

class Ontology(object):

    def __init__(self, filename='./data-2016/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)

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
                    # Append if root, parent and child terms in molecular_function name space (not cellular component or biological process) and if child "is_a" parent (not "regulates" or "part_of")
                    if (('molecular_function' in self.ont[term_id]['namespace']) and ('molecular_function' in self.ont[t_id]['namespace']) and ('molecular_function' in self.ont[ch_id]['namespace']) and (t_id in self.ont[ch_id]['is_a'])):
                        q.append(ch_id)
                    # Append if root, parent and child terms in cellular_component name space (not molecular_function or biological process) and if child "is_a" or "part_of" parent (not "regulates")
                    elif ( ('cellular_component' in self.ont[term_id]['namespace']) and ('cellular_component' in self.ont[t_id]['namespace']) and ('cellular_component' in self.ont[ch_id]['namespace']) and ((t_id in self.ont[ch_id]['is_a']) or (t_id in self.ont[ch_id]['part_of'])) ):
                        q.append(ch_id)
                    else: continue

        return term_set

if __name__ == "__main__":
    
    # Select either GO-temporalsplit or GO-CAFA3 dataset
    if 1: go_path = Path("data-2016") # clas_go_deepgoplus_temporalsplit
    else: go_path = Path("data-cafa") # clas_go_deepgoplus_cafa

    # Download files incl. Gene-Ontology .obo-file from DeepGO website
    if not go_path.exists():
        if not Path(str(go_path) + ".tar.gz").exists():
            os.system("wget https://deepgo.cbrc.kaust.edu.sa/data/" + str(go_path) + ".tar.gz")
        os.system("mkdir " + str(go_path))
        os.system("tar xvzf " + str(go_path) + ".tar.gz -C " + str(go_path))

    go_rels = Ontology(str(go_path/"go.obo"), with_rels=True)
    
    SELECTED_GO_TERM = json.load(open('parameters.json'))["SELECTED_GO_TERM"]
    (term_set) = [go_rels.get_term_set(SELECTED_GO_TERM)]
    
    # Show a few random examples of child terms of selected GO term on QuickGO to inspect and check
    for i in range(0, 10): os.system(f"firefox https://www.ebi.ac.uk/QuickGO/term/{random.sample(tuple(term_set[0]),1)[0]}")

    
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
