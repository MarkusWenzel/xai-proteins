import sys
from integrated_gradient_helper import run_attention_attribution_all_proteins

if __name__ == "__main__":
    try: h = int(sys.argv[1])
    except IndexError: h = 1
    run_attention_attribution_all_proteins(n_steps=500, hooks=[h], data_path="data/clas_go_deepgoplus_temporalsplit/test.json")

