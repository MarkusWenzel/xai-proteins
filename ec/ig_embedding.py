from integrated_gradient_helper import run_embedding_all_proteins,ig_setup
level=1
ec_class=50
model, output_dir, prot_df = ig_setup(level, ec_class, data_path="data/ec50_level1/test.json")
run_embedding_all_proteins(prot_df, model, output_dir)

