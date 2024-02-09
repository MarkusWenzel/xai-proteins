# Insights into the inner workings of transformer models for protein function prediction

## About 

Finetuning pretrained universal protein language models to downstream tasks provides large benefits in protein function prediction.
The used neural networks are, at the same time, notorious for having often millions and sometimes billions of trainable parameters.
Therefore, it can be very difficult to interpret the decision making logic or strategy of these complex models. 

Consequently, explainable machine learning is starting to gain traction in the field of proteomics too.
We are exploring how explainability methods can help to shed light into the inner workings of transformers for protein function prediction.

Attribution methods, such as integrated gradients, make it possible to identify those features in the input space that the model apparently focuses on, because these features turn out to be relevant for the final classification decision of the model.
We extended integrated gradients such that latent representations inside of transformers can be inspected too (separately for each head and layer).

To find out if the identified relevant sequence regions match expectations informed by knowledge from biology or chemistry, we combined this method with a subsequent statistical analysis across proteins where we correlated the obtained relevance with annotations of interest from sequence databases.
In this way, we identified heads inside of the transformer architecture that are specialized for specific protein function prediction tasks. 

The two folders of this repository are dedicated to the explainability analysis for the Gene Ontology (GO) term and Enzyme Commission (EC) number prediction (see the [GO](go/README.md) and [EC](ec/README.md) README files) .

## Publication

You find more information in our article:

Markus Wenzel, Erik Grüner, Nils Strodthoff (2024). [Insights into the inner workings of transformer models for protein function prediction](https://doi.org/10.1093/bioinformatics/btae031), _Bioinformatics_, btae031.
    
    @article{10.1093/bioinformatics/btae031, author = {Wenzel, Markus and Grüner, Erik and Strodthoff, Nils}, title = "{Insights into the inner workings of transformer models for protein function prediction}", journal = {Bioinformatics}, pages = {btae031}, year = {2024}, month = {01}, issn = {1367-4811}, doi = {10.1093/bioinformatics/btae031}, url = {https://doi.org/10.1093/bioinformatics/btae031}}

## Related works

If you are interested in this topic, you are welcome to have a look at our related papers:
* Nils Strodthoff, Patrick Wagner, Markus Wenzel, and Wojciech Samek (2020). [UDSMProt: universal deep sequence models for protein classification](https://doi.org/10.1093/bioinformatics/btaa003). _Bioinformatics_, 36(8), 2401–2409.
* Johanna Vielhaben, Markus Wenzel, Wojciech Samek, and Nils Strodthoff (2020). [USMPep: universal sequence models for major histocompatibility complex binding affinity prediction](https://doi.org/10.1186/s12859-020-03631-1), _BMC Bioinformatics_, 21, 1-16.

## Datasets

* EC datasets by Nils Strodthoff, Patrick Wagner, Markus Wenzel, and Wojciech Samek (2020). [UDSMProt: universal deep sequence models for protein classification](https://doi.org/10.1093/bioinformatics/btaa003). _Bioinformatics_, 36(8), 2401–2409.
* GO datasets ([data-2016.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/data-2016.tar.gz), [data-cafa.tar.gz](https://deepgo.cbrc.kaust.edu.sa/data/data-cafa.tar.gz)) by Maxat Kulmanov and Robert Hoehndorf (2020). [DeepGOPlus: improved protein function prediction from sequence](https://doi.org/10.1093/bioinformatics/btz595). _Bioinformatics_, 36(2), 422–429.

EC and GO data were preprocessed as detailed on <https://github.com/nstrodt/UDSMProt> with <https://github.com/nstrodt/UDSMProt/blob/master/code/create_datasets.sh>, resulting in six files for EC40 and EC50 on levels L0, L1, and L2, and in two files for GO "2016" (a.k.a. "temporalsplit") and GO "CAFA3". Preprocessed data can be accessed [here (EC)](https://datacloud.hhi.fraunhofer.de/s/odHaAoLyTyq4GjL) and [here (GO)](https://datacloud.hhi.fraunhofer.de/s/px6CMpMp3HzAtjE).

## Authors

Markus Wenzel, Erik Grüner, Nils Strodthoff (2024)



