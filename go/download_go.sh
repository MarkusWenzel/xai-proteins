# Gene ontology obo files
if [ ! -f data-2016.tar.gz ]; then
    echo "Downloading Gene Ontology version: temporal split"
    wget https://deepgo.cbrc.kaust.edu.sa/data/data-2016.tar.gz
    mkdir data-2016/
    tar xvzf data-2016.tar.gz -C data-2016/

else
    echo "GO folder temporal split already exists."
fi

if [ ! -f data-cafa.tar.gz ]; then
    echo "Downloading Gene Ontology version: CAFA3 split"
    wget https://deepgo.cbrc.kaust.edu.sa/data/data-cafa.tar.gz
    mkdir data-cafa/
    tar xvzf data-cafa.tar.gz -C data-cafa/

else
    echo "GO folder CAFA3 split already exists."
fi
