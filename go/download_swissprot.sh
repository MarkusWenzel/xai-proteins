#!/bin/bash

cd data

# Download Swissprot 2017 03
if [ ! -f uniprot_sprot-only2017_03.tar.gz ]; then
    echo "Downloading Swissprot 2017 03 dataset"
    curl https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2017_03/knowledgebase/uniprot_sprot-only2017_03.tar.gz -o uniprot_sprot-only2017_03.tar.gz
else
    echo "data/uniprot_sprot-only2017_03.tar.gz already exists."
fi

# Extract and rename Swissprot
if [ ! -f uniprot_sprot_2017_03.xml ]; then
    echo "Extracting data"
    tar -xf uniprot_sprot-only2017_03.tar.gz uniprot_sprot.xml.gz
    gunzip uniprot_sprot.xml.gz
    mv uniprot_sprot.xml uniprot_sprot_2017_03.xml
else
    echo "data/uniprot_sprot_2017_03.xml already exists."
fi

cd ..