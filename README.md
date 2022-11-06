PLEASE READ:
This code is NOT usable so please avoid trying to build using it. There are issues with the underlying RNN code whcih prevents the model from performing even a simple reconstruction task properly. The issue does not seem to be with the code related to the additional discriminators and auxillary losses. I have not found time to fix this and moved along to other things quite some time back. If you wish to work over this and correct it, please consider looking into fixing the underlying RNN. I am writing this to avoid any wastage of time for researchers, considering that I have been emailed quite a bit about if this is usable.


This is a in-progress pytorch replication of the paper https://www.aclweb.org/anthology/P19-1041/. There seems to be no other working pytorch replications of this paper, please feel free to contribute and work towards its completion.
  

# Instructions
Here is an example on how to download, train and infer from the Yelp dataset

    bash ./bash-scripts/download_yelp.sh
    python3 train_rep.py
    python3 inference.py -n 1 -c 'path/to/checkpoint.pytorch' -p 'path/to/model_params.json'

# Directory structure

 - ./dataset_preproc_scripts/yelp.py: is used to preprocess the raw data, create vocab, and store the preprocess data in json form in ./data/yelp. For testing, I am only using a small portion of the whole dataset. To change this, change the variables in __init__() of yelp.py. Everytime a change is made, you need to run bash_delete_yelp_preproc.sh to delete the old data so that new vocabs and processed files are created
 - model_rep.py: contains the torch models defintion,
 - train_rep.py: contains the hyper paramerers, dataloading, model creation, loss definition, training, testing, logging and saving of checkpoints


    
