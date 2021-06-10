
# How to run

## For single dataset generation aka without multitask style transfer (ptb/yelp/snli)

Here is an example on how to download, train the Yelp dataset.

    bash download_yelp.sh
    python3 train.py -dataset yelp

For sample generation and inference.

    python3 inference_yelp.py -n 1 -c 'path/to/checkpoint.pytorch' 

## For multitask generation aka with general style transfer (snli+yelp)

Here is an example on how to download, train the multitask SNLI+Yelp dataset for general style

	bash download_yelp.sh
	bash download_snli.sh
    python3 train.py -dataset multitask

For sample generation and inference.

    python3 inference_multitask.py -n 1 -c 'path/to/checkpoint.pytorch' 

For style transfer between two sentences

	python3 style_transfer.py -n 1 -c 'path/to/checkpoint.pytorch' -p 'path/to/model_params.json'

To enable the encoder classifier please use --hspace True while training any model. To enable Gram Schimdt Orthogonalisation of Cntent Z and Style Z vectors please use ---ortho True while training. Self attention on the encoder can also be added using --attention.

		