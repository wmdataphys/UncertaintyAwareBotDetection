# Twitter Bot classification with Bayesian Neural Network using the BLOC framework

We utilize two models, both of which are under development.

1. A Bayesian Neural Network with Flow approximated posteriors.
2. A simple Deep Neural Network.

We want to compare these two.

To train the bayesian network you can run:

python train_mnf.py --config config/default_config.json

To train the DNN you can run:

python train_mlp.py --config config/default_config.json

In both cases, make sure you update the experiment name (first field of json file) as a directory will be created with this name all models will be saved here.

By default, the evaluation scripts (run_inference.py, make_plots.py) will evaluate the Bayesian network. You can add a command line flag to also evaluate the the MLP.

For example:

python run_inference.py --config config/default_config.json --mlp_eval 1

This will run the MLP evaluation. Make sure that you have specified paths to the DNN and Bayesian Network under the Inference field in the config.json file.
You can also run the plotting code in a similar fashion (assuming you have ran inference with --mlp_eval 1).

For example:

python make_plots.py --config config/default_config.json --mlp_eval 1

This will create plots for both the Bayesian network and the MLP. You can control the folder where these are generated in the config file. Also make sure you specify the path to the out_file in the .json files for inference.



### Related Publications 
- BLOC: https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-023-00410-9

- ELUQuant: https://iopscience.iop.org/article/10.1088/2632-2153/ad2098/meta

- https://arxiv.org/pdf/1703.04977
