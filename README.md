# RNNs for 21cm lightcones
Code for reproducing models and training setup for the article:

>Prelogović D., Mesinger A., Murray S., Fiameni G., Gillet N. 
>_Machine learning astrophysics from 21 cm lightcones: impact of network architectures and signal contamination_
>[[MNRAS]](https://doi.org/10.1093/mnras/stab3215) [[arXiv]](https://arxiv.org/abs/2107.00018)

## Instructions
`run.py` scripts sets up and runs the training, `saliency_map.py` computes a saliency used in the article.
```
$ python run.py -h
usage: Model Fit [-h] [--database {default,ska,horizon_wedge}]
                 [--X_fstring X_FSTRING] [--normalize_X {0,1}]
                 [--N_realizations N_REALIZATIONS] [--model MODEL]
                 [--model_type MODEL_TYPE]
                 [--HyperparameterIndex HYPERPARAMETERINDEX]
                 [--simple_run {0,1}] [--seed SEED] [--warmup WARMUP]
                 [--LR_correction {0,1}] [--mixed_precision {0,1}]
                 [--gpus GPUS] [--workers WORKERS] [--eager {0,1}]
                 [--epochs EPOCHS] [--max_epochs MAX_EPOCHS]
                 [--data_location DATA_LOCATION]
                 [--saving_location SAVING_LOCATION] [--tensorboard {0,1}]
                 [--logs_location LOGS_LOCATION] [--file_prefix FILE_PREFIX]
                 [--verbose {0,1,2}]

optional arguments:
  -h, --help            show this help message and exit
  --database {default,ska,horizon_wedge}
                        Database to be used for the run. (default: default)
  --X_fstring X_FSTRING
                        f-string defining filename for the database.
  --normalize_X {0,1}   Normalize input or not. (default: 1)
  --N_realizations N_REALIZATIONS
                        Number of noise realizations to train on. (default:
                        10)
  --model MODEL         Keras model to be used for training. (default:
                        RNN.SummarySpace3D)
  --model_type MODEL_TYPE
                        RNN or CNN model, inferred from `model` by default.
  --HyperparameterIndex HYPERPARAMETERINDEX
                        Combination index in the list of all possible
                        hyperparameters. (default: 0)
  --simple_run {0,1}    If 1, runs hyperparameters defined in
                        `py21cnn.hyperparamters.HP_simple`. Else, runs
                        hyperparameters defined with `HyperparameterIndex`.
                        (default: 0)
  --seed SEED           By default -1, which picks a seed randomly.
  --warmup WARMUP       In the case of multi-gpu training, for how many epochs
                        to linearly increase learining rate? (default: 0)
  --LR_correction {0,1}
                        In the case of multi-gpu training, should learning
                        rate be multiplied by number of GPUs? (default: 0)
  --mixed_precision {0,1}
                        Either to use mixed precision or not.
  --gpus GPUS           Number of GPUs. (default: 1)
  --workers WORKERS     Number of cpu workers for data loading. (default: 24)
  --eager {0,1}         Run the code eagerly? (default: 1)
  --epochs EPOCHS       Number of epochs to train. (default: 200)
  --max_epochs MAX_EPOCHS
                        In the case of re-training, sets an upper bound on the
                        total number of epochs.By default, the same as
                        `epochs`.
  --data_location DATA_LOCATION
                        Defaults to `data/`
  --saving_location SAVING_LOCATION
                        Defaults to `models/`
  --tensorboard {0,1}   Creating TensorBoard logs during the run? (default: 1)
  --logs_location LOGS_LOCATION
                        Defaults to `logs/`
  --file_prefix FILE_PREFIX
                        File prefix to all outputs of the program - saved
                        models, logs. (default: `None`)
  --verbose {0,1,2}     Verbosity of the stdout. (default: 2)
```
## Citation
To cite this work:
```
@article{prelogovic2022,
  title         = "{Machine learning astrophysics from 21 cm lightcones: impact of network architectures and signal contamination}",
  author        = {Prelogovi{\'c}, David and Mesinger, Andrei and Murray, Steven and Fiameni, Giuseppe and Gillet, Nicolas},
  journal       = {Monthly Notices of the Royal Astronomical Society},
  volume        = {509},
  number        = {3},
  pages         = {3852--3867},
  year          = {2022},
  doi           = {10.1093/mnras/stab3215},
  publisher     = {Oxford University Press}
}
```
