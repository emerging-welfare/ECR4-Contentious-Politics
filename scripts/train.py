from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import argparse
import json
import time

from model import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXPERIMENT_ID = str(time.time()).replace(".","")
SPLITS_PATH = "../../coreference_data/splits/json/"


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--c', required=True)
    args = parser.parse_args()
    parameters = json.load(open(args.c,"r"))

    print("\n\n",parameters,"\n\n")

    return parameters


def main(args):

    args.train_dataset_paths = [os.path.join(SPLITS_PATH,f"{lang}/{lang}_train.json") for lang in args.train_dataset_paths.split("_")]
    args.dev_dataset_paths = [os.path.join(SPLITS_PATH,f"{lang}/{lang}_dev.json") for lang in args.dev_dataset_paths.split("_")]
    args.test_dataset_paths = [os.path.join(SPLITS_PATH,f"{lang}/{lang}_test.json") for lang in args.test_dataset_paths.split("_")]


    earlystop_callback = EarlyStopping(
                                        monitor = "dev_conll", 
                                        min_delta = 0.00,
                                        patience = 3,
                                        mode = 'max'
                                        )
    checkpoint_callback = ModelCheckpoint(
                                        monitor = 'dev_conll', 
                                        dirpath = '../models/',
                                        filename = EXPERIMENT_ID,
                                        mode = 'max'
                                        )


    model =  EventCoreferenceNet(
                                    experiment_id = EXPERIMENT_ID,
                                    model_name = args.model_name,
                                    train_dataset_paths = args.train_dataset_paths,
                                    dev_dataset_paths = args.dev_dataset_paths,
                                    test_dataset_paths = args.test_dataset_paths,
                                    exclude_single_trigs = args.exclude_single_trigs,
                                    exclude_first_sents = args.exclude_first_sents,
                                    
                                    hidden_unit = args.hidden_unit,
                                    dropout = args.dropout,
                                    batch_size = args.batch_size,
                                    learning_rate = args.learning_rate,
                                    eps = args.eps,
                                    seed = args.seed
                                )


    trainer = Trainer(
                      gpus = [args.gpu], 
                      auto_lr_find = True,
                    #   limit_train_batches = 0.1,
                    #   max_epochs = args.num_train_epochs,
                      callbacks = [earlystop_callback,
                                   checkpoint_callback]
                    )

    # trainer.tune(model)
    trainer.fit(model)
    trainer.test(model)

    print("\nTraining is done model saved at: ../models/{EXPERIMENT_ID}.ckpt")

if __name__ == "__main__":

    SPLITS_PATH = "../../coreference_data/splits/json/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_paths",
                        default = "en_pr",
                        type = str)
    parser.add_argument("--dev_dataset_paths",
                        default = "en_pr",
                        type = str)
    parser.add_argument("--test_dataset_paths",
                        default = "en_pr",
                        type = str)

    parser.add_argument("--exclude_single_trigs", default = False, type = bool)
    parser.add_argument("--exclude_first_sents", default = False, type = bool)

    parser.add_argument("--seed", default = 22, type = int)
    parser.add_argument("--model_name", type = str, default = 'bert-base-uncased') 
    parser.add_argument("--learning_rate", type = float, default = 5e-06)
    parser.add_argument("--num_train_epochs", type = int, default = 1)
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--eps", type = float, default = 1e-6)
    parser.add_argument("--dropout", type = float, default = 0.3)
    parser.add_argument("--hidden_unit", default = 128, type = int)
    parser.add_argument("--gpu", default = 0, type = int)

    args = parser.parse_args()
    
    main(args)
