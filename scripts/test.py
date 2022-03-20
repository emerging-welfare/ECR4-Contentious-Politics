
from pytorch_lightning import Trainer
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1
from pytorch_lightning import seed_everything

import argparse
import json

from model import *

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EXPERIMENT_ID = str(time.time()).replace(".","")
SPLITS_PATH = "../../coreference_data/splits/json/"

def finetune_threshold(dev_pairs, dev_predictions, dev_labels, dev_set_name):
    max_conll, max_threshold, max_results = 0, 0, None
    for threshold in [i/100 for i in range(1,100)]:
        result = CoreferenceEvaluator(threshold = threshold, experiment_id = EXPERIMENT_ID).evaluate(dev_pairs, dev_predictions, dev_labels, dev_set_name)
        if result[f"{dev_set_name}_conll"] >= max_conll:
            max_conll = result[f"{dev_set_name}_conll"]
            max_threshold = threshold
            max_results = result
    return max_conll, max_threshold, max_results


def print_results(pairs, predictions, labels, set_name, threshold):
    print(f"\n{set_name} RESULTS WITH THRESHOLD: {threshold}".upper())
    evaluator = CoreferenceEvaluator(threshold = threshold, experiment_id = EXPERIMENT_ID)
    print(json.dumps(evaluator.evaluate(pairs, predictions, labels, set_name), indent=3),"\n")


def evaluate(dev, test):
    
    [dev_pairs, dev_predictions, dev_labels, dev_set_name],[test_pairs, test_predictions, test_labels, test_set_name] = dev,test
    
    print_results(dev_pairs, dev_predictions, dev_labels, dev_set_name,0.5)
    print_results(test_pairs, test_predictions, test_labels, test_set_name,0.5)

    max_conll, max_threshold, max_results = finetune_threshold(dev_pairs, dev_predictions, dev_labels, dev_set_name)
    print(f"\nBEST THRESHOLD FOR {dev_set_name.split('_')[0].upper()}: {max_threshold}\n")
    print_results(dev_pairs, dev_predictions, dev_labels, dev_set_name,max_threshold)
    print_results(test_pairs, test_predictions, test_labels, test_set_name,max_threshold)


def create_test_dataloaders(model,args):

    datasets = []
    for test_dataset_path in args.test_dataset_paths:

        dataset = EMWCoreferenceDataset(
                                        model_name = model.model_name,
                                        train_dataset_paths = [test_dataset_path.replace("test","dev")],
                                        dev_dataset_paths = [test_dataset_path.replace("test","dev")],
                                        test_dataset_paths = [test_dataset_path], 
                                        exclude_single_trigs = model.exclude_single_trigs,
                                        exclude_first_sents = model.exclude_first_sents,  
                                        )
        datasets.append(dataset)

    return datasets                   

def main(args):

    args.test_dataset_paths = [os.path.join(SPLITS_PATH,f"{lang}/{lang}_test.json") for lang in args.test_dataset_paths.split("_")]

    model =  EventCoreferenceNet.load_from_checkpoint(args.model_checkpoint)
    trainer = Trainer(gpus = [args.gpu])

    
    datasets = create_test_dataloaders(model, args)

    test_dataloaders = [DataLoader(test_dataset.test_tensor_dataset,
                                   batch_size = model.batch_size,
                                   num_workers = 4) for test_dataset in datasets]

    dev_dataloaders = [DataLoader(dev_dataset.dev_tensor_dataset,
                                   batch_size = model.batch_size,
                                   num_workers = 4) for dev_dataset in datasets]

    test_dataset_names = [path.split("/")[-1].split(".")[0] for path in args.test_dataset_paths]
    dev_dataset_names = [path.replace("test","dev") for path in test_dataset_names]

    test_predictions = trainer.predict(model = model,
                                       dataloaders = test_dataloaders,
                                       return_predictions = True)
                                       
    test_predictions = map(torch.cat, test_predictions) if len(args.test_dataset_paths)>1 else [torch.cat(test_predictions)]

    dev_predictions =  trainer.predict(model = model,
                                                     dataloaders = dev_dataloaders,
                                                     return_predictions = True)

    dev_predictions = map(torch.cat, dev_predictions) if len(args.test_dataset_paths)>1 else [torch.cat(dev_predictions)]

    
    for _dataset, _dev_dataset_name, _test_dataset_name, _dev_prediction, _test_prediction in zip(datasets, dev_dataset_names, test_dataset_names, dev_predictions, test_predictions):


        evaluate([_dataset.dev_pairs,
                  _dev_prediction,
                  torch.tensor(_dataset.dev_labels),
                  _dev_dataset_name],
                 [_dataset.test_pairs,
                  _test_prediction,
                  torch.tensor(_dataset.test_labels),
                  _test_dataset_name])


if __name__ == "__main__":

    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint',
                        required=True,
                        type = str)
    parser.add_argument('--test_dataset_paths',
                        default = "es_pr",
                        type = str)
    parser.add_argument('--gpu',
                        default = 0,
                        type = int)
    args = parser.parse_args()

    main(args)
    
        

        