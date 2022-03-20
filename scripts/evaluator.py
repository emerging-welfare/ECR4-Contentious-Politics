import json
import subprocess
import torchmetrics
import time
import torch

class CoreferenceEvaluator:
    
    def __init__(self,
                 threshold: float,
                 experiment_id: str
                 ):
        
        self.experiment_id = experiment_id
        
        self.threshold = threshold
        self.accuracy = torchmetrics.Accuracy(threshold = self.threshold)
        self.precision = torchmetrics.Precision(threshold = self.threshold, average = "macro", num_classes = 1, multiclass = False)
        self.recall = torchmetrics.Recall(threshold =  self.threshold, average = "macro", num_classes = 1, multiclass = False)
        self.f1score = torchmetrics.classification.f_beta.F1Score(threshold = self.threshold, average = "macro", num_classes = 1, multiclass = False)

        self.metrics = {"accuracy": self.accuracy,
                        "precision": self.precision,
                        "recall" : self.recall,
                        "f1score": self.f1score}
                 

    def links(self, pairs, outputs, labels, set_):

        self.classification_metrics = {f"{set_}_{name}":metric(outputs,labels.int()).tolist() for name,metric in self.metrics.items()}
        cc = (labels.flatten()== 1.0).nonzero(as_tuple=False).flatten()
        c = ((outputs.flatten()>self.threshold).float() == 1.0).nonzero(as_tuple=False).flatten()

        key = list(map(pairs.__getitem__,cc))
        response = list(map(pairs.__getitem__,c))

        return key,response,set_
    
    def return_metrics(self,key,response,set_):
        result_file = f"{self.experiment_id}_results.txt"

        with open(f"{self.experiment_id}_gold.json","w") as f:
            json.dump({"type":"graph","mentions":list(set([i for j in key for i in j])),"links":key},f)
        with open(f"{self.experiment_id}_sys.json","w") as f:
            json.dump({"type":"graph","mentions":list(set([i for j in response for i in j])),"links":response},f)

        subprocess.run(["scorch",f"{self.experiment_id}_gold.json",
                                 f"{self.experiment_id}_sys.json",result_file])
        
        with open(result_file,"r") as f:
            lines = f.readlines()
            results = [[float(j.split("=")[1]) for j in i.split(":")[1].strip().split("\t") ]for i in lines[:-1]]
            results = results + [float(lines[-1].split(":")[1].strip())]

            muc = dict(zip([set_+"_"+"muc_precision",set_+"_"+"muc_recall",set_+"_"+"muc_f1"],results[0]))
            b_cubed = dict(zip([set_+"_"+"b_cubed_precision",set_+"_"+"b_cubed_recall",set_+"_"+"b_cubed_f1"],results[1]))
            ceaf_e = dict(zip([set_+"_"+"ceaf_e_precision",set_+"_"+"ceaf_e_recall",set_+"_"+"ceaf_e_f1"],results[3]))
            blanc = dict(zip([set_+"_"+"blanc_precision",set_+"_"+"blanc_recall",set_+"_"+"blanc_f1"],results[4]))
            conll = {set_+"_"+"conll":results[-1]}

            metrics = {**self.classification_metrics,**muc,**b_cubed,**ceaf_e,**blanc,**conll}

        subprocess.run(["rm",
                            f"{self.experiment_id}_gold.json",
                            f"{self.experiment_id}_sys.json",
                            f"{self.experiment_id}_results.txt"])

        return metrics
    
    def evaluate(self,pairs,outputs,labels,set_):
        return self.return_metrics(*self.links(pairs, outputs.cpu(), labels.cpu(), set_))