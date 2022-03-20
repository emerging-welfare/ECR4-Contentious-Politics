from datasetparser import *
from evaluator import *

import torch
import torch.nn as nn
from torch.optim import AdamW 
from torch.utils.data import DataLoader
from transformers import AutoModel,AutoConfig

from pytorch_lightning import seed_everything
from pytorch_lightning.core.lightning import LightningModule


class EventCoreferenceNet(LightningModule):

    def __init__(self,

                 model_name: str,

                 train_dataset_paths: list,
                 dev_dataset_paths: list,
                 test_dataset_paths: list,
                 exclude_single_trigs: bool,
                 exclude_first_sents: bool,
                 
                 hidden_unit: int,
                 dropout: int,
                 batch_size: int,
                 learning_rate: float,
                 eps: float,
                 seed: int,
                 
                 experiment_id: str = "000000000000"
                 ):

        super(EventCoreferenceNet, self).__init__()
        self.seed = seed
        seed_everything(self.seed)

        self.save_hyperparameters()
        
        self.experiment_id = experiment_id

        self.model_name = model_name

        self.train_dataset_paths = train_dataset_paths
        self.dev_dataset_paths = dev_dataset_paths
        self.test_dataset_paths = test_dataset_paths
        self.exclude_single_trigs = exclude_single_trigs
        self.exclude_first_sents = exclude_first_sents
        self.dataset = EMWCoreferenceDataset(
                                                model_name = self.model_name,
                                                train_dataset_paths = self.train_dataset_paths,
                                                dev_dataset_paths = self.dev_dataset_paths,
                                                test_dataset_paths = self.test_dataset_paths, 
                                                exclude_single_trigs = self.exclude_single_trigs,
                                                exclude_first_sents = self.exclude_first_sents,  
                                            )

        self.train_tensor = self.dataset.train_tensor_dataset
        self.dev_tensor = self.dataset.dev_tensor_dataset
        self.test_tensor = self.dataset.test_tensor_dataset

        self.evaluator = CoreferenceEvaluator(threshold = 0.5,
                                              experiment_id = self.experiment_id)
        self.baseline_scores()

        
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.transformer = AutoModel.from_pretrained(self.model_name)

        self.hidden_layer1 = nn.Linear(self.config.hidden_size*3, hidden_unit)
        self.dropout = nn.Dropout(p = dropout)                 
        self.hidden_layer2 = nn.Linear(hidden_unit, 1)          
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.lossfn = nn.BCELoss()
        self.batch_size = batch_size
        self.learning_rate  = learning_rate
        self.eps = eps
    

    def baseline_scores(self):
        (i,a,t,l),p = self.dev_tensor.tensors, self.dataset.dev_pairs
        o = torch.ones_like(l)
        for metric,score in self.evaluator.evaluate(p,o,l,"dev_baseline").items():
            self.log(metric,score)

        (i,a,t,l),p = self.test_tensor.tensors, self.dataset.test_pairs
        o = torch.ones_like(l)
        for metric,score in self.evaluator.evaluate(p,o,l,"test_baseline").items():
            self.log(metric,score)

    def predict_step(self, batch, batch_idx, dataloader_idx: int = None):
        i,a,t,l = batch
        return self(i,a,t)


    def forward(self, input_ids, attention_mask,trigger_indexes):
        
        X = self.transformer(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
        X = self.dropout(X)
        INNERP = torch.matmul(trigger_indexes,X)
        INNERP = INNERP / trigger_indexes.sum(dim=2).view(-1,2,1)
        X = torch.cat((INNERP.view(input_ids.shape[0],-1),torch.prod(INNERP,dim=1)),dim=1)
        X = self.dropout(X)
        X = self.hidden_layer1(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.hidden_layer2(X)
        O = self.sigmoid(X)
        return O
    
    def train_dataloader(self):
        return DataLoader(self.train_tensor, batch_size = self.batch_size, shuffle = True, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.dev_tensor, batch_size = self.batch_size, shuffle = False , num_workers = 4)

    def test_dataloader(self):
        return DataLoader(self.test_tensor, batch_size = self.batch_size, shuffle = False , num_workers = 4)

    def training_step(self, batch, batch_idx):
        ids,masks,idx,labels = batch
        outputs = self(ids,masks,idx)
        loss = self.lossfn(outputs, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ids,masks,idx,labels = batch
        outputs = self(ids,masks,idx)
        loss = self.lossfn(outputs, labels)
        self.log('dev_loss', loss)

        return outputs,labels

    def validation_epoch_end(self, validation_step_outputs):
        pairs = self.dataset.dev_pairs
        outputs = torch.cat([o[0] for o in validation_step_outputs])
        labels = torch.cat([o[1] for o in validation_step_outputs])
        
        for metric,score in self.evaluator.evaluate(pairs,outputs,labels,"dev").items():
            self.log(metric,score)


    def test_step(self, batch, batch_idx):
        ids,masks,idx,labels = batch
        outputs = self(ids,masks,idx)

        return outputs,labels
        
    def test_epoch_end(self, test_step_outputs):
        pairs = self.dataset.test_pairs
        outputs = torch.cat([test_step_output for test_step_output,test_step_label in test_step_outputs])
        labels  = torch.cat([test_step_label for test_step_output,test_step_label in test_step_outputs])

        for metric,score in self.evaluator.evaluate(pairs,outputs,labels,"test").items():
            self.log(metric,score)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),lr = self.learning_rate, eps = self.eps, weight_decay=1e-1)
        return optimizer
