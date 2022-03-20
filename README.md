# Event-Coreference-Resolution-for-Contentious-Politics-Events
This repo provides the scripts to reproduce models and scores for ***Event Corefernce Resolution for Contentious Politics Events*** paper.

In order to run the scripts, first create an environment and activate;
```
conda create --name ecr4 python=3.8 && conda activate ecr4
```
Then install the required packages;
```
conda install --file requirements.txt
```
Before moving forward to train or test models, first you need the download the data and code repos.
The dataset is shared in [`coreference_data`](https://github.com/emerging-welfare/coreference_data) repo of our project, __emerging-welfare__.
```
git clone https://github.com/emerging-welfare/coreference_data.git
git clone https://github.com/emerging-welfare/ECR4-Contentious-Politics.git
```

To train a model from scratch, go to `scripts` folder and run `train.py`. For example you can run the following command in order to train a model for `english`, `spanish` and `portuguese` mix dataset;
```
python train.py --train_dataset_paths en_es_pr --dev_dataset_paths en_es_pr --test_dataset_paths en_es_pr
```
A model checkpoint will be saved to `models`. Then you can test the model in any language with following line of code;
```
python test.py --model_checkpoint ../models/<checkpoint_name>.ckpt --test_dataset_paths en_pr
```
You will be able to see results default and fine-tuned threshold for validation and test set of each language, respectively.