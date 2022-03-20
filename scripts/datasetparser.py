import json
from itertools import combinations

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset
import torch

import random


class Doc:
    def __init__(self,
                 sentence_dict: dict,
                 tokenizer: AutoTokenizer,
                 policy,
                 exclude_single_trigs,
                 exclude_first_sents):
        self.policy = policy
        self.exclude_single_trigs = exclude_single_trigs
        self.exclude_first_sents = exclude_first_sents
        
        self.tokenizer = tokenizer
        self.filename = sentence_dict["filename"]
        self.tokens = [[j.lower() for j in i] for i in sentence_dict["tokens"]]
        self.sentences = [" ".join(t) for t in self.tokens]
        self.spans  = sentence_dict["spans"]
        
        self.create_token_charseqs()
        self.parse()
        self.create_pairs()
    
    def create_token_charseqs(self):
        self.charseqs = []
        
        for sentence in self.tokens:
            sentence_charseq = []
            start = 0
            
            for token in sentence:
                sentence_charseq.append([start,start+len(token)])
                start+=len(token)+1
            self.charseqs.append(sentence_charseq)
    
    def check_range_in_range(self, trigger_charseq, token_charseq):
        if token_charseq != [0,0]:
            return all(e in range(*trigger_charseq) for e in range(*token_charseq))
        else:
            return False
        
    def find_trigger_tokenized_index(self, trigger_dict):
        sentence = trigger_dict["sentence"]
        charseq  = trigger_dict["trigger_charseq"]
        tokenizer_output = self.tokenizer(sentence,
                                         padding = True,
                                         return_token_type_ids = False,
                                         return_tensors = "pt",
                                         return_special_tokens_mask = True, 
                                         return_offsets_mapping = True)
        
        offset_mapping = tokenizer_output["offset_mapping"][0].tolist()
        trigger_tokenized_index = [1 if self.check_range_in_range(charseq,o) else 0 for o in offset_mapping]
        
        trigger_dict["trigger_tokenized_index"] = trigger_tokenized_index
        trigger_dict["tokenized_output"] = tokenizer_output
        return trigger_dict
    
    def which_event(self, event_names, event_counts, policy, exclude_single_trigs):
        if policy == "most_common" and exclude_single_trigs == True:    
            event_counts_ = {en:event_counts[en] for en in event_names}
            number_of_occur = list(event_counts_.values())
            events = [event_name for event_name in event_names if event_counts_[event_name] == max(number_of_occur) != 1]
            return sorted(events)[0] if events else None
        
        elif policy == "most_common" and exclude_single_trigs == False:
            if len(set(event_names))>1:
                event_counts_ = {en:event_counts[en] for en in event_names}
                number_of_occur = list(event_counts_.values())
                events = [event_name for event_name in event_names if event_counts_[event_name] == max(number_of_occur)]
                return sorted(events)[0]
            else:
                return event_names[0]
                
    def parse(self):
        self.event_counts = {}
        for sentence_index, markables in enumerate(self.spans):
            if self.exclude_first_sents and sentence_index == 0:
                continue
            sentence_name = self.filename+"|#|#|"+str(sentence_index)
            for markable_index, markable in enumerate(markables):
                token_index, event_attr, event_names = markable
                
                if event_attr == "trigger":
                    trigger_name = sentence_name+"|#|#|"+"_".join(map(str,token_index))
                    for event_name in event_names:
                        event_name = self.filename+"|#|#|"+event_name

                        self.event_counts.setdefault(event_name,0)
                        self.event_counts[event_name] += 1
                        
        self.triggers = {}
        self.events = {}
        for sentence_index, markables in enumerate(self.spans):
            if self.exclude_first_sents and sentence_index == 0:
                continue
            sentence_name = self.filename+"|#|#|"+str(sentence_index)
            for markable_index, markable in enumerate(markables):
                token_index, event_attr, event_names = markable
                
                if event_attr == "trigger":
                    changed_event_names = [self.filename+"|#|#|"+en for en in event_names]
                    trigger_name = sentence_name+"|#|#|"+"_".join(map(str,token_index))
                    
                    event_name = self.which_event(changed_event_names,
                                                  self.event_counts,
                                                  self.policy,
                                                  self.exclude_single_trigs)
                    if event_name:
                        
                        self.events.setdefault(event_name,[])
                        self.events[event_name].append(trigger_name)

                        trigger_charseq = sum([self.charseqs[sentence_index][i] for i in token_index],[])
                        trigger_text = " ".join([self.tokens[sentence_index][i] for i in token_index])
                        trigger_dict ={"sentence": self.sentences[sentence_index],
                                       "tokens": self.tokens[sentence_index],
                                       "trigger_token_index": token_index,
                                       "trigger_charseq": [min(trigger_charseq),max(trigger_charseq)],
                                       "trigger_text": trigger_text}

                        self.triggers[trigger_name] = self.find_trigger_tokenized_index(trigger_dict)

                        
                        
    def create_pairs(self):
        self.pairs = list(combinations(list(self.triggers.keys()),2))
        self.labels = [[0] for i in range(len(self.pairs))]
        
        for i,(t1,t2) in enumerate(self.pairs):
            for _,events in self.events.items():
                if t1 in events and t2 in events:
                    self.labels[i] = [1]
                    
        
class EMWCoreferenceDataset:
    
    def __init__(self,
                 model_name: str,
                 train_dataset_paths: list = None,
                 dev_dataset_paths: list = None,
                 test_dataset_paths: list = None,
                 policy: str = "most_common",
                 exclude_single_trigs: bool = False,
                 exclude_first_sents: bool = False):
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.policy = policy
        self.exclude_single_trigs = exclude_single_trigs
        self.exclude_first_sents = exclude_first_sents

        self.train_dataset_paths = train_dataset_paths
        self.dev_dataset_paths = dev_dataset_paths
        self.test_dataset_paths = test_dataset_paths

        self.train_tensor_dataset,self.dev_tensor_dataset,self.test_tensor_dataset = self.parse()
        
        
                
    def parse(self):

        """
        JSON with a document (dict) for each line.

        Each document dict has 3 keys:
            filename: str // unique file name
            tokens: list // each item is a sentence (list) with tokens as its items 
            spans: list // each item is markables of a sentence
        """
        
        self.train_docs = self.read_docs(self.train_dataset_paths)
        self.dev_docs = self.read_docs(self.dev_dataset_paths)
        self.test_docs = self.read_docs(self.test_dataset_paths)

        self.create_trigger_pair_label()
        
        return self.tokenize(self.train_pairs,self.train_labels,self.train_triggers,"train"),self.tokenize(self.dev_pairs,self.dev_labels,self.dev_triggers,"dev"), self.tokenize(self.test_pairs,self.test_labels,self.test_triggers,"test")
    
    def read_docs(self,
                  paths: list):

        docs = {}
        for path in paths:
            with open(path) as f:
                for raw_doc in f:
                    doc = Doc(json.loads(raw_doc),
                            self.tokenizer,
                            self.policy,
                            self.exclude_single_trigs,
                            self.exclude_first_sents)
                    docs[doc.filename] = doc
        return docs

    def shuffle_set(self,pairs,labels):
        shuffled_indexes = random.sample([i for i in range(len(pairs))],len(pairs))
        shuffled_pairs = [pairs[index] for index in shuffled_indexes]
        shuffled_labels = [labels[index] for index in shuffled_indexes]

        return shuffled_pairs,shuffled_labels

    
    def create_trigger_pair_label(self):
        self.train_triggers = {trigger_name:trigger_dict for doc_name,doc_obj in self.train_docs.items() \
                         for trigger_name, trigger_dict in doc_obj.triggers.items()}
        self.train_pairs    = [p for doc_name, doc_obj in self.train_docs.items() for p in doc_obj.pairs]
        self.train_labels   = [l for doc_name, doc_obj in self.train_docs.items() for l in doc_obj.labels]
        self.train_pairs, self.train_labels = self.shuffle_set(self.train_pairs, self.train_labels)

        self.dev_triggers = {trigger_name:trigger_dict for doc_name,doc_obj in self.dev_docs.items() \
                            for trigger_name, trigger_dict in doc_obj.triggers.items()}
        self.dev_pairs    = [p for doc_name, doc_obj in self.dev_docs.items() for p in doc_obj.pairs]
        self.dev_labels   = [l for doc_name, doc_obj in self.dev_docs.items() for l in doc_obj.labels]
        self.dev_pairs, self.dev_labels = self.shuffle_set(self.dev_pairs, self.dev_labels)

        self.test_triggers = {trigger_name:trigger_dict for doc_name,doc_obj in self.test_docs.items() \
                            for trigger_name, trigger_dict in doc_obj.triggers.items()}
        self.test_pairs    = [p for doc_name, doc_obj in self.test_docs.items() for p in doc_obj.pairs]
        self.test_labels   = [l for doc_name, doc_obj in self.test_docs.items() for l in doc_obj.labels]
        self.test_pairs, self.test_labels = self.shuffle_set(self.test_pairs, self.test_labels)

    
    def tokenize(self,pairs,labels,triggers,set_name):
        
        trig1s = [triggers[trig1] for trig1,_ in pairs]
        trig2s = [triggers[trig2] for _,trig2 in pairs]
        
        trig1_sentences = [trig_dict["sentence"] for trig_dict in trig1s]
        trig2_sentences = [trig_dict["sentence"] for trig_dict in trig2s]
        
        tokenized_outputs = self.tokenizer(text = trig1_sentences,
                                           text_pair = trig2_sentences,
                                           return_tensors = "pt",
                                           return_special_tokens_mask = True,
                                           padding = True,
                                           truncation = "longest_first",
                                           max_length = 512)
        
        self.max_len = tokenized_outputs["input_ids"].shape[1]
        trigger_indexes, problematic_indexes = [],[]
        for index, ((trig1, trig2), label) in enumerate(zip(pairs,labels)):
            trig1_tokenized_index = triggers[trig1]["trigger_tokenized_index"]
            trig2_tokenized_index = triggers[trig2]["trigger_tokenized_index"] \
                                                if "roberta" in self.model_name else \
                                                triggers[trig2]["trigger_tokenized_index"][1:]
            
            
            trig1_trigger_index = trig1_tokenized_index+[0 for _ in range(self.max_len-len(trig1_tokenized_index))]
            trig2_trigger_index = [0 for _ in range(len(trig1_tokenized_index))]+trig2_tokenized_index
            trig2_trigger_index = trig2_trigger_index+[0 for _ in range(self.max_len-len(trig2_trigger_index))]
            
            if 1 in trig1_trigger_index[512:] or 1 in trig2_trigger_index[512:]:
                problematic_indexes.append(index)

            trig1_trigger_index = trig1_trigger_index[:512]
            trig2_trigger_index = trig2_trigger_index[:512]

            trigger_index = [trig1_trigger_index,trig2_trigger_index]
            
            trigger_indexes.append(trigger_index)

        wanted_indexes = [index for index in range(tokenized_outputs["input_ids"].shape[0]) if index not in problematic_indexes]
        input_ids = tokenized_outputs["input_ids"][wanted_indexes]
        attention_mask = tokenized_outputs["attention_mask"][wanted_indexes]
        trigger_indexes = torch.tensor(trigger_indexes).float()[wanted_indexes]
        labels = torch.tensor(labels).float()[wanted_indexes]

        if problematic_indexes:
            mapping = {"train": [self.train_pairs,self.train_labels],
                       "dev": [self.dev_pairs,self.dev_labels],
                       "test": [self.test_pairs,self.test_labels]}

            mapping[set_name][0] = [item for i,item in enumerate(mapping[set_name][0]) if i not in problematic_indexes] # pairs
            mapping[set_name][1] = [item for i,item in enumerate(mapping[set_name][1]) if i not in problematic_indexes] # labels

        return TensorDataset(input_ids,attention_mask,trigger_indexes,labels)