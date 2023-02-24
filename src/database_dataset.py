
import sqlite3
import torch
from torch.utils.data import Dataset,DataLoader 
from typing import Tuple, List
from transformers import AutoTokenizer
from pathlib import Path
from random import shuffle
home = str(Path.home())

class DatabaseDataset(Dataset):
    def __init__(self,tokenizer,inputs=None,device='cuda',dbfile=f"{home}/data/finnish_text/Eduskunta/eduskunta.db",batch_size=1):
        self.connection = sqlite3.connect(dbfile)
        self.cursor = self.connection.cursor()
        self.document_ids = self.query("SELECT DISTINCT document_id FROM documents")
        self.n_documents = len(self.document_ids)
        self.sentence_ids = self.query("SELECT DISTINCT id FROM sentences")
        self.n_sentences = len(self.sentence_ids )
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device
        self.inputs = inputs
    def __len__(self) -> int:
        """length method"""
        if self.inputs is not None:
            return len(self.inputs)
        else:
            return self.n_sentences

    def query(self, query_string: str) -> List[Tuple]:
        """run a query and return the result"""
        self.cursor.execute(query_string)
        return self.cursor.fetchall()


    def encode_fn(self,text):
        """
        Encode text to BERT model input
        Arguments:
            text (list): list of strings to be encoded
            text (str): string to be encoded 
        Returns:
            transformers BatchEncoding with following fields:    
            input_ids - List of token ids
            token_type_ids - List of token 
            attention_mask - List of indices specifying which tokens should be attended to by the model 

        """

        if isinstance(text,str):
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding="longest",
                truncation=True,
                max_length=512,   
            )
        elif isinstance(text,list):
            if len(text) > 1:
                encoded = self.tokenizer.batch_encode_plus(
                    text,
                    add_special_tokens=True,
                    padding="longest",
                    truncation=True,
                    max_length=512,
                )
            else:
                encoded = self.tokenizer.encode_plus(
                    text[0],
                    add_special_tokens=True,
                    padding="longest",
                    truncation=True,
                    max_length=512,
                )
        else:
            raise TypeError(f"Input shoud be string or list, got {type(text)}")
        return encoded
    

    def get_single(self, index: int):
        items = self.query(f"SELECT document_id,id,content FROM sentences WHERE id ={index}")
        return self.process_query(items)

    def get_list(self, indices: list):
        """__getitem__ handler for list inputs"""
        items = self.query(f"SELECT document_id,id,content FROM sentences WHERE id in ({','.join([str(x) for x in indices])})")
        return self.process_query(items)

    def get_slice(self, index: slice):
        """__getitem__ handler for slice inputs"""
        (start, stop, step) = (index.start,index.stop,1 if index.step is None else index.step,)
        assert not start is None and not stop is None
        return self.get_list(list(range(start, stop, step)))

    def process_query(self, items):
        input_data = {}
        meta_data = {}
        """__getitem__ post-query processing"""
        #item[0] : document_id, item[1] : id, item[2] : content
        text = [x[2] for x in items]
        document_ids = [x[0] for x in items]
        sentece_ids = [x[1] for x in items]
        meta_data['document_ids']=document_ids
        meta_data['sentence_ids']=sentece_ids
        encoded = self.encode_fn(text)
        input_data['attention_mask']=torch.tensor(encoded['attention_mask'])
        input_data['input_ids']=torch.tensor(encoded['input_ids'])
        return input_data,meta_data
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
                index = index.tolist()
        if self.inputs is None:
            if isinstance(index, slice):
                return self.get_slice(index)
            if isinstance(index, list):
                return self.get_list(index)
            if isinstance(index, int):
                return self.get_single(index)
            raise ValueError(f"Type of {str(index)} not supported by __getitem()__")
        else:
            input_data = {}
            encoded = self.encode_fn(self.inputs)
            input_data['attention_mask']=torch.tensor(encoded['attention_mask'])
            input_data['input_ids']=torch.tensor(encoded['input_ids'])
            return input_data




def collate(item_list):
    """Receives a batch in making. It is a list of dataset items, which are themselves dictionaries with the keys as returned by the dataset
    since these need to be zero-padded, then this is what we should do now. Is an argument to DataLoader"""
    items = [i[0] for i in item_list]
    meta = [i[1] for i in item_list]
    meta_dict = {}
    for k in 'sentence_ids','document_ids':
        meta_dict[k] = [v[k][0] for v in meta]
    batch = {}
    for k in "input_ids", "attention_mask":
        batch[k] = pad_with_zero([item[k] for item in items])
    return batch,meta_dict


def collate_input(item_list):
    """Receives a batch in making. It is a list of dataset items, which are themselves dictionaries with the keys as returned by the dataset
    since these need to be zero-padded, then this is what we should do now. Is an argument to DataLoader"""
    batch = {}
    for k in "input_ids", "attention_mask":
        batch[k] = pad_with_zero([item[k] for item in item_list])
    return batch


def pad_with_zero(vals):
    padded_vals = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True)
    return padded_vals.to('cuda')

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1", padding_side="right",do_lower_case=False)
    db_dataset = DatabaseDataset(tokenizer)
    dataloader = DataLoader(db_dataset,collate_fn=collate,batch_size=5)
    for i,meta in dataloader:
        print(i)
        print(meta)
        break