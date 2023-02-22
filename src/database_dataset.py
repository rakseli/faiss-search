import sqlite3
import torch
from typing import Tuple, List
from transformers import AutoTokenizer
from pathlib import Path
from random import shuffle
home = str(Path.home())

class DatabaseDataset(torch.utils.data.Dataset):
    def __init__(self,tokenizer,device='cuda',dbfile=f"{home}/data/finnish_text/Eduskunta/eduskunta.db",batch_size=1):
        self.connection = sqlite3.connect(dbfile)
        self.cursor = self.connection.cursor()
        self.document_ids = self.query("SELECT DISTINCT document_id FROM documents")
        self.n_documents = len(self.document_ids)
        self.sentence_ids = self.query("SELECT DISTINCT id FROM sentences")
        self.n_sentences = len(self.sentence_ids )
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self) -> int:
        """length method"""
        return self.n_sentences()

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
        input_data['attention_mask']=torch.tensor(encoded['attention_mask'],device=self.device)
        input_data['input_ids']=torch.tensor(encoded['input_ids'],device=self.device)
        return input_data,meta_data
        
    def __getitem__(self, index):
        """get all movie id indexes for a given user index"""
        if torch.is_tensor(index):
            index = index.tolist()
        if isinstance(index, slice):
            return self.get_slice(index)
        if isinstance(index, list):
            return self.get_list(index)
        if isinstance(index, int):
            return self.get_single(index)
        raise ValueError(f"Type of {str(index)} not supported by __getitem()__")


    def __iter__(self):
        return DatabaseIterator(self, batch_size=self.batch_size)


class DatabaseIterator:
    def __init__(self, db_object, batch_size=1):
        self.ordering = list(range(db_object.n_sentences))
        self.index = 0
        self.iterable = db_object
        self.batch_size = batch_size

    def __next__(self):
        indexes = self.ordering[self.index : (self.index + self.batch_size)]
        result = self.iterable[indexes]
        self.index = self.index + self.batch_size
        return result

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1", padding_side="right",do_lower_case=False)
    db_dataset = DatabaseDataset(tokenizer,batch_size=5)
    for i,meta in db_dataset:
        print(i)
        print(meta)
        break