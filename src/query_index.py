import faiss 
import transformers
import torch
import sqlite3
from torch.utils.data import DataLoader 
from src.embed_sbert import mean_pooling
from src.database_dataset import DatabaseDataset, collate_input
from pathlib import Path

home = str(Path.home())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = transformers.AutoTokenizer.from_pretrained('TurkuNLP/sbert-cased-finnish-paraphrase',padding_side="right",do_lower_case=False)
model = transformers.AutoModel.from_pretrained('TurkuNLP/sbert-cased-finnish-paraphrase').to(device)
index = faiss.read_index(f"{home}/data/finnish_text/Eduskunta/faiss_index_filled.faiss")
connection = sqlite3.connect(f"{home}/data/finnish_text/Eduskunta/eduskunta.db")
cursor = connection.cursor()

while True:
    var = input("Give search query: ")
    db_dataset = DatabaseDataset(tokenizer,inputs=var)
    dataloader = DataLoader(db_dataset,collate_fn=collate_input,batch_size=1)
    inputs = next(iter(dataloader))   
    with torch.no_grad():
        model_output = model(**inputs)
        sentence_embeddings = mean_pooling(model_output, inputs['attention_mask'])
        sentence_embeddings=sentence_embeddings.cpu()
        W,I=index.search(sentence_embeddings.numpy(),k=5)
        I = I[0]
        print(I)
        for i in I:
            cursor.execute(f"SELECT document_id,id,content FROM sentences WHERE id ={i}")
            print(cursor.fetchall())
    



