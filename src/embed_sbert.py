import sys
import torch
import transformers
import tqdm
import pickle
from src.database_dataset import DatabaseDataset,collate
from pathlib import Path
from torch.utils.data import DataLoader 

home = str(Path.home())

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained('TurkuNLP/sbert-cased-finnish-paraphrase',padding_side="right",do_lower_case=False)
    model = transformers.AutoModel.from_pretrained('TurkuNLP/sbert-cased-finnish-paraphrase').to(device)
    db_dataset = DatabaseDataset(tokenizer)
    dataloader = DataLoader(db_dataset,collate_fn=collate,batch_size=1000)
    with tqdm.tqdm() as pbar:
        for i,(batch,meta_data) in enumerate(dataloader):
            with torch.no_grad(), open(f"{home}/data/finnish_text/Eduskunta/embeddings_{i}.pickle","wb") as f_out:
                model_output = model(**batch)
                sentence_embeddings = mean_pooling(model_output, batch['attention_mask'])
                sentence_embeddings=sentence_embeddings.cpu()
                bsize=sentence_embeddings.shape[0]
                pickle.dump((meta_data["sentence_ids"],sentence_embeddings),f_out)
                pbar.update(bsize)
