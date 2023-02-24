import faiss
import torch
import random
import tqdm
import pickle
import os
from get_files import get_files_from_dir
from pathlib import Path

def random_sample_of_batches(batch_files, proportion=0.30,seed=666):
    """Takes a random sample of batches from all batch_files
    this is used to make training data for faiss. Proportion
    is from [0,1] interval"""
    all_batches = []
    random.seed(seed)
    random.shuffle(batch_files)
    batches_n = int(len(batch_files)*proportion)
    with tqdm.tqdm() as pbar:
        for i,b in enumerate(batch_files):
            if i == batches_n:
                break
            with open(b, "rb") as f:
                for i in range(batches_n):
                    try:
                        sent_ids, embedding_batch = pickle.load(f)
                        all_batches.append(embedding_batch)
                        pbar.update(embedding_batch.shape[0])
                    except EOFError as e:
                        print(e)
    random.shuffle(all_batches)
    print("Got", len(all_batches), "random batches")
    return torch.vstack(all_batches)


if __name__ == "__main__":
    home = str(Path.home())
    embeddings = get_files_from_dir(f"{home}/data/finnish_text/Eduskunta",'embeddings_')
    if not os.path.exists(f"{home}/data/finnish_text/Eduskunta/faiss_index_pretrained.faiss"):
        sampled = random_sample_of_batches(embeddings)
        d = 768 #bert embedding dim
        quantizer = faiss.IndexFlatL2(d)
        m = 8  # number of subvectors in each compressed vector
        bits = 8 # number of bits in each centroid
        nlist = 128 #n number of Voronoi cells
        index = faiss.IndexIVFPQ(quantizer,d,nlist,m,bits)
        print("Training on", sampled.shape, "vectors")
        index.train(sampled.numpy())
        print("Done training")
        trained_index = index
        faiss.write_index(trained_index,f"{home}/data/finnish_text/Eduskunta/faiss_index_pretrained.faiss")
    else:
        trained_index = faiss.read_index(f"{home}/data/finnish_text/Eduskunta/faiss_index_pretrained.faiss")

    if not os.path.exists(f"{home}/data/finnish_text/Eduskunta/faiss_index_filled.faiss"):
        for batchfile in tqdm.tqdm(embeddings):
            with open(batchfile,"rb") as f:
                    try:
                        sentece_ids, embedded_batch = pickle.load(f)
                        trained_index.add_with_ids(embedded_batch.numpy(), sentece_ids)
                    except EOFError as e:
                        print(e)
                    
        index_filled = trained_index
        faiss.write_index(index_filled, f"{home}/data/finnish_text/Eduskunta/faiss_index_filled.faiss")
        print("Index has", index_filled.ntotal, "vectors. Done.")
    else:
        print("faiss_index_filled.faiss aldready exists")