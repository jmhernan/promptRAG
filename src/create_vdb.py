from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import faiss
import torch
import pandas as pd

class IndexTextEmbeddings:
    def __init__(self, model_name='sentence-transformers/multi-qa-mpnet-base-dot-v1', device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    @staticmethod
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

    def get_embeddings(self, text_list):
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return self.cls_pooling(model_output).detach().cpu().numpy()

    def create_dataset(self, data_frame, text_column_name):
        dataset = Dataset.from_pandas(data_frame)

        # FIX ME: Won't scale for larger datasets need to implement batching
        embeddings = []
        for text in dataset[text_column_name]:
            embedding = self.get_embeddings([text])[0]  # get_embeddings expects a list
            embeddings.append(embedding)
        dataset = dataset.add_column("embeddings", embeddings)
        return dataset
    
    @staticmethod
    def add_faiss_index(dataset, column_name="embeddings"):
        dataset.add_faiss_index(column=column_name)
        return dataset