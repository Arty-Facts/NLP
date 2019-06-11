import torch
import IPython

class Network(torch.nn.Module):

    def __init__(self, embeddings, hidden_dim, output_dim):
        super().__init__()
        self.embeddings = torch.nn.ModuleList(embeddings)
        input_dim = sum((emb.embedding_dim for emb in embeddings))
        self.model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), \
                                         torch.nn.ReLU(), \
                                         torch.nn.Linear(hidden_dim, output_dim))

    def forward(self, features):
        input_data = torch.tensor([])
        v = torch.tensor([])
        for i, emb in enumerate(self.embeddings):
            if len(features.size()) == 1:
                v = emb(features[i])
            else:
                try:
                    v = emb(features[:, i])
                except:
                    IPython.embed()
            input_data = torch.cat((input_data, v), -1)
        return self.model(input_data)
