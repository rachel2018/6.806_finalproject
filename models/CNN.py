import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, embeddings, hidden_dim, dropout_ratio, cuda, query_length):
        super(CNN, self).__init__()
        self.cuda = cuda
        self.query_length = query_length
        self.convolution = nn.Conv1d(embeddngs.shape, hidden_dim, kernel_size=3, padding=2)
        self.tanh = nn.Tanh()
        self.embedding_layer = nn.Embedding(embeddings.shape, embeddings.shape)
        self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.embedding_layer.requires_grad = False
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, tensor):
        mask = (tensor != 0)
        if self.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)

        length = torch.unsqueeze(torch.sum(mask,1),1).expand(mask.data.shape)
        mask = torch.div(mask,length)

        output= self.embedding_layer(tensor)
        output = output.permute(0,2,1)
        conv_output = self.convolution(output)
        dropout_output = self.dropout(conv_output)
        tanh_output = self.tanh(dropout_output)
        output = tanh_output[:,:,:self.query_length]
        N, hd, co =  output.data.shape
        expanded_mask = torch.unsqueeze(mask,1).expand(N, hd, co)
        masked = torch.mul(expanded_mask,output)
        #using average pooling method which performs better
        output = torch.sum(masked, dim=2)
        return output
