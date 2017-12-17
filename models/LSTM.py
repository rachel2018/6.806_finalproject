import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, embeddings, hidden_dim, batch_size=20, dropout_ratio, cuda):
        super(LSTM, self).__init__()
        self.cuda = cuda
        self.embedding_layer = nn.Embedding(embeddings.shape, embeddings.shape)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer.requires_grad = False
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden(batch_size)
        self.lstm = nn.LSTM(embeddings.shape, hidden_dim//2, bidirectional=True, dropout=dropout_ratio)
        return

    def forward(self, tensor):
        mask = (tensor != 0)
        if self.cuda:
            mask = mask.type(torch.cuda.FloatTensor)
        else:
            mask = mask.type(torch.FloatTensor)
        
        length = torch.unsqueeze(torch.sum(mask,1),1).expand(tensor.data.shape)
        mask = torch.unsqueeze(torch.div(mask,length),2)
        output = self.embedding_layer(tensor)
        batch_size = output.data.shape[0]
        perm_output = output.permute(1,0,2)
        self.hidden = self.init_hidden(batch_size)
        output, self.hidden = self.lstm(perm_output, self.hidden)
        N, hd, co =  output.data.shape
        mask = mask.permute(1,0,2)
        mask = mask.expand(N, hd, co)
        mask = torch.mul(mask,output)

        #using average pooling for best performance
        output = torch.sum(mask,0)
        return output

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim//2))
        c = autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim//2))
        if self.cuda:
            h = h.cuda()
            c = c.cuda()
        return (h,c)