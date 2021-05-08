import torch
import torch.nn as nn
import torch.nn.functional as F 

class BahadauDecoder(nn.Module):
    def __init__(self, output_size, hidden_size,num_layer, p):
        super(BahadauDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.fc_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_encoder = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

        self.dropout = nn.Dropout(p=p)
        self.rnn = nn.LSTM(self.hidden_size*2, self.hidden_size, self.num_layers)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)
    
                                                                                                    # -------shapes-------#
    def forward(self, inputs, hidden, encoder_output):                                         # input([1]), hidden([1,1,256]), enc_ouput([1,3,256])
        embedded = self.embedding(inputs)                                                      # [1,256]
        embedded = self.dropout(embedded)

        # alignment score =                                                                    # first previous hidden state 
        # w(combines) * tanh(w(decoder) * H(Dec) + w(encoder) * output(encoder))               # of decoder is last hiden state of encoder
                                                                                                   
        
        x = torch.tanh(self.fc_linear(hidden) + self.fc_encoder(encoder_output))               # [1,1,256] + [1,3,256] = [1,3,256]
        alignment_scores = x.bmm(self.weight.squeeze(2))                                       # [ 1,3,256] * [1, 256] therefore unsqueeze(2) --> [1,256,1]
                                                                                               # bmm = (b,n,m)*(b,m,p) == (b,n,p) -->[1.3.1]

        attention_weights = F.Softmax(alignment_scores)                                        # [1,3,1]

        # context vector= (attn_wghts * enc_output)
        context_vector = torch.bmm(attention_weights.squeeze(2).unsqueeze(0),encoder_output)   # [1,3,1] * [1,3,256] ie [1,3,1].squeeze(2) = [1,3]
                                                                                               # [1,3].unsqueeze(1) = [1,1,3]*[1,3,256] = [1,1,256]
        
        # concatinate [embedded, context_vector]
        output = torch.cat([embedded, context_vector[0]], 1).unsqueeze(0)                      # [ 1,256] , [1,1,256][0] --> [1,256]

        # through decoder                   
        output , hidden, cell = self.rnn(output, hidden)                                       # output = [1,1,256]

        output = F.log_softmax(self.classifier(output[0]), dim=1)                               

        return output, hidden, attention_weights




