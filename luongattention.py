import torch
import torch.nn as nn  
import torch.nn.functional as F

class LuongAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers,attention, p):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.attention = attention  
        
        self.Lstm_decoder = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(p)
        self.classifier = nn.Linear(self.hidden_size*2, self.output_size)
    
    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs)                                               #[1,256]
        embedded = self.dropout(embedded)

        lstm_output , hidden = self.Lstm_decoder(embedded.view(1,1,-1), hidden)         #[1,256]-->[1,1,256], [1,1,256]

        #attention class
        attention_weights = self.attention(lstm_output, encoder_outputs)                #[1,1,256], [1,3,256], # will be lstm_output.view(1,-1,1), output = [1,3,1]
        alignment_scores = F.softmax(attention_weights.view(1,-1))                      #[1,3,1]-->[1,3] # to be used for next step # didnot knew it before hand, after 
                                                                                        # doing the second step got that the alignemnt shlould be of sshape
                                                                                        # [1,1,3] so that it could be multiplied with encoder outputs in next step

        context_vector = torch.bmm(alignment_scores.unsqueeze(0), encoder_outputs)      # [1,3]-->[1,3].unsqueeze(0)=[1,1,3] * [1,3,256]

        output = torch.cat([lstm_output, context_vector], -1)

        output = F.log_softmax(self.classifier(output[0]), dim=1)
        
        return output, hidden, attention_weights



class Attention(nn.Module):
    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        
        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        
        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(1, hidden_size))
    
    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
        # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(decoder_hidden.view(1,-1,1)).squeeze(-1)
        
        elif self.method == "general":
        # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(out.view(1,-1,1)).squeeze(-1)
        
        elif self.method == "concat":
        # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden+encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)
