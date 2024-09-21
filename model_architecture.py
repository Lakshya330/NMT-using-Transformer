import math
import torch
import numpy as np
from torch import nn
from torchinfo import summary
from torch.nn.functional import softmax, relu
from torch.utils.tensorboard.writer import SummaryWriter


#### EMBEDDING
class Embedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):

        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, input):

        return self.embedding(input)


#### POSITIONAL ENCODING

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int):
        super(PositionalEncoding, self).__init__()

        # Create a position tensor (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Create a div_term tensor (1, d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # Calculate the positional encodings
        self.pos_enc = torch.zeros(seq_len, d_model)
        self.pos_enc[:, 0::2] = torch.sin(position * div_term)
        self.pos_enc[:, 1::2] = torch.cos(position * div_term)

        # Register buffer so the positional encoding is not a learnable parameter
        self.register_buffer('pos_encoding', self.pos_enc.unsqueeze(0))  # Shape: (1, seq_len, d_model)


    def forward(self, embedding):
        seq_len = embedding.shape[1]
        pos_enc = self.pos_encoding[:, :seq_len, :]

        return embedding + pos_enc


#### MULTI HEAD ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.single_head_dim = d_model // self.num_heads
        self.query_matrix = nn.Linear(d_model, d_model)
        self.key_matrix = nn.Linear(d_model, d_model)
        self.value_matrix = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.5)

    def forward(self, pos_emb, mask = None):

        batch_size = pos_emb.shape[0]
        seq_len = pos_emb.shape[1]

        q = self.query_matrix(pos_emb)
        k = self.key_matrix(pos_emb)
        v = self.value_matrix(pos_emb)


        q = q.view(batch_size, seq_len, self.num_heads, self.single_head_dim).transpose(2, 1)
        k = k.view(batch_size, seq_len, self.num_heads, self.single_head_dim).transpose(2, 1)
        v = v.view(batch_size, seq_len, self.num_heads, self.single_head_dim).transpose(2, 1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.single_head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, value = -1e9)

        attention_weights = softmax(scores, dim = -1)

        output = torch.matmul(attention_weights, v).transpose(1, 2)
        output = output.reshape(batch_size, seq_len, -1)


        return attention_weights, self.dropout(self.fc(output))


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(CrossAttention, self).__init__()

        self.num_heads = num_heads
        self.single_head_dim = d_model // self.num_heads

        self.key_matrix = nn.Linear(d_model, d_model)
        self.query_matrix = nn.Linear(d_model, d_model)
        self.value_matrix = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)


    def forward(self, encoder_output, masked_decoder_output):
        batch_size = masked_decoder_output.shape[0]
        seq_len = masked_decoder_output.shape[1]


        q = self.query_matrix(masked_decoder_output)
        k = self.key_matrix(encoder_output)
        v = self.value_matrix(encoder_output)

        q = q.view(batch_size, seq_len, self.num_heads, self.single_head_dim).transpose(2, 1)
        k = k.view(batch_size, encoder_output.shape[1], self.num_heads, self.single_head_dim).transpose(2, 1)
        v = v.view(batch_size, encoder_output.shape[1], self.num_heads, self.single_head_dim).transpose(2, 1)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.single_head_dim)
        attention_weights = softmax(scores, dim = -1)
        output = torch.matmul(attention_weights, v).transpose(1, 2)
        output = output.reshape(batch_size, seq_len, -1)

        return attention_weights, self.fc(output)

class AddLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super(AddLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)


    def forward(self, res_input1, res_input2):
        residual_output = res_input1 + res_input2
        normalized = self.layer_norm(residual_output)

        return normalized


class FFN(nn.Module):
    def __init__(self, hidden_dim: int, d_model: int):
        super(FFN, self).__init__()

        self.hidden_layer = nn.Linear(d_model, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(0.6)

    def forward(self, normalized_output):
        hidden_output = relu(self.hidden_layer(normalized_output))
        output = self.output_layer(hidden_output)

        return self.dropout(output)

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int):
        super(EncoderBlock, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.add_layer_norm = AddLayerNorm(d_model)
        self.ffn = FFN(hidden_dim, d_model)

    def forward(self, pos_emb):

        attention_weights, attention_outputs = self.multi_head_attention(pos_emb)
        normalized = self.add_layer_norm(pos_emb, attention_outputs)
        output = self.ffn(normalized)
        normalized_ff_out = self.add_layer_norm(normalized, output)

        return normalized_ff_out


class Encoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, seq_len: int, num_heads: int, hidden_dim: int, num_layers: int):
        super(Encoder, self).__init__()

        self.embedding = Embedding(d_model, vocab_size)
        self.pos_enc = PositionalEncoding(seq_len, d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, hidden_dim) for _ in range(num_layers - 1)])

    def forward(self, input):

        embedding = self.embedding(input)
        pos_emb = self.pos_enc(embedding)

        output = pos_emb

        for layer in self.layers:
            output = layer(output)

        return output


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_dim: int):
        super(DecoderBlock, self).__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = CrossAttention(d_model, num_heads)
        self.add_layer_norm = AddLayerNorm(d_model)
        self.ffn = FFN(hidden_dim, d_model)


    def forward(self, pos_emb, encoder_output, mask = None):
        attention_weights, attention_output = self.multi_head_attention(pos_emb, mask)
        normalized = self.add_layer_norm(pos_emb, attention_output)
        cross_att_weights, cross_att_output = self.cross_attention(encoder_output, attention_output)
        normalized_cross_att = self.add_layer_norm(normalized, cross_att_output)
        output = self.ffn(normalized_cross_att)
        normalized_ff_out = self.add_layer_norm(normalized_cross_att, output)

        return normalized_ff_out



class Decoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int, vocab_size: int, num_heads: int, hidden_dim: int, num_layers: int):
        super(Decoder, self).__init__()

        self.embedding = Embedding(d_model, vocab_size)
        self.pos_enc = PositionalEncoding(seq_len, d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, hidden_dim) for _ in range(num_layers - 1)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, input, encoder_output, mask = None):

        embedding = self.embedding(input)
        pos_emb = self.pos_enc(embedding)

        output = pos_emb

        for layer in self.layers:
            output = layer(output, encoder_output, mask)

        output = self.linear(output)

        return output


class Transformer(nn.Module):
    def __init__(self, d_model: int, seq_len: int, src_vocab_size: int, trg_vocab_size: int, hidden_dim: int, num_heads: int, num_layers: int):
        super(Transformer, self).__init__()

        self.encoder = Encoder(d_model, src_vocab_size, seq_len, num_heads, hidden_dim, num_layers).to(device)
        self.decoder = Decoder(d_model, seq_len, trg_vocab_size, num_heads, hidden_dim, num_layers).to(device)


    def forward(self, source, target, mask = None):

        encoder_output = self.encoder(source)
        output = self.decoder(target, encoder_output, mask)

        return output


### HYPERPARAMETERS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir = "/content/Transformer/runs")
d_model = 350
max_length = 30
src_vocab_size = len(german.vocab.stoi)
trg_vocab_size = len(english.vocab.stoi)
hidden_dim = 1024
num_heads = 2
num_layers = 2


transformer = Transformer(d_model, max_length, src_vocab_size, trg_vocab_size, hidden_dim, num_heads, num_layers).to(device)

writer.add_graph(transformer, (torch.rand(32, 30, device = device).to(torch.long), torch.rand(32, 30, device = device).to(torch.long), torch.tril(torch.ones(30, 30)).unsqueeze(0).unsqueeze(0).to(device)))
summary(transformer, input_data = [torch.rand(32, 30, device = device).to(torch.long), torch.rand(32, 30, device = device).to(torch.long), torch.tril(torch.ones(30, 30)).unsqueeze(0).unsqueeze(0).to(device)])
