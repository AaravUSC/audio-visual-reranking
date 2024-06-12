import torch
from torch import nn
import torch.nn.functional as F

import math


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(
            0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float(
        ) * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + \
            self.pos_encoding[:token_embedding.shape[1], :].transpose(0, 1).repeat_interleave(token_embedding.shape[0], dim=0))


class TransformerPipeline(nn.Module):
    # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self,
                 d_audio: list,
                 d_out: int,
                 depth: int,
                 max_target_len: int,
                 dropout = 0.1):
        super(TransformerPipeline, self).__init__()

        self.d_audio_L, self.d_audio_H = d_audio
        self.d_out = d_out

        self.emb = nn.Embedding(d_out, self.d_audio_H)
        self.positional_encoder = PositionalEncoding(
            dim_model=self.d_audio_H,
            dropout_p=0,
            max_len=max_target_len
        )
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_audio_H,
            nhead=8,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.out = nn.Linear(self.d_audio_H, d_out)

    def forward(self,
                mem: torch.tensor,  # encoder states
                tgt: torch.tensor,  # decoder inputs
                tgt_mask: torch.tensor = None,
                tgt_pad_mask: torch.tensor = None
                ):

        # Embedding + Positional Encoding
        # (B x L x H)
        tgt = self.emb(tgt) * math.sqrt(self.d_audio_H)
        # mem = self.positional_encoder(mem)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks
        transformer_out = self.decoder(tgt,                 # (B, L, E)
                                       memory=mem,          # (B, L_S, H_S)
                                       tgt_mask=tgt_mask,   # (L, L)
                                       tgt_key_padding_mask=tgt_pad_mask)  # (B, L)
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        # Lower triangular matrix
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float(
            '-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        return mask

    def create_pad_mask(self,
                        matrix: torch.tensor,
                        pad_token: int) -> torch.tensor:
        return (matrix == pad_token)


class MultimodalTransformerPipeline(nn.Module):
    # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self,
                 d_audio: list,
                 d_vision: int,
                 d_out: int,
                 depth: int,
                 max_target_len: int,
                 dropout: float):
        super(MultimodalTransformerPipeline, self).__init__()

        self.d_audio_L, self.d_audio_H = d_audio
        self.d_vision = d_vision
        self.d_out = d_out
        self.max_target_len = max_target_len

        self.emb = nn.Embedding(d_out, self.d_audio_H)
        self.fusion = nn.Linear(self.d_audio_H + self.d_vision, self.d_audio_H)
        self.positional_encoder = PositionalEncoding(
            dim_model=self.d_audio_H,
            dropout_p=0,
            max_len=max_target_len
        )
        decoder_layer = nn.TransformerDecoderLayer(
            self.d_audio_H,
            nhead=8,
            batch_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.out = nn.Linear(self.d_audio_H, d_out)

    def forward(self,
                mem: torch.tensor,  # encoder states
                vis: torch.tensor,  # visual features
                tgt: torch.tensor,  # decoder inputs
                tgt_mask: torch.tensor = None,
                tgt_pad_mask: torch.tensor = None
                ):

        # Embedding + Positional Encoding
        # (B x L x H)
        tgt = self.emb(tgt)
        # (B x L x H)
        vis = vis.repeat_interleave(tgt.shape[1], dim=1)
        tgt = self.fusion(torch.cat([tgt, vis], axis=2))
        tgt = F.relu(tgt) * math.sqrt(self.d_audio_H)
        tgt = self.positional_encoder(tgt)

        # Transformer blocks
        transformer_out = self.decoder(tgt,                 # (B, L, E)
                                       memory=mem,          # (B, L_S, H_S)
                                       tgt_mask=tgt_mask,   # (L, L)
                                       tgt_key_padding_mask=tgt_pad_mask)  # (B, L)
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        # Lower triangular matrix
        mask = torch.tril(torch.ones(size, size) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float(
            '-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        return mask

    def create_pad_mask(self,
                        matrix: torch.tensor,
                        pad_token: int) -> torch.tensor:
        return (matrix == pad_token)


class GRUPipeline(nn.Module):
    # https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    def __init__(self,
                 d_audio: list,
                 d_out: int,
                 depth: int,
                 bidirectional: bool):
        super(GRUPipeline, self).__init__()

        self.d_audio_L, self.d_audio_H = d_audio
        self.d_out = d_out

        self.emb = nn.Embedding(d_out, self.d_audio_H)
        self.attn = nn.Linear(self.d_audio_H * 2, self.d_audio_L)
        self.attn_combine = nn.Linear(self.d_audio_H * 2, self.d_audio_H)
        self.gru = nn.GRU(self.d_audio_H, self.d_audio_H, depth,
                          batch_first=True, bidirectional=bidirectional)
        self.out = nn.Linear(
            self.d_audio_H * (2 if bidirectional else 1), d_out)

    def forward(self,
                input: torch.tensor,  # The input token
                out_hidden: torch.tensor,  # Last hidden state
                out_encoder: torch.tensor):  # Encoder states

        # Embed input token
        # (B x 1 x 1) -> (B x 1 x H)
        inp_emb = self.emb(input)

        # Get attention weights
        # (B x 1 x 2H) -> (B x 1 x S)
        weights_attn = self.attn(
            torch.cat([inp_emb, out_hidden[-1:, :, :].transpose(0, 1)], axis=2))
        weights_attn = F.softmax(weights_attn, dim=1)

        # Attend over encoder out
        # (B x 1 x S) X (B x S x H)
        attn_applied = torch.bmm(weights_attn, out_encoder)

        # Concat with context
        # (B x 1 x 2H) -> (B x 1 x H)
        output = self.attn_combine(torch.cat((inp_emb, attn_applied), axis=2))
        output = F.relu(output)

        # Feed into LSTM and decision layer
        self.gru.flatten_parameters()
        output, out_hidden = self.gru(output, out_hidden)
        output = self.out(output).squeeze(1)

        return output, out_hidden, weights_attn
