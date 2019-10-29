import torch
import torch.nn as nn
import dlutil as dl


class MultiHeadAttention(nn.Module):
    def __init__(self, q_in_features, k_in_features, v_in_features, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = nn.Linear(q_in_features, d_model)
        self.wk = nn.Linear(k_in_features, d_model)
        self.wv = nn.Linear(v_in_features, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = dl.scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)
        concat_attention = torch.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


class EncoderLayer(nn.Module):
    def __init__(self, in_features, num_heads, d_model, dff, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(
            in_features, in_features, in_features, num_heads, d_model)
        self.dp1 = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(d_model)
        self.dn = dl.pointwise_dense_network(d_model, dff, d_model)
        self.dp2 = nn.Dropout(drop_prob)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, enc_padding_mask):
        attn_output, attn_weight_block = self.mha(x, x, x, enc_padding_mask)
        attn_output = self.dp1(attn_output)
        out1 = self.ln1(x + attn_output)
        dn_output = self.dn(out1)
        dn_output = self.dp2(dn_output)
        out2 = self.ln2(out1 + dn_output)
        return out2, attn_weight_block


class DecoderLayer(nn.Module):
    def __init__(self, in_features, num_heads, d_model, dff, drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(
            in_features, in_features, in_features, num_heads, d_model)
        self.dp1 = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(d_model)
        self.mha2 = MultiHeadAttention(
            d_model, d_model, d_model, num_heads, d_model)
        self.dp2 = nn.Dropout(drop_prob)
        self.ln2 = nn.LayerNorm(d_model)
        self.dn = dl.pointwise_dense_network(d_model, dff, d_model)
        self.dp3 = nn.Dropout(drop_prob)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, dec_look_ahead_mask, enc_padding_mask):
        attn1, attn_weight_block1 = self.mha1(x, x, x, dec_look_ahead_mask)
        attn1 = self.dp1(attn1)
        out1 = self.ln1(x + attn1)

        attn2, attn_weight_block2 = self.mha2(
            out1, enc_output, enc_output, enc_padding_mask)
        attn2 = self.dp2(attn2)
        out2 = self.ln2(out1 + attn2)

        dn_output = self.dn(out2)
        dn_output = self.dp3(dn_output)
        out3 = self.ln3(out2 + dn_output)
        return out3, attn_weight_block1, attn_weight_block2


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, dff, enc_vocab_size, maximum_position_encoding, drop_prob=0.1, device=None):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.enc_vocab_size = enc_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.d_model = d_model
        self.embedding = nn.Embedding(enc_vocab_size, d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(
            d_model, num_heads, d_model, dff, drop_prob) for _ in range(num_layers)])
        self.dp = nn.Dropout(drop_prob)

    def forward(self, x, enc_padding_mask):
        seq_len = x.size(1)
        attn_weights = {}
        x = self.embedding(x)
        x *= torch.sqrt(torch.ones([], dtype=torch.float32,
                                   device=x.device) * self.d_model)
        min_rate = 1.0 / self.maximum_position_encoding
        pos_encoding = dl.positional_encoding(
            min_rate, self.maximum_position_encoding, self.d_model, x.device)
        x += pos_encoding[:, :seq_len, :]
        x = self.dp(x)
        for i in range(self.num_layers):
            x, attn_w = self.enc_layers[i](x, enc_padding_mask)
            attn_weights[f'encoder_layer{i+1}_w'] = attn_w
        return x, attn_weights


class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, dff, dec_vocab_size, maximum_position_encoding, drop_prob=0.1, device=None):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.dec_vocab_size = dec_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.embedding = nn.Embedding(dec_vocab_size, d_model)
        self.dec_layers = nn.ModuleList([DecoderLayer(
            d_model, num_heads, d_model, dff, drop_prob) for _ in range(num_layers)])
        self.dp = nn.Dropout(drop_prob)

    def forward(self, x, enc_output, dec_look_ahead_mask, enc_padding_mask):
        seq_len = x.size(1)
        attn_weights = {}

        x = self.embedding(x)
        x *= torch.sqrt(torch.ones([], dtype=torch.float32,
                                   device=x.device) * self.d_model)
        min_rate = 1.0 / self.maximum_position_encoding
        pos_encoding = dl.positional_encoding(
            min_rate, self.maximum_position_encoding, self.d_model, x.device)
        x += pos_encoding[:, :seq_len, :]
        x = self.dp(x)
        for i in range(self.num_layers):
            x, attn_w1, attn_w2 = self.dec_layers[i](
                x, enc_output, dec_look_ahead_mask, enc_padding_mask)
            attn_weights[f'decoder_layer{i+1}_w1'] = attn_w1
            attn_weights[f'decoder_layer{i+1}_w2'] = attn_w2
        return x, attn_weights


class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, dff, enc_vocab_size, dec_vocab_size, maximum_position_encoding=10000, enc_mask_mode=1, dec_mask_mode=2, drop_prob=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, num_heads,
                               d_model, dff, enc_vocab_size, maximum_position_encoding, drop_prob)
        self.decoder = Decoder(num_layers, num_heads,
                               d_model, dff, dec_vocab_size, maximum_position_encoding, drop_prob)
        self.attns = None
        self.enc_mask_mode = enc_mask_mode
        self.dec_mask_mode = dec_mask_mode


    def forward(self, enc_seq, dec_seq):
        enc_padding_mask = dl.create_mask(enc_seq, self.enc_mask_mode)
        dec_lh_mask = dl.create_mask(dec_seq, self.dec_mask_mode)
        enc_output, enc_attns = self.encoder(enc_seq, enc_padding_mask)
        dec_output, dec_attns = self.decoder(
            dec_seq, enc_output, dec_lh_mask, enc_padding_mask)
        self.attns = (enc_attns, dec_attns)
        return dec_output
