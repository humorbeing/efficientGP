import torch
import numpy as np


class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]

class CrossAttention(torch.nn.Module):
    def __init__(self, dim, dim_ffn):
        super().__init__()
        self.norm_q = torch.nn.LayerNorm(dim)
        self.norm_k = torch.nn.LayerNorm(dim)
        self.att = torch.nn.MultiheadAttention(dim, 1, batch_first=True)
        self.norm_ffn = torch.nn.LayerNorm(dim)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_ffn),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_ffn, dim),
        )

    def forward(self, q, k, key_padding_mask=None):
        qn = self.norm_q(q)
        kn = self.norm_k(k)
        x = q + self.att(qn, kn, kn, need_weights=False, key_padding_mask=key_padding_mask)[0]
        x = x + self.ffn(self.norm_ffn(x))
        return x
    
class PumpModel(torch.nn.Module):
    def __init__(self, n_query, n_self_attention, dim, n_head=1, dropout=0.0):
        super().__init__()
        dim_ffn = dim * 4
        self.embed_point = torch.nn.Linear(6, dim)
        self.encoder = CrossAttention(dim, dim_ffn)
        self.embed_condition = torch.nn.Linear(1, dim)
        self.pos_enc = PositionalEncoding(dim, max_len=n_query+3)
        self.self_attention = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(dim, n_head, dim_ffn, dropout, batch_first=True, norm_first=True), n_self_attention, enable_nested_tensor=False,
        )
        self.decoder = CrossAttention(dim, dim_ffn)

        self.linear_output = torch.nn.Linear(dim, 4)
        self.query = torch.nn.Parameter(torch.randn([1, n_query, dim]))

        self.inoutlet_token = torch.nn.Parameter(torch.randn([1, 2, dim]))
        self.norm_out = torch.nn.LayerNorm(dim)
        self.linear_inlet = torch.nn.Linear(dim, 1)
        self.linear_outlet = torch.nn.Linear(dim, 1)
    
    def forward(self, x, n, c):
        # [Input]
        # x shape: BatchSize X NumPoint(padded) X 6(pos + normal)
        # n shape: BatchSize; example = [2, 10, 5]
        # c shape: BatchSize X 1
        # [Output]
        # y shape: BatchSize X NumPoint(padded) X 4
        # io shape: BatchSize X 2
        q = self.query.repeat([x.shape[0], 1, 1])
        p = self.embed_point(x)
        padding_mask = torch.arange(x.shape[1]).to(x.device)[None] >= n[:, None]
        x = self.encoder(q, p, key_padding_mask=padding_mask)
        c = self.embed_condition(c)
        inoutlet_token = self.inoutlet_token.repeat([x.shape[0], 1, 1])
        x = torch.cat([c[:, None], inoutlet_token, x], dim=1)
        x = self.self_attention(self.pos_enc(x))
        x = self.norm_out(x)
        y = self.linear_output(self.decoder(p, x[:, 3:]))
        io = torch.cat([self.linear_inlet(x[:, 1]), self.linear_outlet(x[:, 2])], dim=-1)
        return y, io



if __name__ == "__main__":
    from efficientGP.transforms.mesh.utils import pack

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PumpModel(n_query=64, n_self_attention=2, dim=32).to(device)

    n_padded = 65000
    x = torch.randn([8, n_padded, 6], device=device)
    nt = torch.randint(50000, n_padded, [8], device=device)
    c = torch.randn([8, 1], device=device)
    n = nt.cpu().numpy().astype(np.int32)

    y_train = torch.randn([8, n_padded, 4], device=device)
    ios = torch.randn([8, 2], device=device)

    y_pred, io_pred = model(x, nt, c)
    loss_y = torch.nn.functional.mse_loss(pack(y_pred, n), pack(y_train, n))
    loss_io = torch.nn.functional.mse_loss(io_pred, ios)

    print('done')
