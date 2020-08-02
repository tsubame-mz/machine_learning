import torch
import torch.nn as nn
import torch.nn.functional as F 
import math

# 単語IDから単語ベクトルへの変換
class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        # 学習済みモデルを読込み更新されないようにする
        self.emb = nn.Embedding.from_pretrained(embeddings=text_embedding_vectors, freeze=True)

    def forward(self, x):
        return self.emb(x)


# 単語の位置を表すベクトルを付与
class PositionEncoder(nn.Module):
    def __init__(self, d_model=300, max_seq_len=256, device="cpu"):
        super(PositionEncoder, self).__init__()

        self.d_model = d_model      # 単語ベクトルの次元数
        self.pe = torch.zeros(max_seq_len, d_model).to(device) # 位置情報ベクトル

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                self.pe[pos, i    ] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                self.pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))
        self.pe = self.pe.unsqueeze(0)  # バッチの次元を付与
        self.pe.requires_grad = False   # 勾配を計算させない

    def forward(self, x):
        return (math.sqrt(self.d_model) * x) + self.pe


# SingleHeadAttention
class Attention(nn.Module):
    def __init__(self, d_model=300):
        super(Attention, self).__init__()

        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Attentionを計算
        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model)
        weights = weights.masked_fill(mask.unsqueeze(1) == 0, -1e9)  # <pad>の重みが0になるようにする
        normalized_weights = F.softmax(weights, dim=-1)
        h = torch.matmul(normalized_weights, v)
        return self.out(h), normalized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model=300, d_hidden=1024, drop_ratio=0.1):
        super(FeedForward, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=300, d_hidden=10204, drop_ratio=0.1):
        super(TransformerBlock, self).__init__()

        # Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model)
        self.dropout1 = nn.Dropout(drop_ratio)

        # FeedForward
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_hidden, drop_ratio)
        self.dropout2 = nn.Dropout(drop_ratio)

    def forward(self, x, mask):
        # Attention
        h, normalized_weights = self.attn(self.norm1(x), mask)
        h = x + self.dropout1(h)

        # FeedForward
        h = h + self.dropout2(self.ff(self.norm2(h)))

        return h, normalized_weights


class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, d_out=2):
        super(ClassificationHead, self).__init__()

        self.layer = nn.Linear(d_model, d_out)
        nn.init.normal_(self.layer.weight, std=0.02)
        nn.init.normal_(self.layer.bias, 0)

    def forward(self, x):
        return F.softmax(self.layer(x[:, 0, :]), dim=1)   # 最初の単語(<cls>)のみ使用する


class TransformerClassification(nn.Module):
    def __init__(self, emb_vectors, d_model=300, max_seq_len=256, d_hidden=1024, d_out=2, drop_ratio=0.1, device="cpu"):
        super(TransformerClassification, self).__init__()

        self.emb = Embedder(emb_vectors)
        self.pe = PositionEncoder(d_model, max_seq_len, device)
        self.trm1 = TransformerBlock(d_model, d_hidden, drop_ratio)
        self.trm2 = TransformerBlock(d_model, d_hidden, drop_ratio)
        self.norm = nn.LayerNorm(d_model)
        self.head = ClassificationHead(d_model, d_out)

    def forward(self, x, mask):
        h = self.pe(self.emb(x))
        h, attn_w1 = self.trm1(h, mask)
        h, attn_w2 = self.trm2(h, mask)
        h = self.norm(h)
        h = self.head(h)
        return h, attn_w1, attn_w2

