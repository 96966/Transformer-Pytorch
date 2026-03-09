import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 注意力模块
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model) # 实例化操作
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # mask==0就是mask ==Fasle的意思
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力权重到V
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def forward(self, q, k, v, mask=None):
        #  q，k, v在encoder中指的是输入x，x，x；而在decoder中，q来自一个decoder序列，k, v 来自encoder序列
        batch_size = q.size(0)
        
        # 线性变换并分割为多头
        Q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        #如果将上述-1改为seq_len,则需要补充seq_len = q.size(1)  # 需要额外获取seq_len
        
        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )   # contiguous()- 确保内存连续
        
        # 输出线性变换
        output = self.W_o(attn_output)
        
        return output, attn_weights
    
# 前馈神经网络
class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        return self.net(x)

# 位置编码
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 维度为【1，max_len, 512]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # 这里会自动广播
        return self.dropout(x)

# 编码器层
class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
    
#解码器层
class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力（掩码）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


#Transformer模型
class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_len=100, dropout=0.1,  pad_idx=0):
        super().__init__()
        
        # 源语言（输入）和目标语言（输出）的词嵌入层
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])  #创建了6个独立的层，此时它们还没有被‘连接’，只是放在了一个列表中
         
        # 解码器层
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self._init_parameters()
    
    def _init_parameters(self):
        """Xavier初始化所有参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 嵌入层特殊初始化
        nn.init.normal_(self.encoder_embedding.weight, mean=0, std=self.d_model**-0.5)
        nn.init.normal_(self.decoder_embedding.weight, mean=0, std=self.d_model**-0.5)
        
        # 输出层特殊初始化
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

        
    def generate_mask(self, src, tgt):
        """生成掩码"""
        device = src.device
        # 源序列填充掩码
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) 
        # 目标序列填充掩码
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) 
        tgt_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) 
       
        tgt_mask = tgt_pad_mask & causal_mask   # 是&而不是|
        
        return src_mask, tgt_mask
    
    
    def forward(self, src, tgt):
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        #src嵌入向量
        src_embedded = self.dropout(self.positional_encoding(
            self.encoder_embedding(src) * math.sqrt(self.d_model)
        ))

        # 编码器
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        # tgt嵌入向量
        tgt_embedded = self.dropout(self.positional_encoding(
            self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        ))
        
        # 解码器
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask) # 对于encoder中K,V 是一样的
        
        # 输出
        output = self.fc_out(dec_output)
        
        return output