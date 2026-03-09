# 从本地加载数据集
import os
import re
import time
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence  # 作用: 序列填充 (Padding)
from datasets import load_dataset  # 所属库: Hugging Face Datasets (datasets), 作用: 加载和管理数据集。
from tokenizers import Tokenizer #从tokenizers 中导入Tokenizer类
from tokenizers.models import BPE  # BPE类定义结构与推理 管理推理参数
from tokenizers.trainers import BpeTrainer  # BpeTrainer定义训练过程, 管理训练参数
from tokenizers.pre_tokenizers import Whitespace  # 作用: 预分词器 (按空格切分)


def clean_iwslt_line(text):
    """
    清洗 IWSLT 数据中的 XML 标签和多余空白。
    """
    # 去除首尾空白
    text = text.strip()
    # 简单的正则去除常见的 XML 标签 (如 <seg>, </seg>, <doc>, </doc> 等)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def load_local_iwslt(split="train", data_dir="./iwslt14", max_char_len=256):
    """
    加载本地 IWSLT 数据集的函数
    
    参数:
        split: 数据集划分 ('train', 'validation', 'test')
        data_dir: 数据目录
        max_char_len: 最大字符长度阈值。
                      **注意**: 这里用于过滤异常长的行，而不是截断。
                      真正的 Token 截断应在 Tokenizer 中进行。
                      对于翻译任务，建议设为 200-300 之间，避免丢失完整语义。
    """
    # 文件名映射
    file_map = {
        "train": ("train.tags.de-en.en", "train.tags.de-en.de"),
        "validation": ("IWSLT14.TED.dev2010.de-en.en", "IWSLT14.TED.dev2010.de-en.de"),
        "test": ("IWSLT14.TED.tst2012.de-en.en", "IWSLT14.TED.tst2012.de-en.de")
    }
    
    if split not in file_map:
        raise ValueError(f"不支持的 split: {split}")
    
    en_file, de_file = file_map[split]
    en_path = os.path.join(data_dir, en_file)
    de_path = os.path.join(data_dir, de_file)
    
    if not os.path.exists(en_path) or not os.path.exists(de_path):
        raise FileNotFoundError(f"未找到数据文件。请检查路径:\nEN: {en_path}\nDE: {de_path}")

    en_texts = []
    de_texts = []
    skipped_count = 0
    
    print(f"正在加载 {split} 数据集...")
    
    # 同时读取两个文件以确保行号对齐
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(de_path, 'r', encoding='utf-8') as f_de:
        
        for line_en, line_de in zip(f_en, f_de):
            # 1. 清洗数据 (去除 XML 标签和空白)
            clean_en = clean_iwslt_line(line_en)
            clean_de = clean_iwslt_line(line_de)
            
            # 2. 基础有效性检查 (去除空行)
            if not clean_en or not clean_de:
                skipped_count += 1
                continue
            
            # 3. 长度过滤 (Filtering) 而不是截断 (Truncation)
            # 如果任一句子超过阈值，丢弃整对数据，防止语义不对齐
            if len(clean_en) > max_char_len or len(clean_de) > max_char_len:
                skipped_count += 1
                continue
            
            en_texts.append(clean_en)
            de_texts.append(clean_de)
    
    print(f"加载完成。原始行数对齐，过滤了 {skipped_count} 个样本 (原因: 空行 或 长度 > {max_char_len})")
    print(f"剩余有效样本数: {len(en_texts)}")

    # 转换为 Dataset 格式
    class LocalDataset:
        def __init__(self, en_list, de_list):
            self.data = []
            # 再次确认长度一致 (理论上 zip 后处理过应该一致，但做个防御性编程)
            assert len(en_list) == len(de_list), "英德数据长度不一致！"
            
            for en, de in zip(en_list, de_list):
                self.data.append({
                    'translation': {
                        'en': en,
                        'de': de
                    }
                })
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
            
        # 兼容 HuggingFace dataset 的一些常用方法 (可选)
        def column_names(self):
            return ['translation']
    
    return LocalDataset(en_texts, de_texts)


class IWLTDataset(Dataset):
    """IWSLT 2014 英德翻译数据集 - 使用本地数据"""
    def __init__(self, split="train", max_char_len=256, data_dir="./iwslt14"):
        super().__init__()
        # 使用本地加载函数
        self.dataset = load_local_iwslt(split=split, data_dir=data_dir, max_char_len=max_char_len)

        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['translation']['en'], item['translation']['de']
    
# 定义分词器
def train_tokenizer(dataset, vocab_size=12000, save_path="./tokenizer.json"):
    """训练BPE分词器"""
    # 收集所有文本
    texts = []
    for i in range(len(dataset)):
        en, de = dataset[i]
        texts.append(en)
        texts.append(de)
    
    # 初始化BPE分词器
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"], 
        min_frequency=2,        # 忽略出现次数太少的字符组合
        show_progress=True 
    )
    
    # 训练分词器
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(save_path)
    return tokenizer
'''

# 使用分词器
text = "Hello, how are you?"
encoded = tokenizer.encode(text)
print(encoded.ids)      # 输出token IDs
print(encoded.tokens)   # 输出tokens
'''

def collate_fn(batch, src_tokenizer, tgt_tokenizer, max_len=512):
    """批处理函数
    将 DataLoader 抓取的一个 Batch 的原始文本对 (src_text, tgt_text) 转换为模型训练所需的输入格式。
    执行了分词编码、特殊标记注入、长度控制以及序列填充四个关键步骤，确保同一个 Batch 内的所有样本具有相同的序列长度。
    """
    src_texts, tgt_texts = zip(*batch)
    #print(f'src_text:{src_texts[:5]}')
    #print(f'src_text:{tgt_texts[:5]}')
    # 编码文本
    src_encoded = []
    tgt_encoded = []

    # 获取特殊token的ID
    pad_id = src_tokenizer.token_to_id("[PAD]")
    #print(f'pad_id:{pad_id}')
    sos_id = src_tokenizer.token_to_id("[SOS]")
    eos_id = src_tokenizer.token_to_id("[EOS]")
    
    for src_text, tgt_text in zip(src_texts, tgt_texts):
        # 添加特殊标记
        src_tokens = src_tokenizer.encode(src_text).ids
        tgt_tokens = tgt_tokenizer.encode(tgt_text).ids
        # 2. 手动添加特殊标记（在ID层面操作）
        # Encoder输入: 只有 [EOS] 在结尾
        src_tokens = src_tokens + [eos_id]  # 注意：不加 [SOS]
        # Decoder输入: [SOS] + 文本 + [EOS]
        tgt_tokens = [sos_id] + tgt_tokens + [eos_id]
        
        if len(src_tokens) > max_len:
            src_tokens =src_tokens[:max_len-1] + [eos_id]
        if len(tgt_tokens) > max_len:
            tgt_tokens = [sos_id] + tgt_tokens[0:max_len-2] + [eos_id]
        
        src_encoded.append(torch.tensor(src_tokens))
        tgt_encoded.append(torch.tensor(tgt_tokens))
    
    # 填充序列
    src_padded = pad_sequence(src_encoded, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_encoded, batch_first=True, padding_value=pad_id)
    
    return src_padded, tgt_padded





# 训练一个模型
#def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch, clip=1.0):
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        #print(f'前5个src:{src[0:5]}')
        #print(f'前5个tgt:{tgt[0:5]}')
        optimizer.zero_grad()
        
        # 准备输入和目标
        tgt_input = tgt[:, :-1]  # 去掉EOS
        #print(f'tgt_input:{tgt_input[:1,:]}')
        tgt_target = tgt[:, 1:]  # 去掉SOS

        
        # 前向传播
        output = model(src, tgt_input)
        
        # 计算损失
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
        loss = loss        
        # 反向传播
        loss.backward()

        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        #scheduler.step() 
    

        total_loss += loss.item()
        
        if batch_idx % 12 == 0:
            lr = optimizer.param_groups[0]['lr']
            cur_loss = total_loss / (batch_idx + 1)
            elapsed = time.time() - start_time
            print(f'Epoch: {epoch:3d} | Batch: {batch_idx:5d}/{len(dataloader):5d} | LR: {lr:.2e} |' 
                  f'Loss: {cur_loss:6.4f} | Time: {elapsed:5.1f}s') #Grad Norm:{grad_norm.item():.4f}|
    return total_loss / len(dataloader)
    
#定义评估函数
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_target.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 定义翻译函数
def translate(model, sentence, src_tokenizer, tgt_tokenizer, device, max_len=256):
    model.eval()
    
    src_tokens = src_tokenizer.encode(sentence).ids
    eos_id = src_tokenizer.token_to_id("[EOS]")
    src_tokens = src_tokens + [eos_id]
    src_tensor = torch.tensor(src_tokens).unsqueeze(0).to(device)
    
    tgt_tokens = [tgt_tokenizer.token_to_id("[SOS]")]
    pad_id = tgt_tokenizer.token_to_id("[PAD]")
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)
        
        # 获取最后一个 token 的 logits
        next_token_logits = output[:, -1, :]
        
        # 可选：应用 Temperature 采样 (例如 temp=0.7) 增加多样性，防止过早陷入 PAD 或 EOS
        # 这里为了稳定性先保持 argmax，但确保不预测 PAD
        next_token_logits[:, pad_id] = -float('inf') 
        
       # next_token = next_token_logits.argmax(-1).item()
        temperature = 0.7  # 小于 1 会让分布更尖锐，但保留一点随机性
        probs = torch.softmax(next_token_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        tgt_tokens.append(next_token)
        
        if next_token == tgt_tokenizer.token_to_id("[EOS]"):
            break
    
    # 解码时去掉 SOS 和 EOS
    # 注意：如果第一个生成的 token 就是 EOS，列表切片需小心
    if len(tgt_tokens) <= 2:
        return ""
        
    translation = tgt_tokenizer.decode(tgt_tokens[1:-1])
    return translation

# ==========================================
# 新增：BLEU 评估函数
# ==========================================
import sacrebleu
def calculate_bleu(model, dataloader, device, src_tokenizer, tgt_tokenizer, max_len=50):
    """
    在验证集上计算 BLEU 分数
    """
    model.eval()
    hypotheses = []
    references = []
    
    # 获取特殊 token ID
    sos_id = tgt_tokenizer.token_to_id("[SOS]")
    eos_id = tgt_tokenizer.token_to_id("[EOS]")
    pad_id = tgt_tokenizer.token_to_id("[PAD]")
    
    print(f"   开始计算 BLEU (采样前 1000 句以加速)...")
    
    with torch.no_grad():
        # 为了速度，通常只计算验证集的一部分，或者整个验证集如果它不大的话
        # 这里我们遍历整个 dataloader，如果太慢可以在外面切片 dataloader
        count = 0
        limit = 1000 # 限制最多评估 1000 句，防止评估太慢
        
        for src, tgt in dataloader:
            if count >= limit:
                break
                
            src = src.to(device)
            tgt = tgt.to(device)
            batch_size = src.size(0)
            
            # --- 贪婪搜索解码 (Greedy Decoding) ---
            # 初始化输入为 [SOS]
            ys = torch.ones(batch_size, 1).fill_(sos_id).long().to(device)
            
            for i in range(max_len - 1):
                output = model(src, ys)
                next_word_logits = output[:, -1, :]
                
                # Mask PAD token 以防预测出 PAD
                next_word_logits[:, pad_id] = -float('inf')
                
                next_word = next_word_logits.argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_word], dim=1)
                
                # 如果所有句子都生成了 EOS，提前停止
                if (next_word == eos_id).all():
                    break
            
            # --- 将索引转换为文本 ---
            for j in range(batch_size):
                if count >= limit:
                    break
                
                # 获取预测序列 (去掉 bos)
                pred_seq = ys[j, 1:].cpu().tolist()
                # 获取真实目标序列 (去掉 bos 和 eos/padding)
                # tgt[j] 格式: [bos, w1, w2, ..., eos, pad, pad]
                true_seq = tgt[j].cpu().tolist()
                
                # 简单清理：截取到第一个 EOS 之前
                try:
                    pred_end = pred_seq.index(eos_id)
                    pred_seq = pred_seq[:pred_end]
                except ValueError:
                    pass # 没有 EOS，保留全部
                
                try:
                    true_start = true_seq.index(sos_id) + 1
                    true_end = true_seq.index(eos_id)
                    true_seq = true_seq[true_start:true_end]
                except ValueError:
                    true_seq = true_seq[1:] # 容错处理

                # 解码
                # 注意：tokenizers 库的 decode 方法可以直接处理 ID 列表
                hyp_text = tgt_tokenizer.decode(pred_seq)
                ref_text = tgt_tokenizer.decode(true_seq)
                
                hypotheses.append(hyp_text)
                references.append(ref_text)
                count += 1
    
    # --- 计算 SacreBLEU ---
    # sacrebleu 期望 references 是列表的列表 [[ref1], [ref2], ...]
    if not hypotheses:
        return 0.0
        
    references_formatted = [[ref] for ref in references]
    bleu_score = sacrebleu.corpus_bleu(hypotheses, references_formatted)
    
    model.train() # 恢复训练模式
    return bleu_score.score

# 定义plot_loss_curve函数
import matplotlib
import matplotlib.pyplot as plt
def plot_loss_curve(train_losses, val_losses=None, save_path='loss_curve.png'):
    """
    绘制训练和验证损失曲线
    
    参数:
        train_losses: 训练损失列表，每个epoch一个值
        val_losses: 验证损失列表，每个epoch一个值（可选）
        save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # 绘制训练损失
    plt.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=8, 
             label='Training Loss', alpha=0.8, markerfacecolor='white')
    
    if val_losses is not None:
        plt.plot(epochs[:len(val_losses)], val_losses, 'r-s', linewidth=2, 
                 markersize=8, label='Validation Loss', alpha=0.8, 
                 markerfacecolor='white')
    
    # 添加最后一点的数值标签
    if train_losses:
        plt.annotate(f'{train_losses[-1]:.4f}', 
                    xy=(len(train_losses), train_losses[-1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, color='blue')
    
    if val_losses and len(val_losses) == len(train_losses):
        plt.annotate(f'{val_losses[-1]:.4f}', 
                    xy=(len(val_losses), val_losses[-1]),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=10, color='red')
    
    # 设置图表属性
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    
    title = 'Training Loss' + (' and Validation Loss' if val_losses else '')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置y轴范围，留出一些空间
    all_losses = train_losses + (val_losses if val_losses else [])
    y_min, y_max = min(all_losses), max(all_losses)
    plt.ylim([y_min * 0.9, y_max * 1.1])
    
    # 添加次要网格
    plt.grid(True, which='minor', alpha=0.2, linestyle=':')
    
    # 可以选择使用对数刻度（如果损失变化范围很大）
    if max(all_losses) / min(all_losses) > 100:  # 如果最大最小相差100倍以上
        plt.yscale('log')
        plt.ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    #plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #print(f"损失曲线已保存到: {save_path}")
    
    plt.show()


