import os
import time
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer 
from train import collate_fn
from train import IWLTDataset
from train import train_tokenizer
from model import Transformer
from train import train_epoch
from train import evaluate
from train import translate
from train import calculate_bleu
from train import plot_loss_curve

def main():
    """主训练函数"""
    # 超参数
    BATCH_SIZE = 128
    NUM_EPOCHS = 31
    VOCAB_SIZE = 12000
    max_token =  128
    max_char_len = 256
    LEARNING_RATE = 1.0
    
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    # 加载数据集
    print("加载数据集...")
    train_dataset = IWLTDataset("train", max_char_len=max_char_len)
    val_dataset = IWLTDataset("validation", max_char_len=max_char_len)
    
    # 定义路径和参数
    SRC_PATH = "./src_tokenizer.json"
    TGT_PATH = "./tgt_tokenizer.json"
    

    # --- 处理源语言分词器 (Source) ---
    if os.path.exists(SRC_PATH):
        print(f"发现已存在的源语言分词器: {SRC_PATH}，正在加载...")
        src_tokenizer = Tokenizer.from_file(SRC_PATH)
    else:
        print(f"未找到源语言分词器，开始训练并保存到 {SRC_PATH} ...")
        # 调用你的训练函数
        src_tokenizer = train_tokenizer(train_dataset, VOCAB_SIZE, SRC_PATH)

    # --- 处理目标语言分词器 (Target) ---
    if os.path.exists(TGT_PATH):
        print(f"发现已存在的目标语言分词器: {TGT_PATH}，正在加载...")
        tgt_tokenizer = Tokenizer.from_file(TGT_PATH)
    else:
        print(f"未找到目标语言分词器，开始训练并保存到 {TGT_PATH} ...")
        # 调用你的训练函数
        tgt_tokenizer = train_tokenizer(train_dataset, VOCAB_SIZE, TGT_PATH)

    print("分词器准备就绪！")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, src_tokenizer, tgt_tokenizer, max_token)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, src_tokenizer, tgt_tokenizer, max_token)
    )


     # 创建模型
    print("创建模型...")
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,  
        tgt_vocab_size=VOCAB_SIZE,
        max_len=max_token,
        dropout=0.3
    ).to(device)
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度器
    def lr_lambda(step, d_model=512, warmup_steps=4000):
        if step == 0:
            return 1e-7
        scale_factor = d_model ** -0.5
        return scale_factor*min(step ** -0.5, step * warmup_steps ** -1.5)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 忽略padding的损失
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    train_losses = []
    val_losses = []
    best_bleu = 0.0
    
    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
       
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer,scheduler, criterion, device, epoch)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        '''
        if epoch % 2 == 0:
            bleu = calculate_bleu(model, val_loader, device, src_tokenizer, tgt_tokenizer)
            print(f"[Val]   BLEU: {bleu:.2f}")
        '''

        #scheduler.step() # 内部已经更新过
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, './best_transformer.pth')
            print(f"✓ 保存最佳模型，验证损失: {val_loss:.4f}")
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} 完成 | "
              f"训练损失: {train_loss:.4f} | "
              f"验证损失: {val_loss:.4f} | "
              f"时间: {epoch_time:.1f}s")
        
        # 示例翻译
        if epoch % 5 == 0:
            test_sentences = [
                "Hello, how are you?",
                "I love machine learning.",
                "The weather is nice today."
            ]
            
            print("\n示例翻译:")
            for sentence in test_sentences:
                translation = translate(model, sentence, src_tokenizer, tgt_tokenizer, device)
                print(f"英文: {sentence}")
                print(f"德文: {translation}")
                print("-" * 50)
            print()
    # 训练结束后绘制损失曲线
    plot_loss_curve(train_losses, val_losses)
    


if __name__ == "__main__":
    main()
   