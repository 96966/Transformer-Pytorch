import os
import time
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Subset 
from train_debug import collate_fn
from train_debug import IWLTDataset
from train_debug import train_tokenizer
from model import Transformer
from train_debug import train_epoch
from train_debug import evaluate
from train_debug import translate
from train_debug import calculate_bleu
from train_debug import plot_loss_curve

def main():
    """主训练函数"""
    # 超参数
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
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
    
    

# ==========================================
# 调试模式：只取前 10 条数据
# ==========================================
    DEBUG_MODE = True 

    if DEBUG_MODE:
        print("进入调试模式：仅使用 10 条数据进行过拟合测试...")
        
        # 1. 创建一个只包含前 10 个索引的 Subset
        # 假设 train_dataset 是原本定义好的 Dataset 对象
        debug_indices = list(range(10)) 
        debug_dataset = Subset(train_dataset, debug_indices)
        
        # 2. 构建 DataLoader
        # 注意：batch_size 设小一点 (比如 2 或 4),方便观察每一条
        # shuffle=False 确保每次 epoch 顺序一致，方便对比
        train_loader = DataLoader(
            debug_dataset,
            batch_size=2, 
            shuffle=False, 
            collate_fn=lambda batch: collate_fn(batch, src_tokenizer, tgt_tokenizer, max_len=max_token)
        )
        
        
        NUM_EPOCHS = 500  
        
        # 打印数据，方便后面核对
        print("\n--- 待背诵的 10 条黄金数据 ---")
        for i in range(10):
            s, t = train_dataset[i]
            print(f"{i}: {s[:30]}... -> {t[:30]}...")
        print("-----------------------------\n")



    # 创建模型
    print("创建模型...")
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,  
        tgt_vocab_size=VOCAB_SIZE,
        max_len=max_token,
        dropout=0.1
    ).to(device)
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    '''
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    
    # 学习率调度器
    def lr_lambda(step, d_model=256, warmup_steps=4000):
        if step == 0:
            return 1e-7
        scale_factor = d_model ** -0.5

        return scale_factor*min(step ** -0.5, step * warmup_steps ** -1.5)
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    '''
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # 忽略padding的损失
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0)

    train_losses = []
    val_losses = []
    
    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        if DEBUG_MODE and (epoch + 1) % 20 == 0: # 每 20 个 epoch 打印一次，避免刷屏
            model.eval()
            print(f"\n🔮 [Epoch {epoch+1}] 模型背诵测试结果:")
            
            with torch.no_grad():
                # 重新遍历那 10 条数据
                for i in range(10):
                    # 获取原始文本
                    src_text, true_tgt_text = train_dataset[i]
                    
                    # 1. 准备输入 (模拟推理过程)
                    src_encoded = src_tokenizer.encode(src_text).ids
                    src_tensor = torch.tensor([src_encoded]).to(device) # [1, Seq_Len]
                    
                    # 2. 运行模型推理                    
                    tgt_ids = [tgt_tokenizer.token_to_id("[SOS]")]
                    max_len = 50
                    for _ in range(max_len):
                        tgt_tensor = torch.tensor([tgt_ids]).to(device)
                        output_logits = model(src_tensor, tgt_tensor) 
                        
                        # 取最后一个位置的预测
                        next_token_logits = output_logits[0, -1, :] 
                        next_token = torch.argmax(next_token_logits, dim=-1).item()
                        
                        if next_token == tgt_tokenizer.token_to_id("[EOS]"):
                            break
                        tgt_ids.append(next_token)
                    
                    # 3. 解码并对比
                    pred_text = tgt_tokenizer.decode(tgt_ids, skip_special_tokens=True)
                    
                    # 简单判断是否完全匹配
                    match = "✅ MATCH!" if pred_text.strip() == true_tgt_text.strip() else "❌ Mismatch"
                    
                    print(f"  Sample {i}:")
                    print(f"    真实: {true_tgt_text[:50]}...")
                    print(f"    预测: {pred_text[:50]}...")
                    
            model.train() # 切回训练模式
            print("-" * 30)

        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_losses.append(train_loss)
        '''
        if epoch % 2 == 0:
            bleu = calculate_bleu(model, train_loader, device, src_tokenizer, tgt_tokenizer)
            print(f"[Val]   BLEU: {bleu:.2f}")
        # 验证
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)


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
    '''


if __name__ == "__main__":
    main()
   