# -*- coding: utf-8 -*-

"""
# @Date: 2025/8/2 20:30
# @File: train_simple.py
# @Author: 10993
# @Description: 简单的Transformer训练示例 - 中文到英文翻译
"""

import torch
from torch import nn, optim

from Transformer import Transformer


def simple_training_example():
    """简单的训练示例，使用'我爱水课' -> 'I love easy course'"""

    # 定义词汇表
    src_vocab = {'<pad>': 0, '<unk>': 1, '我': 2, '爱': 3, '水': 4, '课': 5}
    tgt_vocab = {'<pad>': 0, '<unk>': 1, '<start>': 2, '<end>': 3, 'I': 4, 'love': 5, 'easy': 6, 'course': 7}

    # 反向词汇表，用于解码
    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}

    # 创建模型参数
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    d_model = 128  # 减小模型尺寸以便快速训练
    num_heads = 4
    num_encoders = 2
    num_decoders = 2
    max_len = 50

    # 创建模型
    model = Transformer(
        d_model=d_model,
        num_heads=num_heads,
        num_encoders=num_encoders,
        num_decoders=num_decoders,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=max_len
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略pad token
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 准备训练数据
    # 输入: 我 爱 水 课
    src_tensor = torch.tensor([[src_vocab['我'], src_vocab['爱'], src_vocab['水'], src_vocab['课']]], dtype=torch.long)

    # 目标输入: <start> I love easy course
    tgt_input_tensor = torch.tensor(
        [[tgt_vocab['<start>'], tgt_vocab['I'], tgt_vocab['love'], tgt_vocab['easy'], tgt_vocab['course']]],
        dtype=torch.long)

    # 期望输出: I love easy course <end>
    tgt_output_tensor = torch.tensor(
        [[tgt_vocab['I'], tgt_vocab['love'], tgt_vocab['easy'], tgt_vocab['course'], tgt_vocab['<end>']]],
        dtype=torch.long)

    print("开始训练...")
    model.train()

    # 进行多次迭代训练
    for epoch in range(100):
        optimizer.zero_grad()

        # 前向传播
        output = model(src_tensor, tgt_input_tensor)

        # 计算损失
        loss = criterion(output.reshape(-1, tgt_vocab_size), tgt_output_tensor.reshape(-1))

        # 反向传播
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("训练完成!")

    # 简单推理测试
    print("\n进行推理测试...")
    model.eval()
    with torch.no_grad():
        # 编码器输入仍然是"我爱水课"
        src_test = src_tensor

        # 解码器初始输入只有<start>
        tgt_test = torch.tensor([[tgt_vocab['<start>']]], dtype=torch.long)

        # 逐步生成翻译
        for i in range(10):  # 最多生成10个词
            output = model(src_test, tgt_test)
            print(f"Debug: output shape: {output.shape}")

            # 如果输出是4维的，需要去掉多余的维度
            if output.dim() == 4:
                output = output.squeeze(1)  # 去掉第二个维度

            print(f"Debug: output shape after squeeze: {output.shape}")

            # 获取最后一个词的预测
            pred = output[:, -1, :].argmax(dim=-1)  # 得到(batch_size,)的张量
            print(f"Debug: pred shape: {pred.shape}")
            print(f"Debug: pred value: {pred}")
            print(f"Debug: tgt_test shape before cat: {tgt_test.shape}")

            # 将pred扩展为与tgt_test相同的维度
            pred_extended = pred.unsqueeze(1)  # 变为(batch_size, 1)

            print(f"Debug: pred_extended shape: {pred_extended.shape}")
            print(f"Debug: pred_extended: {pred_extended}")

            tgt_test = torch.cat([tgt_test, pred_extended], dim=1)
            print(f"Debug: tgt_test shape after cat: {tgt_test.shape}")

            # 如果预测到结束符，停止生成
            if pred.item() == tgt_vocab['<end>']:
                break

        # 解码生成的序列
        generated_tokens = tgt_test.squeeze().tolist()
        if isinstance(generated_tokens, int):
            generated_tokens = [generated_tokens]

        generated_words = [tgt_vocab_inv.get(token, '<unk>') for token in generated_tokens]
        print(f"输入: 我爱水课")
        print(f"输出: {' '.join(generated_words[1:-1])}")  # 去掉<start>和<end>

    return model


if __name__ == "__main__":
    simple_training_example()
