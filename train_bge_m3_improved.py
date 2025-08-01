from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
import torch
from torch.optim import AdamW
import random

# 设置随机种子确保可重现性   确保设计随机数的操作每次运行都一致
random.seed(42)
torch.manual_seed(42)

def augment_data(original_data, num_augmentations=50):
    """数据增强：通过修改现有数据生成更多训练样本"""
    augmented_samples = []
    
    # 模板库，用于生成更多样本
    templates = [
        {"query": "什么是{concept}", "positive": ["{concept}是{description}"], "negative": ["{unrelated}"]},
        {"query": "如何{action}", "positive": ["{action}的方法和步骤"], "negative": ["{unrelated}"]},
        {"query": "{concept}怎么用", "positive": ["{concept}的使用方法和技巧"], "negative": ["{unrelated}"]},
    ]
    
    # 概念库
    concepts = [ #中间的参数目前用不到
        ("机器学习", "人工智能的一个分支", "深度学习", "如何做菜"),
        ("Python", "一种编程语言", "Java", "如何开车"),
        ("神经网络", "机器学习模型", "支持向量机", "如何游泳"),
        ("数据挖掘", "从数据中发现模式", "数据清洗", "如何画画"),
        ("自然语言处理", "让计算机理解语言", "计算机视觉", "如何弹琴"),
        ("强化学习", "通过交互学习", "监督学习", "如何跳舞"),
        ("卷积神经网络", "用于图像处理的网络", "循环神经网络", "如何唱歌"),
        ("梯度下降", "优化算法", "随机森林", "如何写作"),
        ("过拟合", "模型训练问题", "欠拟合", "如何摄影"),
        ("交叉验证", "模型评估方法", "特征选择", "如何烹饪")
    ]
    
    # 生成增强数据
    for _ in range(num_augmentations):
        template = random.choice(templates) #随机选择一个模板 下面就是随机选择知识点
        concept, description, _, unrelated = random.choice(concepts)
        
        # 随机选择模板并填充
        if "{concept}" in template["query"]: #如果模板使用concept占位 不错的用法  
            query = template["query"].format(concept=concept) #这些都是根据模板的内容来写的   positive是数组 里面只有一个元素 所以用第一个来填充
            positive = [template["positive"][0].format(concept=concept, description=description)]
            negative = [template["negative"][0].format(unrelated=unrelated)]
        else:
            action = f"学习{concept}"
            query = template["query"].format(action=action)
            positive = [template["positive"][0].format(action=action)]
            negative = [template["negative"][0].format(unrelated=f"学习{unrelated}")]
        
        augmented_samples.append({ #结构是 问题  一个或多个正答案  一个负答案
            "query": query,
            "positive": positive,
            "negative": [negative[0]]
        })
    
    return augmented_samples

def load_and_augment_data(file_path="data.jsonl"):
    """加载原始数据并进行增强"""
    # 加载原始数据
    original_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            original_data.append(json.loads(line))
    
    print(f"原始数据样本数: {len(original_data)}")
    
    # 数据增强
    augmented_data = augment_data(original_data, num_augmentations=100)
    print(f"增强后数据样本数: {len(augmented_data)}")
    
    # 合并原始数据和增强数据
    all_data = original_data + augmented_data
    print(f"总训练样本数: {len(all_data)}")
    
    return all_data

def create_training_samples(data):
    """创建训练样本"""
    train_samples = []
    
    for item in data:
        query = item["query"]
        positives = item["positive"]
        negatives = item["negative"]
        
        # 为正样本创建训练对
        for pos in positives:
            train_samples.append(InputExample(texts=[query, pos], label=1.0))
        
        # 为负样本创建训练对
        for neg in negatives:
            train_samples.append(InputExample(texts=[query, neg], label=0.0))
    
    print(f"总训练对数量: {len(train_samples)}")
    return train_samples

def main():
    # 加载 BGE-M3 基础模型
    print("加载 BGE-M3 模型...")
    model = SentenceTransformer("bge-m3", device="cuda")
    
    # 加载和增强数据
    print("加载和增强训练数据...")
    all_data = load_and_augment_data()
    
    # 创建训练样本
    train_samples = create_training_samples(all_data)
    
    # 创建 DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)
    
    # 使用 MultipleNegativesRankingLoss 替代 CosineSimilarityLoss
    # 这个损失函数更适合对比学习
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    
    # 微调训练
    print("开始微调训练...")
    model.fit( # 注意这个函数没有optimizer这个参数 优化器是默认的
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,  # 增加训练轮数
        warmup_steps=100,  # 增加warmup步数
        show_progress_bar=True,
        output_path="./bge-m3-finetuned-improved",
        save_best_model=True,  # 保存最佳模型
    )
    
    print("训练完成！模型已保存到 ./bge-m3-finetuned-improved")

if __name__ == "__main__":
    main() 