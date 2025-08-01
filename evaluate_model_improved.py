from sentence_transformers import SentenceTransformer, util
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def load_test_data(file_path="data.jsonl"):
    """加载测试数据"""
    test_queries = []
    test_positives = []
    test_negatives = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            test_queries.append(data["query"])
            test_positives.append(data["positive"])
            test_negatives.append(data["negative"])
    
    return test_queries, test_positives, test_negatives

def evaluate_model(model, test_queries, test_positives, test_negatives):
    """评估模型效果"""
    positive_scores = []
    negative_scores = []
    
    for i, query in enumerate(test_queries):
        # 编码查询
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # 计算与正样本的相似度
        for pos in test_positives[i]:
            pos_embedding = model.encode(pos, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, pos_embedding).item()
            positive_scores.append(similarity)
        
        # 计算与负样本的相似度
        for neg in test_negatives[i]:
            neg_embedding = model.encode(neg, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(query_embedding, neg_embedding).item()
            negative_scores.append(similarity)
    
    return positive_scores, negative_scores

def calculate_metrics(positive_scores, negative_scores):
    """计算评估指标"""
    # 计算平均分数
    avg_positive = np.mean(positive_scores)
    avg_negative = np.mean(negative_scores)
    
    # 计算分数差异
    score_diff = avg_positive - avg_negative
    
    # 计算准确率（正样本分数 > 负样本分数的比例）
    correct = 0
    total = 0
    
    for pos_score in positive_scores:
        for neg_score in negative_scores:
            total += 1
            if pos_score > neg_score:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    
    # 计算标准差
    std_positive = np.std(positive_scores)
    std_negative = np.std(negative_scores)
    
    return {
        "avg_positive": avg_positive,
        "avg_negative": avg_negative,
        "score_difference": score_diff,
        "accuracy": accuracy,
        "std_positive": std_positive,
        "std_negative": std_negative
    }

def plot_scores(positive_scores, negative_scores, model_name):
    """绘制分数分布图"""
    plt.figure(figsize=(10, 6))
    plt.hist(positive_scores, alpha=0.7, label='Positive pairs', bins=30, color='green')
    plt.hist(negative_scores, alpha=0.7, label='Negative pairs', bins=30, color='red')
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('Frequency')
    plt.title(f'Similarity Score Distribution - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{model_name}_scores_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 加载测试数据
    print("加载测试数据...")
    test_queries, test_positives, test_negatives = load_test_data()
    
    # 测试原始模型
    print("测试原始 BGE-M3 模型...")
    original_model = SentenceTransformer("./bge-m3", device="cuda")
    orig_pos_scores, orig_neg_scores = evaluate_model(original_model, test_queries, test_positives, test_negatives)
    orig_metrics = calculate_metrics(orig_pos_scores, orig_neg_scores)
    
    # 测试改进后的微调模型
    print("测试改进后的微调模型...")
    try:
        finetuned_model = SentenceTransformer("./bge-m3-finetuned-improved", device="cuda")
        ft_pos_scores, ft_neg_scores = evaluate_model(finetuned_model, test_queries, test_positives, test_negatives)
        ft_metrics = calculate_metrics(ft_pos_scores, ft_neg_scores)
        
        # 打印结果对比
        print("\n" + "="*60)
        print("模型效果对比")
        print("="*60)
        print(f"{'指标':<15} {'原始模型':<15} {'微调模型':<15} {'改进':<15}")
        print("-"*60)
        print(f"{'正样本平均分':<15} {orig_metrics['avg_positive']:<15.4f} {ft_metrics['avg_positive']:<15.4f} {ft_metrics['avg_positive'] - orig_metrics['avg_positive']:<15.4f}")
        print(f"{'负样本平均分':<15} {orig_metrics['avg_negative']:<15.4f} {ft_metrics['avg_negative']:<15.4f} {ft_metrics['avg_negative'] - orig_metrics['avg_negative']:<15.4f}")
        print(f"{'分数差异':<15} {orig_metrics['score_difference']:<15.4f} {ft_metrics['score_difference']:<15.4f} {ft_metrics['score_difference'] - orig_metrics['score_difference']:<15.4f}")
        print(f"{'准确率':<15} {orig_metrics['accuracy']:<15.4f} {ft_metrics['accuracy']:<15.4f} {ft_metrics['accuracy'] - orig_metrics['accuracy']:<15.4f}")
        print(f"{'正样本标准差':<15} {orig_metrics['std_positive']:<15.4f} {ft_metrics['std_positive']:<15.4f} {ft_metrics['std_positive'] - orig_metrics['std_positive']:<15.4f}")
        print(f"{'负样本标准差':<15} {orig_metrics['std_negative']:<15.4f} {ft_metrics['std_negative']:<15.4f} {ft_metrics['std_negative'] - orig_metrics['std_negative']:<15.4f}")
        
        # 绘制对比图
        plot_scores(orig_pos_scores, orig_neg_scores, "Original_BGE_M3")
        plot_scores(ft_pos_scores, ft_neg_scores, "Improved_Finetuned_BGE_M3")
        
        # 判断改进效果
        print("\n" + "="*60)
        print("改进效果分析")
        print("="*60)
        
        if ft_metrics['score_difference'] > orig_metrics['score_difference']:
            improvement = ft_metrics['score_difference'] - orig_metrics['score_difference']
            print(f"✅ 微调成功！分数差异增加了 {improvement:.4f}")
        else:
            print("❌ 微调效果不明显，分数差异没有增加")
        
        if ft_metrics['accuracy'] > orig_metrics['accuracy']:
            acc_improvement = ft_metrics['accuracy'] - orig_metrics['accuracy']
            print(f"✅ 准确率提升了 {acc_improvement:.4f}")
        else:
            print("❌ 准确率没有提升")
        
    except Exception as e:
        print(f"微调模型加载失败: {e}")
        print("请确保已经完成改进的模型微调训练")

if __name__ == "__main__":
    main() 