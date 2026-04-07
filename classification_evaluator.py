"""
classification_evaluator.py

分类模型评估与可视化模块
适用于二分类机器学习模型（如 RandomForest, XGBoost, LightGBM 等）

--------------------------------------------------
📌功能目录：
# 📌分类性能评估
- print_classification_report           输出分类指标报告
- plot_confusion_matrix                 绘制混淆矩阵热图
- plot_roc_curve                        绘制 ROC 曲线
- plot_pr_curve                         绘制 Precision-Recall 曲线
- plot_probability_distribution         绘制正负类概率分布图
- plot_threshold_metrics                不同阈值下的 Precision、Recall 和 F1曲线图，辅助选择最优决策阈值

# 📌特征重要性分析
- plot_feature_importance_comparison    Gini vs. Permutation Importance 对比图
- plot_feature_importance_auto          绘制特征重要性图（使用于集合模型/多个模型）

# 📌特征结构探索与降维
- plot_spearman_clustermap              绘制 Spearman 相关性 + 聚类热力图
- select_features_by_clustering_and_evaluate 聚类选代表特征并重新训练模型

# 📌样本分布与数据结构
- plot_class_distribution               查看标签类别分布
--------------------------------------------------
"""

# 通用库
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import pandas as pd  # 导入Pandas库，用于数据处理和分析

# 模型评估与指标
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# 特征重要性评估
from sklearn.inspection import permutation_importance

# 特征结构分析（聚类与相关性）
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# ==============================================
# 📊 分类性能评估
# ==============================================
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    绘制并显示混淆矩阵热图。

    参数：
        y_true: array-like，真实标签
        y_pred: array-like，预测标签
        save_path: str，可选；保存图片的路径（如 'result/conf_matrix.png'）
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predicted: 0", "Predicted: 1"],
                yticklabels=["True: 0", "True: 1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")    
    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def plot_roc_curve(y_true, y_scores, save_path=None):
    """
    绘制 ROC 曲线并显示 AUC 值。

    参数：
        y_true: array-like，真实标签（0/1）
        y_scores: array-like，预测为正类的概率（通常来自 predict_proba[:, 1]）
        save_path: str，可选；保存图像路径（如 'result/roc.png'）
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()



def plot_pr_curve(y_true, y_scores, save_path=None):
    """
    绘制 Precision-Recall 曲线并显示 PR AUC。

    参数：
        y_true: array-like，真实标签（0/1）
        y_scores: array-like，预测为正类的概率（通常来自 predict_proba[:, 1]）
        save_path: str，可选；保存图像路径（如 'result/pr_curve.png'）
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def print_classification_report(y_true, y_pred):
    """
    控制台输出分类报告（包含精确率、召回率、F1 等）。

    参数：
        y_true: array-like，真实标签（0/1）
        y_pred: array-like，模型预测标签（0/1）
    """
    print("分类报告（Precision, Recall, F1）:")
    print(classification_report(y_true, y_pred, digits=4))
    
    
def plot_probability_distribution(y_true, y_prob, save_path=None):
    """
    绘制模型预测概率分布图，显示不同真实类别（y_true）下的概率密度。

    参数：
        y_true: array-like，真实标签（0/1）
        y_prob: array-like，预测为正类的概率（如 predict_proba[:, 1]）
    """
    import pandas as pd
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="y_prob", hue="y_true", bins=20, kde=True,
                 stat="density", common_norm=False)
    plt.xlabel("Predicted Probability")
    plt.title("Probability Distribution by Class")
    plt.grid(True)
    plt.tight_layout()
    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    
    
def plot_threshold_metrics(y_true, y_prob, save_path=None):
    """
    绘制不同阈值下的 Precision、Recall 和 F1-score 曲线图，用于辅助选择最优决策阈值。

    参数：
        y_true: array-like，真实标签（0/1）
        y_prob: array-like，预测为正类的概率（如 model.predict_proba[:, 1]）
        save_path: str，可选；图像保存路径（如 'result/threshold_metrics.png'）

    效果：
        展示模型在不同概率阈值下的分类性能指标趋势（precision / recall / F1）
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.plot(thresholds, f1[:-1], label='F1 Score')
    plt.axvline(best_threshold, color='gray', linestyle='--', label=f"Best F1 @ {best_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.title("Threshold vs Precision / Recall / F1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    
    

# ==============================================
# 🧠 特征重要性分析
# ==============================================

def plot_feature_importance(model, feature_names=None, top_n=20, save_path=None):
    """
    绘制模型的特征重要性条形图（基于 feature_importances_ 属性）。

    参数：
        model: 拥有 feature_importances_ 的模型（如 RandomForest/XGBoost）
        feature_names: list，可选；特征名称列表（默认按索引命名）
        top_n: int，显示前 top_n 个重要特征
        save_path: str，可选；图像保存路径（如 'result/feature_importance.png'）
    """
    if not hasattr(model, "feature_importances_"):
        print("模型不支持 feature_importances_ 属性")
        return

    importances = model.feature_importances_
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    import pandas as pd
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x="Importance",
        y="Feature",
        color='#1f77b4',
        edgecolor='none')
    
    plt.title("Top Feature Importances")
    plt.grid(True)
    plt.tight_layout()

    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    

    
def plot_feature_importance_auto(model_or_models, feature_names=None, top_n=20, save_path=None, show_std=False):
    """
    绘制模型或模型集成的特征重要性图。

    参数：
        model_or_models: 单个模型（支持 feature_importances_）或多个模型组成的列表
        feature_names: list[str]，可选，特征名称
        top_n: int，显示前 top_n 个特征
        save_path: str，可选，图像保存路径
        show_std: bool，是否绘制标准差误差条（仅多模型时有效）
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 判断是单个模型还是模型列表
    if isinstance(model_or_models, list):
        models = model_or_models
        for i, m in enumerate(models):
            if not hasattr(m, "feature_importances_"):
                raise ValueError(f"第 {i} 个模型不支持 feature_importances_")
        all_importances = np.array([m.feature_importances_ for m in models])
        mean_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)
    else:
        model = model_or_models
        if not hasattr(model, "feature_importances_"):
            raise ValueError("模型不支持 feature_importances_ 属性")
        mean_importance = model.feature_importances_
        std_importance = None

    # 构造特征名
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(mean_importance))]

    # 构造 DataFrame
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_importance
    })

    if std_importance is not None and show_std:
        df["Std"] = std_importance
    else:
        df["Std"] = 0.0  # 占位

    df = df.sort_values("Importance", ascending=False).head(top_n)

    # 绘图
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Importance", y="Feature", xerr=df["Std"] if show_std else None, palette="viridis")
    title = "Feature Importances"
    if isinstance(model_or_models, list):
        title += f" (Mean of {len(model_or_models)} Models)"
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    


def plot_feature_importance_comparison(model, X, y, title=None, n_repeats=10,save_path=None):
    """
    可视化 Gini importance 与 Permutation importance 的对比图。

    参数：
        model: 拥有 feature_importances_ 的训练好模型（如 RandomForest、XGBoost、LightGBM）
        X: pandas.DataFrame，训练特征集，必须包含列名
        y: array-like，训练标签
        title: str，可选图标题
        n_repeats: int，Permutation Importance 重复打乱次数（默认 10）

    效果：
        左图：Gini importance（模型结构中的分裂频率）
        右图：Permutation importance（特征扰动对性能的实际影响）
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("模型必须有 feature_importances_ 属性（如 RandomForest）")
    import pandas as pd
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X 必须是 pandas.DataFrame，必须包含列名")
    
    # Gini importance
    mdi_importances = pd.Series(model.feature_importances_, index=X.columns)

    # Permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=0)

    # 图形绘制
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # 左图：Gini importance 条形图
    mdi_importances.sort_values().plot.barh(ax=ax1)
    ax1.set_title("Gini importance")
    ax1.set_xlabel("Gini importance")

    # 右图：Permutation importance 箱线图
    sorted_idx = perm_importance.importances_mean.argsort()
    ax2.boxplot(
        perm_importance.importances[sorted_idx].T,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgray", color="black"),
        medianprops=dict(color="orange"),
        tick_labels=X.columns[sorted_idx]  # ✅ 新版 Matplotlib 用 tick_labels
    )
    ax2.set_title("Permutation Importance (train set)")
    ax2.set_xlabel("Decrease in accuracy score")

    # 图标题
    fig.suptitle(title or "Impurity-based vs. Permutation Importances")
    fig.tight_layout()
    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    

# ==============================================
# 🧩 特征结构探索与降维
# ==============================================

def plot_spearman_clustermap(X, title=None, save_path=None, figsize=(12, 8)):
    """
    绘制特征间 Spearman 相关性 + 层次聚类树状图。

    参数：
        X: pandas.DataFrame，特征集
        title: str，可选图标题
        save_path: str，可选；保存图像路径（如 'result/spearman_clustermap.png'）
        figsize: tuple，图像尺寸，默认 (12, 8)
    """
    # Step 1: 计算 Spearman 相关系数矩阵
    corr = spearmanr(X).correlation
    corr = (corr + corr.T) / 2  # 保证对称性
    np.fill_diagonal(corr, 1)

    # Step 2: 构造距离矩阵并进行聚类
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    # Step 3: 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左图：树状图
    dendro = hierarchy.dendrogram(
        dist_linkage,
        labels=X.columns.to_list(),
        ax=ax1,
        leaf_rotation=90
    )
    ax1.set_title("Hierarchical Clustering")

    # 右图：热力图，按聚类结果重排序
    dendro_idx = np.arange(len(dendro["ivl"]))
    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]],
               cmap="coolwarm", aspect="auto")
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical", fontsize=8)
    ax2.set_yticklabels(dendro["ivl"], fontsize=8)
    ax2.set_title("Spearman Correlation")

    # 图标题与保存
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def select_features_by_clustering_and_evaluate(model, X_train, X_test, y_train, y_test, distance_threshold=1.0):
    """
    基于 Spearman 层次聚类自动选择代表特征，并评估模型在测试集上的表现。

    参数：
        model: 已实例化但尚未训练的分类模型（需支持 predict_proba）
        X_train: DataFrame，训练集特征
        X_test: DataFrame，测试集特征
        y_train: array-like，训练集标签
        y_test: array-like，测试集标签
        distance_threshold: float，聚类距离阈值（越小越严格）

    返回：
        selected_features_names: list[str]，去除冗余后保留的特征名
    """
    # Step 1: Spearman 聚类分析
    corr = spearmanr(X_train).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    cluster_ids = hierarchy.fcluster(dist_linkage, distance_threshold, criterion="distance")

    # Step 2: 每个 cluster 保留第一个特征
    cluster_to_features = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_to_features[cluster_id].append(idx)
    selected_indices = [v[0] for v in cluster_to_features.values()]
    selected_features_names = X_train.columns[selected_indices]

    # Step 3: 使用精简特征训练模型
    X_train_sel = X_train[selected_features_names]
    X_test_sel = X_test[selected_features_names]
    model.fit(X_train_sel, y_train)

    # Step 4: 预测与评估
    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:, 1]  # 正类概率

    print(f"使用 {len(selected_features_names)} 个特征进行建模，评估结果如下：")
    print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision      : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall         : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score       : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC        : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR AUC         : {average_precision_score(y_test, y_prob):.4f}")

    return selected_features_names


# ==============================================
# 📦 样本分布与数据结构
# ==============================================

def plot_class_distribution(y, save_path=None):
    """
    绘制分类任务中样本标签的类别分布柱状图。

    参数：
        y: array-like，样本的分类标签（如 0/1）
        save_path: str，可选；图像保存路径（如 'result/class_distribution.png'）
    """
    import numpy as np
    labels, counts = np.unique(y, return_counts=True)

    plt.figure()
    plt.bar(labels, counts, tick_label=[f"Class {i}" for i in labels])
    plt.title("Class Distribution")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()

    # === 关键改动：保存透明背景图 ===
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
