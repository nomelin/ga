import torch


def matching_loss(predicted, target):
    """
    计算双向最近距离的匹配成本。
    参数:
        predicted (Tensor): 预测种群坐标，形状为 (N, 2)。
        target (Tensor): 目标种群坐标，形状为 (N, 2)。
    返回:
        torch.Tensor: 匹配损失值。
    """
    # 计算每个预测点到目标点的最近距离
    dist_matrix = torch.cdist(predicted, target, p=2)  # 计算欧几里得距离矩阵
    min_pred_to_target = dist_matrix.min(dim=1).values.mean()  # 取最小距离的平均值作为预测点到目标点的距离
    min_target_to_pred = dist_matrix.min(dim=0).values.mean()  # 取最小距离的平均值作为目标点到预测点的距离
    return min_pred_to_target + min_target_to_pred


def diversity_loss(predicted):
    """
    计算预测点之间的多样性成本。
    参数:
        predicted (Tensor): 预测种群坐标，形状为 (N, 2)。
    返回:
        torch.Tensor: 多样性损失值。
    """
    dist_matrix = torch.cdist(predicted, predicted, p=2)  # 计算欧几里得距离矩阵
    # PyTorch 的自动微分机制要求计算图中的变量不可在梯度追踪过程中被修改。
    mask = torch.eye(dist_matrix.size(0), device=dist_matrix.device).bool()  # 对角线掩码
    dist_matrix = dist_matrix.masked_fill(mask, float('inf'))  # 替换对角线为 inf
    min_distances = dist_matrix.min(dim=1).values
    return -min_distances.mean()  # 取负值以鼓励更大的距离


def calculate_loss(predicted, target, alpha=1.0, beta=0.1):
    """
    总损失函数，结合匹配损失和多样性损失。
    参数:
        predicted (Tensor): 预测种群坐标，形状为 (N, 2)。
        target (Tensor): 目标种群坐标，形状为 (N, 2)。
        alpha (float): 匹配损失的权重。
        beta (float): 多样性损失的权重。
        Loss=α⋅Matching Loss+β⋅Diversity Loss
    返回:
        torch.Tensor: 总损失值。
    """
    match_loss = matching_loss(predicted, target)
    div_loss = diversity_loss(predicted)
    # print(f"match_loss: {match_loss}, div_loss: {div_loss}, total_loss: {alpha * match_loss + beta * div_loss}")
    return alpha * match_loss + beta * div_loss
