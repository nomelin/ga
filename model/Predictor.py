import torch

from lib.plot_solution_sets import savetogif
from model.PopulationPredictorLSTM import load_model, PopulationPredictorLSTM
from model.train import load_population_data


def predict(model, x, device='cuda'):
    """
    使用模型预测种群数据。
    参数:
        model (nn.Module): 预测模型。
        x [[]]: 输入数据，形状为 (num_individuals, input_dim)。
        device (str): 预测模型的设备（如 'cpu' 或 'cuda'）。
    返回:
        [[]]: 预测的种群数据，形状为 (num_individuals, output_dim)。
    """
    model.eval()
    x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device)
    with torch.no_grad():
        output = model(x)
    output = output.cpu().numpy()
    return output


if __name__ == '__main__':
    json_file = '../data/optimal_solutions_dy3.json'
    samples = load_population_data(json_file)
    weight_path = "weight/model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 初始化模型
    model = PopulationPredictorLSTM()
    # 加载模型权重
    load_model(model, weight_path, device)
    # model.to(device)
    current = []
    predicted = []
    for sample in samples:
        input, output = sample[0].cpu().numpy(), sample[1].cpu().numpy()
        pred_output = predict(model, input, device)
        print(
            f"size of input: {input.shape}, size of output: {output.shape} , size of pred_output: {pred_output.shape}")
        current.append(output)
        predicted.append(pred_output)
    savetogif(current, predicted)
