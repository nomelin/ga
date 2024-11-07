import numpy as np
from ga import nsgaii
from lib import eval_convergence_metric
from lib import eval_diversity_metric
from lib import visual
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination


# 定义目标函数
def f1(x1, x2):
    return x1 ** 2 + x2 ** 2


def f2(x1, x2):
    return (x1 - 2) ** 2 + x2 ** 2


# 定义变量范围
variable_ranges = [(-2, 2), (-2, 2)]

# 定义目标函数
funcs = [f1, f2]

# 创建可视化对象
visualizer1 = visual.ObjectiveVisualizer(funcs, variable_ranges, 500)
# visualizer2 = visual.ObjectiveVisualizer(funcs, variable_ranges, 500)
# 调用 NSGA-II 算法
final_solutions_nsga2 = nsgaii.nsga2(
    visualizer=visualizer1,
    funcs=funcs,
    variable_ranges=variable_ranges,
    precision=0.01,
    pop_size=100,
    num_generations=10
)

print("NSGA-II Final solutions (decoded):")
for solution in final_solutions_nsga2:
    print(solution)

visualizer2 = visual.ObjectiveVisualizer(funcs, variable_ranges, 500)


# 定义用于SPEA2的自定义优化问题
class CustomProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, xl=np.array([-2, -2]), xu=np.array([2, 2]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1_vals = np.square(x[:, 0]) + np.square(x[:, 1])
        f2_vals = np.square(x[:, 0] - 2) + np.square(x[:, 1])
        out["F"] = np.column_stack([f1_vals, f2_vals])


# 使用SPEA2算法求解
problem = CustomProblem()
termination = get_termination("n_gen", 10)
algorithm = SPEA2(pop_size=100)
result_spea2 = minimize(problem, algorithm, termination=termination, seed=1, verbose=False)

# 输出SPEA2算法的最终解
final_solutions_spea2 = result_spea2.X
final_solutions_spea2_values = []
for solution in final_solutions_spea2:
    final_solutions_spea2_values.append(problem.evaluate(solution))

print("SPEA2 Final solutions (decoded):")
for solution in final_solutions_spea2:
    print(solution)

# 生成理论帕累托最优解集





def is_dominated(point1, point2):
    """
    判断解point1是否支配解point2。
    如果point1在所有目标函数上都小于或等于point2，并且在至少一个目标函数上严格小于point2，则point1支配point2。
    """
    f1_1, f2_1 = point1
    f1_2, f2_2 = point2
    return f1_1 <= f1_2 and f2_1 <= f2_2 and (f1_1 < f1_2 or f2_1 < f2_2)


def generate_theoretical_pareto_front(num_points=100):
    """
    通过暴力法生成理论的帕累托前沿，展示两个目标函数 f1(x1, x2) 和 f2(x1, x2) 的权衡。
    生成的帕累托前沿是在给定自变量范围 [-2, 2] 内的理论最优解。
    """
    # 生成均匀分布的x1和x2值，这些值决定了帕累托前沿
    x1_values = np.linspace(-2, 2, num_points)
    x2_values = np.linspace(-2, 2, num_points)

    # 用来保存不被支配的解
    pareto_front = []

    # 遍历每一个 x1 和 x2 的组合
    for x1 in x1_values:
        for x2 in x2_values:
            f1_value = x1 ** 2 + x2 ** 2
            f2_value = (x1 - 2) ** 2 + x2 ** 2

            # 创建当前解
            current_point = [f1_value, f2_value]

            # 检查当前解是否被其他解支配
            dominated = False
            for existing_point in pareto_front:
                if is_dominated(existing_point, current_point):
                    dominated = True
                    break
                elif is_dominated(current_point, existing_point):
                    # 如果当前点支配了已有的点，则移除已有的点
                    pareto_front.remove(existing_point)

            # 如果当前解没有被支配，则加入帕累托前沿
            if not dominated:
                pareto_front.append(current_point)

    # 将结果转换为 NumPy 数组
    pareto_front = np.array(pareto_front)

    return pareto_front


# 调用函数生成理论帕累托前沿
theoretical_front = generate_theoretical_pareto_front()

# 计算NSGA-II的收敛度量和多样性度量
convergence_calculator_nsga2 = eval_convergence_metric.ConvergenceMetricCalculator(theoretical_front,
                                                                                   final_solutions_nsga2, 500)
convergence_metric_nsga2 = convergence_calculator_nsga2.calculate_convergence_metric()
print("NSGA-II Convergence metric: ", convergence_metric_nsga2)

diversity_calculator_nsga2 = eval_diversity_metric.DiversityMetricCalculator(final_solutions_nsga2)
diversity_metric_nsga2 = diversity_calculator_nsga2.get_diversity_metric()
print("NSGA-II Diversity metric: ", diversity_metric_nsga2)

# 计算SPEA2的收敛度量和多样性度量
convergence_calculator_spea2 = eval_convergence_metric.ConvergenceMetricCalculator(theoretical_front,
                                                                                   final_solutions_spea2, 500)
convergence_metric_spea2 = convergence_calculator_spea2.calculate_convergence_metric()
print("SPEA2 Convergence metric: ", convergence_metric_spea2)

diversity_calculator_spea2 = eval_diversity_metric.DiversityMetricCalculator(final_solutions_spea2)
diversity_metric_spea2 = diversity_calculator_spea2.get_diversity_metric()
print("SPEA2 Diversity metric: ", diversity_metric_spea2)
visualizer2.draw_populations(final_solutions_spea2_values, 10)


print(theoretical_front)
input("1")
