import importlib.util
import os
import threading
from queue import Queue

from flask import Flask, request, jsonify
from flask_cors import CORS

from ga.nsgaii import nsga2
from lib import global_var
from lib.visual import ObjectiveVisualizer

app = Flask(__name__)
# 允许所有来源跨域访问
CORS(app)  # 添加 CORS 支持
# 全局队列，用于存储可视化数据
visualization_queue = Queue()


@app.route('/hello', methods=['GET'])
def hello():
    """
    HTTP 接口：测试接口。
    """
    return jsonify({"status": "success", "message": "Hello, world!"})


@app.route('/start', methods=['POST'])
def start():
    """
    HTTP 接口：启动 NSGA-II 算法。
    """
    print("调用 /start 接口")
    data = request.get_json()
    func_file_name = data.get("func_file", None)
    if not func_file_name:
        print("Missing func_file parameter")
        return jsonify({"status": "error", "message": "Missing func_file parameter"}), 400

    try:
        print(f"func_file: {func_file_name}")
        # 动态加载函数文件
        funcs_dir = "../funcs"
        file_path = os.path.join(funcs_dir, f"{func_file_name}.py")
        spec = importlib.util.spec_from_file_location(func_file_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"Loaded module: {module}")

        # 从文件中加载函数和变量范围
        funcs = getattr(module, "funcs", None)
        variable_ranges = getattr(module, "variable_range", None)
        is_dynamic = getattr(module, "is_dynamic", False)
        if not funcs or not variable_ranges:
            return jsonify({"status": "error", "message": "Invalid function file structure"}), 400

        precision = data.get("precision", 0.01)
        pop_size = data.get("pop_size", 100)
        num_generations = data.get("num_generations", 100)
        crossover_rate = data.get("crossover_rate", 0.9)
        mutation_rate = data.get("mutation_rate", 0.01)
        # 打印调试信息
        print(f"Loaded funcs: {funcs}, variable_ranges: {variable_ranges}, is_dynamic: {is_dynamic}")

        # 启动算法
        visualizer = ObjectiveVisualizer(
            variable_ranges=variable_ranges,
            visual_mode=2,
            resolution=100,
            queue=visualization_queue)
        print("visualizer created.")
        threading.Thread(
            target=nsga2,
            args=(visualizer, {0: [funcs, ['min', 'min']]}, variable_ranges, precision, pop_size, num_generations,
                  crossover_rate, mutation_rate, is_dynamic)
        ).start()
        print("nsga2 started.")
        return jsonify({"status": "success", "message": "NSGA-II started."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/poll', methods=['GET'])
def poll_visualization():
    """
    HTTP 接口：轮询获取可视化数据。
    """
    if not visualization_queue.empty():
        data = visualization_queue.get()  # 获取并删除队列中的第一个元素
        # print(f"返回可视化数据, data: {data}")
        print(f"返回可视化数据")
        return jsonify({"status": "success", "has_new_data": True, "data": data})
    else:
        print(f"没有可视化数据")
        return jsonify({"status": "success", "has_new_data": False, "data": None})


@app.route('/stop', methods=['POST'])
def stop():
    """
    HTTP 接口：结束算法运行，并清空队列缓存。
    """
    try:
        # 算法运行在一个单独的线程中，可以通过设置全局变量或线程标志结束算法逻辑
        global_var.set_algorithm_running(False)
        # 清空队列中的所有数据
        while not visualization_queue.empty():
            visualization_queue.get()

        print("NSGA-II 停止运行，队列已清空。")
        return jsonify({"status": "success", "message": "算法已停止，队列已清空。"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/queryFuncs', methods=['GET'])
def query_funcs():
    """
    HTTP 接口：获取函数文件列表。
    """
    try:
        # 获取 funcs 文件夹中的所有 Python 文件
        funcs_dir = "../funcs"
        func_files = [f[:-3] for f in os.listdir(funcs_dir) if f.endswith(".py")]
        return jsonify({"status": "success", "func_files": func_files})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False)
