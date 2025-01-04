<template>
  <div style="display: flex; height: 100vh; padding: 20px; font-weight: bold;">
    <!-- 左侧区域 -->
    <div
        style="width: 25%; display: flex; flex-direction: column; justify-content: space-between; padding: 10px; border-right: 1px solid #ddd;">
      <!-- 表单部分 -->
      <div>
        <el-form :model="form" label-width="80px" style="width: 100%">
          <el-form-item label="算法">
            <el-select v-model="form.algorithm" placeholder="选择算法" style="width: 100%">
              <el-option v-for="item in algorithmOptions" :key="item.value" :label="item.label" :value="item.value"/>
            </el-select>
          </el-form-item>
          <el-form-item label="函数文件">
            <el-select v-model="form.funcFile" placeholder="选择函数文件" style="width: 100%">
              <el-option
                  v-for="item in funcFileOptions"
                  :key="item.value"
                  :label="item.label"
                  :value="item.value"
              />
            </el-select>
          </el-form-item>

          <el-form-item label="种群大小">
            <el-input-number v-model="form.popSize" :min="10" :max="500" style="width: 100%"/>
          </el-form-item>

          <el-form-item label="代数">
            <el-input-number v-model="form.generations" :min="1" :max="200" style="width: 100%"/>
          </el-form-item>

          <el-button type="primary" @click="startAlgorithm" style="width: 100%;">启动算法</el-button>
        </el-form>
      </div>

      <!-- 左下部分 -->
      <div style="padding-top: 20px;">
        <!-- 轮询状态显示-->
        <div v-if="intervalId">
          <el-tag type="success">轮询中</el-tag>
        </div>
        <div v-else>
          <el-tag type="warning">未启动轮询</el-tag>
        </div>
        <div style="margin-top: 10px;"></div>
        <el-button type="primary" @click="testConnection" style="width: 100%;">测试连接</el-button>
        <div style="margin-top: 10px;"></div>
        <el-button type="warning" @click="stopPolling" style="width: 100%;">结束轮询</el-button>
        <div style="margin-top: 10px;"></div>
        <el-button type="danger" @click="clearChart" style="width: 100%;">清除图表</el-button>

      </div>
    </div>

    <!-- 右侧可视化框 -->
    <div style="width: 75%; padding: 10px;">
      <div id="chart" style="width: 100%; height: 100%;"></div>
    </div>
  </div>
</template>


<script>
import * as echarts from "echarts";

export default {
  name: "ScatterVisualization",
  data() {
    return {
      form: {
        algorithm: "nsga2",
        funcFile: null,
        popSize: 100,
        generations: 20,
      },
      algorithmOptions: [
        {value: "nsga2", label: "NSGA-II"},
        {value: "nsga2pro", label: "NSGA-II-Pro"}
      ],
      funcFileOptions: [],
      chartInstance: null,
      intervalId: null,
    };
  },
  mounted() {
    // 获取函数文件列表
    this.$request.get("/queryFuncs")
        .then((res) => {
          if (res.status === "success") {
            this.funcFileOptions = res.func_files.map((file) => ({
              label: file,
              value: file,
            }));
          } else {
            this.$message.error(res.message);
          }
        })
        .catch((err) => {
          this.$message.error("获取函数列表失败：" + err.message);
        });
  },
  methods: {
    testConnection() {
      // 测试连接，调用后端接口
      this.$request.get("/hello")
          .then((res) => {
            if (res.status === "success") {
              this.$message.success("连接成功！");
            } else {
              this.$message.error(res.message);
            }
          })
          .catch((err) => {
            this.$message.error("连接失败：" + err.message);
          });
    },
    stopPolling() {
      // 结束轮询
      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
    },
    clearChart() {
      // 清除图表
      if (this.chartInstance) {
        this.chartInstance.dispose();
        this.chartInstance = null;
      }
    },
    startAlgorithm() {
      this.stopPolling(); // 结束轮询
      this.clearChart(); // 清除图表
      if (!this.form.funcFile) {
        this.$message.error("请选择函数文件！");
        return;
      }

      this.$request.post("/start", {
        func_file: this.form.funcFile,
        pop_size: this.form.popSize,
        num_generations: this.form.generations,
      }).then((res) => {
        if (res.status === "success") {
          this.$message.success("算法已启动！");
          this.pollData(); // 启动轮询
        } else {
          this.$message.error(res.message);
        }
      }).catch((err) => {
        this.$message.error("算法启动失败：" + err.message);
      });
    },
    pollData() {
      // 定时轮询获取可视化数据
      this.intervalId = setInterval(() => {
        this.$request.get("/poll")
            .then((res) => {
              if (res.has_new_data) {
                this.updateChart(res.data);
              }
            })
            .catch((err) => {
              this.$message.error("获取数据失败：" + err.message);
            });
      }, 1000); // 每1秒轮询一次
    },
    updateChart(data) {
      if (!this.chartInstance) {
        this.chartInstance = echarts.init(document.getElementById("chart"));
      }

      // 解空间点
      const solutionData = data.solution_points.F1.map((f1, index) => [f1, data.solution_points.F2[index]]);

      // 种群数据
      const populationData = data.population_data.flatMap((rank) =>
          rank.points.map((point) => [point.f1, point.f2])
      );

      const option = {
        title: {
          text: `目标优化第 ${data.generation + 1} 代`,
          left: "center",
        },
        xAxis: {
          name: "f1",
        },
        yAxis: {
          name: "f2",
        },
        series: [
          {
            name: "解空间",
            type: "scatter",
            data: solutionData,
            itemStyle: {color: "lightgray"},
          },
          {
            name: "当前种群",
            type: "scatter",
            data: populationData,
            itemStyle: {color: "#a94826"},
          },
        ],
      };

      this.chartInstance.setOption(option);
    },
  },
  beforeDestroy() {
    // 清除轮询
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  },
};
</script>

<style scoped>
.el-form-item {
  margin-bottom: 20px;
}
</style>
