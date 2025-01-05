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
          <el-row>
            <el-col :span="form.algorithm === 'nsga2'? 24 : 12">
              <el-form-item label="突变概率">
                <el-input-number v-model="form.mutation_rate" :min="0" :max="1" :step="0.01" style="width: 100%"/>
              </el-form-item>
              <el-form-item label="交叉概率">
                <el-input-number v-model="form.crossover_rate" :min="0" :max="1" :step="0.01" style="width: 100%"/>
              </el-form-item>
              <el-form-item label="解精度">
                <el-input-number v-model="form.precision" :min="0" :max="1" :step="0.01" style="width: 100%"/>
              </el-form-item>

              <el-form-item label="种群大小(2倍)">
                <el-input-number v-model="form.popSize" :min="10" :max="500" style="width: 100%"/>
              </el-form-item>

              <el-form-item label="代数">
                <el-input-number v-model="form.generations" :min="1" :max="200" style="width: 100%"/>
              </el-form-item>
            </el-col>
            <el-col :span="12" v-if="form.algorithm === 'nsga2pro'">
              <el-form-item label="使用差分变异">
                <el-switch v-model="form.use_diff_mutation"/>
              </el-form-item>
              <el-form-item label="使用预测模型">
                <el-switch v-model="form.use_predict_model"/>
              </el-form-item>
            </el-col>

          </el-row>


          <div class="button-container">
            <el-button class="primary-button" type="primary" @click="startAlgorithm" plain>启动算法</el-button>
            <el-button class="normal-button" type="danger" @click="stopAlgorithm" plain>停止算法&清除缓存
            </el-button>
          </div>
          <div style="margin-top: 10px;"></div>
          <el-form-item label="分辨率">
            <el-input-number v-model="form.resolution" :min="10" :max="500" style="width: 100%"/>
          </el-form-item>
          <el-form-item label="轮询间隔">
            <el-input-number v-model="form.poolingTime" :min="100" :max="2000" style="width: 100%"/>
          </el-form-item>
        </el-form>
      </div>

      <!-- 左下部分 -->
      <div style="padding-top: 20px;">
        <!-- 轮询状态显示-->
        <div v-if="intervalId">
          <el-tag type="success" size="medium">轮询中</el-tag>
        </div>
        <div v-else>
          <el-tag type="warning" size="medium">未启动轮询</el-tag>
        </div>
        <div style="margin-top: 10px;"></div>
        <div class="button-container">
          <el-button class="primary-button" type="warning" @click="stopPolling" plain>结束轮询</el-button>
          <el-button class="normal-button" type="primary" @click="testConnection" plain>测试连接</el-button>
        </div>
        <div style="margin-top: 10px;"></div>
        <el-button class="normal-button" type="danger" @click="clearChart" plain>清除图表</el-button>
        <div style="margin-top: 10px;"></div>
        <div class="button-container">
          <div style="color: #72767b"> gif渲染时间</div>
          <el-input-number v-model="gifChangeTime" :min="100" :max="2000" style="width: 50%"/>
          <el-button class="normal-button" type="primary" @click="downLoadGifImg" plain style="margin-left: 10px;">下载Gif</el-button>
        </div>

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
        funcFile: "标准动态问题1",
        popSize: 100,
        generations: 20,
        resolution: 200,
        mutation_rate: 0.01,
        crossover_rate: 0.9,
        precision: 0.01,
        poolingTime: 800,
        use_diff_mutation: false,
        use_predict_model: false,
      },
      algorithmOptions: [
        {value: "nsga2", label: "NSGA-II"},
        {value: "nsga2pro", label: "NSGA-II-Pro"}
      ],
      funcFileOptions: [],
      chartInstance: null,
      intervalId: null,
      frames: [],// 存放每一代的可视化数据
      maxRankCount: -1,//用于防止残留高rank点
      gifChangeTime: 200, // gif 切换时间
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
    downLoadGifImg() {
      if (!this.chartInstance) {
        this.$message.error("不存在图表");
      }
      if (this.intervalId) {
        this.$message.error("请结束轮询,再下载Gif");
      }
      const gifshot = require('gifshot'); // 确保 gifshot 已安装并引入
      const frames = [];
      const timelineOptions = this.chartInstance.getOption().timeline;
      const timelineLength = timelineOptions[0]?.data.length || 0;
      console.log("时间轴帧数量:" + timelineLength);

      if (timelineLength === 0) {
        this.$message.error("时间轴数据为空，无法导出");
        return;
      }

      let currentFrame = 0;
      let gifWidth = 0;
      let gifHeight = 0;

      const captureFrame = () => {
        // 切换到时间轴的当前帧
        this.chartInstance.dispatchAction({
          type: 'timelineChange',
          currentIndex: currentFrame,
        });

        // 等待图表渲染完成
        setTimeout(() => {
          const dataURL = this.chartInstance.getDataURL({
            type: 'png',
            pixelRatio: 1, // 调整为需要的清晰度
            backgroundColor: '#fff', // 设置背景色
          });

          // 创建一个 Image 对象获取宽度和高度
          const img = new Image();
          img.onload = () => {
            if (gifWidth === 0 || gifHeight === 0) {
              gifWidth = img.width;
              gifHeight = img.height;
            }
            frames.push(dataURL); // 保存当前帧图像

            currentFrame++;
            if (currentFrame < timelineLength) {
              captureFrame(); // 捕获下一帧
            } else {
              // 全部帧捕获完毕，生成 GIF
              this.$message.success("GIF 正在准备中，请稍候...");
              gifshot.createGIF(
                  {
                    images: frames,
                    gifWidth: gifWidth, // 设置 GIF 宽度
                    gifHeight: gifHeight, // 设置 GIF 高度
                    frameDuration: 1, //一帧
                  },
                  function (obj) {
                    if (!obj.error) {
                      const gifData = obj.image;
                      const a = document.createElement('a');
                      a.href = gifData;
                      a.download = 'chart-animation.gif';
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                      this.$message.success("GIF 下载成功");
                    } else {
                      this.$message.error("GIF 生成失败：" + obj.errorMsg);
                    }
                  }.bind(this)
              );
            }
          };
          img.src = dataURL; // 设置 Image 的 src 为当前帧的 DataURL
        }, this.gifChangeTime); // 确保图表完成渲染，必要时增加此值
      };


      // 开始捕获帧
      captureFrame();

    },
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
    stopAlgorithm() {
      this.stopPolling(); // 结束轮询
      this.$request.post('/stop')
          .then((res) => {
            if (res.status === 'success') {
              this.$message.success(res.message);
            } else {
              this.$message.error(res.message);
            }
          })
          .catch((err) => {
            this.$message.error('停止算法失败：' + err.message);
          });
    },
    startAlgorithm() {
      this.stopAlgorithm(); // 停止算法并清除缓存，结束轮询
      this.clearChart(); // 清除图表
      this.frames = []; // 清空帧数据
      if (!this.form.funcFile) {
        this.$message.error("请选择函数文件！");
        return;
      }

      this.$request.post("/start", {
        algorithm: this.form.algorithm,
        func_file: this.form.funcFile,
        pop_size: this.form.popSize,
        num_generations: this.form.generations,
        resolution: this.form.resolution,
        mutation_rate: this.form.mutation_rate,
        crossover_rate: this.form.crossover_rate,
        precision: this.form.precision,
        use_crossover_and_differential_mutation: this.form.use_diff_mutation,
        use_prediction: this.form.use_predict_model,
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
              // console.log(res);
              if (res.has_new_data) {
                console.log("-------------------------------generation:" + res.data.generation);
                this.updateChart(res.data);
                if (res.data.generation + 1 >= this.form.generations) {
                  this.stopPolling(); // 结束轮询
                }
              }
            })
            .catch((err) => {
              this.$message.error("获取数据失败：" + err.message);
            });
      }, this.form.poolingTime);
    },
    updateChart(data) {
      if (!this.chartInstance) {
        this.chartInstance = echarts.init(document.getElementById("chart"));
      } else {
        console.log("清空echarts");
        this.chartInstance.clear(); // 清除旧内容
      }
      // console.log("data：" + JSON.stringify(data));

      const solutionData = data.solution_points;
      const populationData = data.population_data;
      const maxRanks = populationData.length;
      // console.log("solutionData:" + JSON.stringify(solutionData));
      // console.log("populationData:" + JSON.stringify(populationData));
      // let populationSize = 0;
      // for (let i = 0; i < populationData.length; i++) {
      //   populationSize += populationData[i].points.length;
      // }
      // console.log("populationSize: " + populationSize)

      // const allData = [
      //   ...solutionData,
      //   ...populationData.flatMap((rank) => rank.points),
      // ];
      // console.log("数据量:" + allData.length);
      // console.log("allData:"+ JSON.stringify(allData))
      console.log("maxRanks:" + maxRanks);

      const rankSeries = [];
      this.maxRankCount = maxRanks > this.maxRankCount ? maxRanks : this.maxRankCount;
      console.log("maxRankCount:" + this.maxRankCount);

      for (let i = 0; i < this.maxRankCount; i++) {
        rankSeries.push(
            populationData[i]
                ? {
                  name: `Rank ${i}`,
                  type: "scatter",
                  data: populationData[i].points.map((point) => [point.f1, point.f2]),
                  itemStyle: {
                    color: `hsl(${
                        120 + ((0 - 120) * i) / (maxRanks - 1)
                    }, 90%, 40%)`,
                  },
                }
                : {
                  name: ``,
                  type: "scatter",
                  data: [],//清除高rank的残留点
                  itemStyle: {color: "transparent"},
                }
            //给高rank点填充空的数据,避免残留.
            // timeline组件在切换时会合并新旧数据,导致旧的高rank点残留,timeline组件切换不走这个方法.目前找不到方法可以设置.
            //或许可以利用 timeLine 组件的相关回调.
        );
      }

      this.frames[data.generation] = {
        title: `多目标优化第 ${data.generation + 1} 代`,
        solutionData,
        rankSeries,
      };

      const timelineData = this.frames.map((_, index) => `第${index + 1}代`);
      const timelineOptions = this.frames.map((frame, index) => {
        let maxX = -Infinity, minX = Infinity, maxY = -Infinity, minY = Infinity;

        // 动态计算当前帧的数据范围
        const allData = [
          ...frame.solutionData,
          ...frame.rankSeries.flatMap((rank) => rank.data),
        ];
        for (const p of allData) {
          const x = p[0], y = p[1];
          if (x != null) {
            if (x > maxX) maxX = x;
            if (x < minX) minX = x;
          }
          if (y != null) {
            if (y > maxY) maxY = y;
            if (y < minY) minY = y;
          }
        }

        // 更新当前帧的 series
        const series = [
          {
            name: "解空间",
            type: "scatter",
            data: frame.solutionData,
            symbolSize: 11,
            itemStyle: {color: "lightgray"},
            large: true,
          },
          ...frame.rankSeries,
        ];

        return {
          title: {text: frame.title},
          xAxis: {name: "F1", type: "value", min: minX, max: maxX},
          yAxis: {name: "F2", type: "value", min: minY, max: maxY},
          legend: {
            data: ["解空间", ...frame.rankSeries.map((rank) => rank.name)],
            orient: "vertical",
            right: 11,
          },
          series,
        };
      });

      const option = {
        baseOption: {
          timeline: {
            axisType: "category",
            data: timelineData,
            autoPlay: false,
            playInterval: 500,
            tooltip: {
              formatter: (p) => `${p.name}`,
            },
            // currentIndex: this.frames.length - 1, // 设置为最新的帧索引
          },
          title: {left: "center"},
          xAxis: {name: "F1", type: "value"}, // 初始设置，具体值在 options 中动态更新
          yAxis: {name: "F2", type: "value"},
          tooltip: {
            trigger: "item",
            axisPointer: {
              type: "cross",
            },
            formatter: (params) =>
                `<b>F1:</b> ${params.value[0]}<br><b>F2:</b> ${params.value[1]}`,
          },
          // legend: {orient: "vertical", right: 11},
          animation: false,
          toolbox: {
            show: true,
            orient: "vertical",
            right: "20%",
            top: "top",
            feature: {
              saveAsImage: {show: true},  // 保存图表
            },
          },
        },
        options: timelineOptions,
      };

      this.chartInstance.setOption(option);
      const latestIndex = this.frames.length - 1;
      this.chartInstance.setOption({
        baseOption: {
          timeline: {
            currentIndex: latestIndex, // 设置为最新的帧索引
          },
        },
      });
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

.button-container {
  display: flex;
  justify-content: space-between; /* 让两个按钮均匀分布在两端，实现左右排列 */
}

.normal-button {
  background-color: #ffffff;
  color: #606266;
  border-radius: 0.5rem;
  font-weight: bold;
  font-size: 1rem;
  width: 100%;
  padding: 0.2rem 0.5rem; /* 按钮的内边距，可以根据需要进行调整 */;
  height: 2.8rem;
}

.primary-button {
  background-color: #0d539f;
  color: #ffffff;
  border-radius: 0.5rem;
  font-weight: bold;
  font-size: 1rem;
  width: 100%;
  padding: 0.2rem 0.5rem; /* 按钮的内边距，可以根据需要进行调整 */;
  height: 2.8rem
}

</style>
