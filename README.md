# 手写 BP 神经网络分类实验

使用 NumPy/CuPy 基础数学运算手写前向传播、Softmax 交叉熵、反向传播和 SGD。默认使用 CUDA GPU；没有 PyTorch、TensorFlow、自动求导或现成神经网络分类器。

**交付状态：代码已编写，作者未执行代码、单元测试或训练。下列检查与实验由你在已配置的 Conda 环境中运行；真实指标、图表和实验报告会在运行后生成。**

## 一、目录与阅读顺序

```text
main.py                       命令入口
configs/baseline.json          基线超参数，可复制修改
src/backend.py                NumPy / CuPy 后端
src/data.py                   数据审计、固定分层划分、标准化、子集采样
src/network.py                手写 BP 核心、模型保存/加载（重点阅读）
src/metrics.py                准确率、宏 F1、每类指标、混淆矩阵
src/training.py               逐轮打乱、分批训练、记录、检查点
src/experiments.py            单因素实验与多种子调度
src/reporting.py              汇总、比较图、中文实验报告
src/finalization.py           完整建模池重训、最终测试、加载模型预测
src/validation.py             有限差分与 CPU/GPU 数值检查
tests/test_bp.py              用户手动运行的 CPU 单元测试
reports/BP算法说明.md           公式、张量形状与代码解释
任务计划.md                   原始审阅计划（保留）
environment.yml               你已导出的环境文件（保留）
environment-win64-lock.txt     你已导出的 Windows 依赖锁定文件（保留）
```

建议先看 [算法说明](reports/BP算法说明.md)，再阅读 `network.py`、`training.py`、`data.py`。主要公式和易出错的位置均有中文注释。

## 二、进入环境

在 **Anaconda Prompt** 中：

```bat
conda activate bp_manual_gpu
cd /d C:\Users\Asus\Desktop\moshi_experiment
```

所有后续命令从项目根目录执行。查看命令帮助：

```bat
python main.py --help
python main.py experiments --help
```

已有环境无需重新安装。依赖是 NumPy、CuPy、Matplotlib 和标准库；CPU 单元测试无需可用 GPU。

## 三、先检查数据和算法

```bat
python main.py inspect --output results/data_check
python -m unittest discover -s tests -v
python main.py check --device gpu --output results/gpu_check.json
```

预期：

1. 数据检查显示：原训练/测试各 75 条；重复过滤后的建模池 74 条；对照训练 59 条；验证 15 条，每类 5 条。
2. 单元测试核对有限差分、多层梯度、损失稳定性、构造数据学习、标准化、划分、模型保存/加载等，成功时显示 `OK`。
3. GPU 检查会核对矩阵乘法、指数、三种激活下的前向/损失/梯度/单步 SGD；成功时显示 `CHECK PASSED`。GPU 无效会明确报错，不自动降级成 CPU。

若只想先检查 CPU 后端：

```bat
python main.py check --device cpu
```

这里的 check 是数值正确性检查，不运行项目的正式训练。单元测试中有一个小型构造数据学习测试，用于检查训练链路。

## 四、先运行单次基线

```bat
python main.py baseline --device gpu --output results/baseline --log-every 100
```

默认：`4→8→3`、Tanh、Xavier、Softmax + 交叉熵、SGD、学习率 0.05、批大小 16、2000 轮、种子 42。配置来自 `configs/baseline.json`。

输出包括：

- `data_audit.json`、`data_split.json`：数据指纹、过滤记录、原始行号、类别数。
- `runs/baseline/seed_42/history.csv`：包含第 0 轮在内的逐轮损失、准确率、宏 F1、验证各类召回率和更新次数。
- `runs/baseline/seed_42/epoch_2000.npz`：权重、偏置、标准化参数、标签映射与配置。
- `learning_curves.png`：训练/验证曲线。
- `runs.csv`、`summary.csv`、`reports/实验报告.md`：单次基线记录。只有一个种子时标准差记为 0，不能说明模型没有波动。

快速检查命令能否串通时，可以另建输出目录减少训练轮数；这不替代正式实验：

```bat
python main.py baseline --device gpu --epochs 30 --output results/baseline_quick --log-every 10
```

## 五、运行完整对照实验

```bat
python main.py experiments --device gpu --output results/full_experiment
```

默认 5 个种子：`7 21 42 87 123`；每个种子共享固定训练/验证划分。默认共 80 次实际训练，另有 10 条来自基线检查点的轮数记录；准确数量以实际配置和 `suite.json` 为准。小数据加上大量 GPU 小算子调用、逐轮评估可能较慢，不保证比 CPU 快。

| 实验组参数 | 对比内容 |
| --- | --- |
| `topology` | `4→4→3`、`4→8→3`、`4→16→3`、`4→8→8→3` |
| `learning_rate` | 0.01、0.05、0.1 |
| `activation` | Sigmoid、Tanh、ReLU；固定初始化策略 |
| `batch_size` | 1、16、全批量（配置中写 0） |
| `standardize` | 不标准化、Z-score |
| `epochs` | 200、1000、2000；复用基线同一训练轨迹的检查点 |
| `data_size` | 分层嵌套采样约 25%、50%、75%、100%；默认实际 15、30、44、59 条 |
| `class_balance` | 默认 `[10,10,10]` 和 `[15,10,5]`，均为 30 条 |

基线只训练一次并复用到各组对比。单因素实验固定其他基线参数，没有全排列搜索，也没有早停。缩短 `--epochs` 时，不会生成超出总轮数的轮数检查点。

可以先只跑某几组或减少种子，但应为每次运行使用新目录：

```bat
python main.py experiments --device gpu --groups topology learning_rate --seeds 7 21 42 --output results/topology_lr
```

修改超参数时，复制配置文件，保持输入层 4、输出层 3。例：

```bat
python main.py baseline --config configs/my_config.json --device gpu --output results/my_baseline
```

额外支持 `l2` 和 `initialization="he"`，默认主实验不改变这些值。主线交叉熵分类已经满足 BP 任务；原计划中的 Sigmoid 输出 + 均方误差、标签噪声等为可选扩展，本版未加入。

## 六、查看和重新生成报告

完整实验结束会自动生成：

```text
results/full_experiment/
  suite.json                   环境、种子、完成状态、选择规则
  data_audit.json               文件指纹与重复记录
  data_split.json               固定划分及原始行号
  runs.json / runs.csv          每个条件和种子的结果
  summary.json / summary.csv    均值、样本标准差、配置
  runs/<条件>/seed_<种子>/      逐轮曲线数据、模型检查点、训练样本行号
  reports/实验报告.md            按真实数据填写的中文报告
  reports/figures/*.png         各组验证指标及收敛曲线
```

只重新生成报告、不训练：

```bat
python main.py report --suite results/full_experiment
```

正式报告含方法、环境、对照表、均值/标准差、各类召回率、曲线和验证集选择的配置。报告会指出观察到的领先条件，并提供分析提示；解释因果时应结合训练/验证曲线，不能把微小差异直接写成普遍规律。

所有对照实验阶段都不计算项目测试集分类分数。数据量和类别比例组仅分析数据影响，不参与最终超参数选择。

## 七、确定配置后，执行最终重训和测试

建议先阅读完整对照报告，再执行一次：

```bat
python main.py finalize --suite results/full_experiment --device gpu --seed 42 --output results/final
```

选择规则固定为：完整训练池配置的多种子验证宏 F1 均值最高，依次用准确率、交叉熵、参数量、条件名打破并列。不依据测试集选择配置或种子。

程序先写入选择配置和种子，再在 74 条建模数据上重新拟合标准化器并重训，按所选轮数训练，最后对 75 条测试数据执行一次评估。全批量配置会随训练数据大小自动使用 74 条。

最终目录包含：

- `selection.json`：验证选择依据、最终配置和重训种子。
- `model.npz`：完整可加载模型，包含标准化参数与标签映射。
- `history.csv`、`learning_curves.png`：最终重训记录（无验证集，因为已合并完整建模池）。
- `test_metrics.json`：准确率、宏 F1、交叉熵、各类精确率/召回率/F1、混淆矩阵、错误数。
- `test_predictions.csv`、`misclassified.csv`：逐条预测及误分类；零错误时错误文件只有表头。
- `confusion_matrix.png`、`最终评估报告.md`。

对照实验和最终评估会核对原始文件 SHA256，避免数据变动后误用旧配置。项目移动后可给 finalize 加 `--data-dir 新的数据目录`。

如果只需要重新生成最终报告和图表，无需重新训练或重新计算测试分数：

```bat
python main.py report --final-dir results/final
```

## 八、加载模型进行识别分类

单个样本：

```bat
python main.py predict --model results/final/model.npz --features 5.1 3.5 1.4 0.2 --device gpu
```

批量输入：创建空白分隔的四列 TXT，**不含标签列、表头**，例如：

```text
5.1 3.5 1.4 0.2
6.0 2.9 4.5 1.5
6.7 3.1 5.6 2.4
```

保存为 `samples.txt` 后：

```bat
python main.py predict --model results/final/model.npz --input samples.txt --output results/predictions.csv --device gpu
```

输出原始类别标签 1/2/3 及各类概率。模型也可以在 CPU 加载，使用 `--device cpu` 即可。

## 九、运行约定与问题排查

- **输出目录已存在**：换一个新的 `--output` 名称。训练不会覆盖旧结果；报告命令允许重新生成同一目录中的汇总和图表。
- **中途失败**：`suite.json` 记录 `failed` 和错误，已完成记录保留。修复问题后用新目录重跑；本版不支持断点续训，也不使用不完整实验选择最终配置。
- **CuPy 导入或 CUDA 错误**：确认 `conda activate bp_manual_gpu`，先单独运行 `python main.py check --device gpu`；不要让失败的 GPU 运行和 CPU 运行混在同一目录。
- **训练出现 NaN/Inf**：程序会中止；检查自定义配置的学习率、输入尺度和数据。默认已启用标准化。
- **图表生成失败但训练已完成**：训练文件仍保留；修复 Matplotlib 后执行 report 命令，不必重训。
- **耗时解释**：记录同步后的累计训练及逐轮评估时间，包含此前检查点写盘开销；不是纯内核基准测试。GPU 的首次调用/编译开销和操作系统负载会影响计时。
- **正式实验解释**：同一固定验证集下的多种子统计反映训练随机性，不能当作对所有新数据的置信区间。最终测试每多错一条，准确率约变化 1.33 个百分点。

原始数据、既有环境文件和审阅计划不由训练程序修改。程序不会自动安装依赖或修改系统环境。
