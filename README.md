# DCQ-DETR 光学–SAR 旋转框检测

| 文件 | 职责 |
|---|---|
| `functions.py` | 旋转框几何、Hungarian 匹配、检测与去噪损失、旋转框 AP 和候选召回率 |
| `data.py` | 光学和 SAR 配对读取、标注转换、同步几何增强、局部退化与掩码 |
| `model.py` | 双 ResNet-50、独立多尺度编码、候选 Query、旋转可变形注意力和查询级补偿 |
| `train.py` | 训练和评估入口、日志、模型保存与断点续训 |

## 用本机 OGSOD-2.0 训练

在当前项目目录运行以下 PowerShell 命令。本机已有的虚拟环境含 CUDA 版 PyTorch 和 torchvision。

~~~powershell
$python = 'E:\python项目\.venv\Scripts\python.exe'
$dataset = 'E:\python项目\OGSOD-2.0'

& $python train.py --data-root $dataset --dataset ogsod2 --device cuda --output runs/ogsod2_full
~~~

默认以 256×256 图块、300 个 Query 和 6 层解码器训练 36 个 epoch，后 3 层执行双向条件再采样。CUDA 默认启用混合精度，CPU 可用 `--device cpu`。从 ResNet-50 ImageNet 权重初始化可用 `--imagenet`，由 torchvision 读取缓存或下载权重。

将 `--epochs 1 --train-limit 8 --eval-limit 6` 加到命令末尾，可以先用分布在整个划分中的少量样本检查运行链路。每次新实验使用独立的输出目录。

## GPU 吞吐

在本机 RTX 5060 Ti 上训练 256×256 的 OGSOD-2.0 时，可以使用以下配置。

~~~powershell
& $python train.py --data-root $dataset --dataset ogsod2 --device cuda --batch-size 8 --workers 4 --output runs/ogsod2_b8
~~~

`metrics.jsonl` 中的 `train_imgs_per_sec` 记录每秒处理的配对图块数，只计训练区间。短调试可以用 `--workers 0`，省去 Windows 多进程启动时间。调整批量会改变每轮的优化器更新次数，消融实验应使用相同批量。

## 单卡 RTX 5090 服务器

在服务器上的项目目录中创建独立环境。下面的 CUDA 12.8 wheel 支持 Blackwell，示例选用已在本机验证过的 PyTorch 与 torchvision 版本组合。OpenCV 依赖使用 headless 包，适合服务器。 [PyTorch CUDA 版本](https://pytorch.org/get-started/previous-versions/) · [RTX 5090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.ones(1, device='cuda').item())"
~~~

~~~bash
python train.py --preset rtx5090 --device cuda:0 --dataset ogsod2 --data-root /data/OGSOD-2.0 --output runs/ogsod2_5090

python train.py --preset rtx5090 --device cuda:0 --dataset m4sar --data-root /data/M4-SAR --output runs/m4sar_5090
~~~

启动时空闲显存至少 26 GiB 的情况下，256×256 图块从 batch 32 起步，512×512 图块从 batch 8 起步。预设根据图块面积和启动时的空闲显存下调批量。支持 BF16 的 CUDA 卡优先用 BF16，旋转采样保留 float32 坐标。训练数据进程跨 epoch 复用，默认按可用 CPU 数分配至多 8 个进程，每个进程预取 2 个 batch。

用 `--batch-size`、`--workers`、`--prefetch-factor`、`--no-persistent-workers` 或 `--amp-dtype fp16` 覆盖预设。Docker 中使用多进程预取时，可以为容器配置 `--shm-size=4g`。根据 `metrics.jsonl` 的 `train_peak_gpu_gib` 和 `train_imgs_per_sec` 调整批量。

训练检查点记录实际运行配置。只指定 `--resume` 续训时，同一设备类型会沿用已保存的批量、加载和精度设置。显式传入 `--preset rtx5090` 可按当前设备重新选择，或直接用命令行覆盖单项。做消融时，各组固定批量和混合精度设置。

## 读写权重和评估

训练目录保存 `last.pt`、`best.pt` 和 `metrics.jsonl`。`last.pt` 保存模型、AdamW、调度器和随机状态。`best.pt` 按原始输入的验证 AP 选择，用于评估。

~~~powershell
& $python train.py --data-root $dataset --resume runs/ogsod2_full/last.pt --epochs 48

& $python train.py --data-root $dataset --resume runs/ogsod2_full/best.pt --eval-only --eval-split test --eval-conditions clean optical sar
~~~

`--epochs` 是累计训练轮数。验证使用代码内的旋转 IoU 和 101 点插值 AP，输出 AP、AP50、AP75、各类 AP50 与初始框 Recall@K。指标取值为 0 到 1。`optical` 和 `sar` 分别在同一批样本上加入人工局部遮挡和人工局部降分辨率，随机掩码按样本索引固定。目标旋转框内的平均退化强度达到 0.5 时，计入受损目标的候选召回和检测召回。

## 数据和消融设置

**OGSOD-2.0** 从 `rgb/<split>/images`、`sar/<split>/images` 读取同名图块，使用 `rgb/<split>/labelTxt` 的 DOTA 四点标注计算最小外接旋转框。默认类别来自本机标注，依次为 `bridge`、`harbor`、`storage_tank`。评估包含标注中的所有有效框。SAR 默认保留 RGB 编码的三个通道，灰度文件复制成三通道。通道数一致的其他版本可通过 `--sar-channels` 指定输入层。

**M4-SAR** 用 `--dataset m4sar --data-root <解压目录>` 切换到 512×512 图块，按发布目录读取 `optical/images/<split>`、`sar/images/<split>`、`optical/labels/<split>`。标签为 YOLO-OBB 的类别 ID 和 8 个归一化顶点坐标，默认类别顺序来自作者配置。

| 参数 | 实验设置 |
|---|---|
| `--proposal quota` | 光学和 SAR 分别保留 Top-150 候选 |
| `--proposal global` | 两路候选合并后选全局 Top-300 |
| `--sampling base` | 单次读取每一路特征并等权聚合 |
| `--sampling gate` | 保持双向再采样，缺陷分数进入融合门控 |
| `--sampling weights` | 缺陷分数进入融合门控和注意力权重 |
| `--sampling full` | 缺陷分数进入融合门控、注意力权重和采样偏移 |

候选总数由 `--num-queries` 设置。`gate`、`weights`、`full` 使用相同参数量和注意力调用次数。比较时使用相同的种子、训练退化概率和评估条件。原始特征通过 PyTorch `grid_sample` 做稀疏采样，匹配借鉴 RHINO 的 Hausdorff 和 Gaussian KLD 代价，训练使用独立实现的对比去噪与逐层参考框更新。

[OGSOD-2.0](https://github.com/wchao0601/GaLD#optical-guided-sar-object-detection-dataset) · [M4-SAR 作者数据配置](https://github.com/wchao0601/M4-SAR/blob/master/ultralytics/cfg/datasets/M4-SAR.yaml) · [RHINO](https://github.com/SIAnalytics/RHINO)
