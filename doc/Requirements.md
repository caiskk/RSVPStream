| **类别** | **具体内容** |
| --- | --- |
| **脑电算法实时部署系统** |  |
| - 硬件平台 | FPGA + ARM (暂定3588+FPGA) |
| - 主要功能 | 脑电信号预处理、脑电模型推理、异构设计、模块化设计、数据流优化、卷积加速、可扩展性 |
| - 信道筛选 | 从64*1000维度的脑电数据筛选出54通道信号 |
| - 降采样率 | 4倍降采样，输出维度54*250 |
| - 推理架构 | XGBoosting架构 |
| - 推理准确率 | FPGA部署后推理准确率下降≤5% |
| - 运行时间 | 单次运行时间≤10ms |
| - 硬件重量 | ≤200g |
| - 通讯协议 | 支持网口和串口 |
| - 通讯延时 | ≤1ms |
| **海陆分割算法** |  |
| - 图像融合 | 多通道融合，低分辨率多光谱图像与高分辨率全色图像融合，支持0.3~5.0m分辨率，融合误差≤1% |
| - 地理信息匹配 | 匹配准确率≥95% |
| - 海陆分割 | 分辨率0.3~5.0m，海岸线识别准确率≥90%，基准距离≤100pixels，分割时间≤20s |
| **视图预处理程序** |  |
| - 视频预处理 | 读取外部视频文件/流，解码H.264、H.265，抽帧，缓存管理，并行处理能力≥2路 |
| - 解码速率 | ≥60fps (1080P) |
| - 图像预处理 | 读取、解码、分割大幅图像，图像分割后为640*640像素，单张图像处理延迟≤10ms，最大支持30000*30000像素图像 |
| **技术交付内容** |  |
| - 软件交付 | 脑电算法实时部署系统源码、海陆分割算法Python源码、视图预处理程序模块源码、运行虚拟环境包 |
| - 文档交付 | 设计说明、使用手册、性能测试报告、算法移植部署操作手册 |
| **交付周期** | 合同签订后3个月 |

代码：https://github.com/bowenliee/XGB-DIM-for-RSVP.git

https://github.com/bowenliee/XGB-DIM-for-RSVP.git

# XGB-DIM-for-RSVP

An ensemble learning method for EEG classification in RSVP tasks<br/>
Require:<br/>
**`torch**(necessary)` | **`numpy**(necessary)` | sklearn.metrics.roc_curve | sklearn.metrics.auc | h5py  <br/>
The EEG device is NeuroScan. If EEG from other device is used, the channel list should be adjusted to the form of Neuroscan.

## XGB_DIM

A CPU version, i.e., the version in paper 'Assembling global and local spatial-temporal filters to extract discriminant information of EEG in RSVP task'. <br/>
Journal of Neural Engineering, https://iopscience.iop.org/article/10.1088/1741-2552/acb96f <br/>
In this version, you can find the whole details of parameter optimization, including the extreme gradient boosting, gradient calculation and the specific implementation of Adam. You can compare them with the derivation in the paper.

## XGB_DIM_GPU_v2

A GPU version based on Torch lib. `90% time cost reduced` <br/>
Almost all the 'for' cycles have been replaced by tensor calculation, which greatly improves the speed. The models and their losses are clearly defined in each class. Because the optimizer of torch is used, only the details of extreme gradient boosting are retained. <br/>
The code in this version is more concise, and it is easy to adjust the internal structure or use it to generate improved versions.

## multi_XGB_DIM_GPU_v1

This version is based on XGB_DIM_GPU_v2 and makes full use of negative samples. Its performance is better than the ordinary GPU version. If the random sampling mode is used, the stability is also improved.

##