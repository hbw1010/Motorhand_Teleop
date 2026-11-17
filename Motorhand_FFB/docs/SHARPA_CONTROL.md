# Motorhand → Sharpa 遥操、反馈与接管

Sharpa与Wuji采用并列设备模块，共同C++会话负责主导方切换、逆解、Jacobian和手套输出。Sharpa使用22轴SDK关节力矩，**不采集、不使用触觉**。原有`wuji-*`命令保持兼容。

以下命令均在仓库根目录执行。首次启动自动构建；所有真实输出必须显式添加`--execute`。

## 先看仿真

终端一：假手套开合、Sharpa动力学仿真及名义反馈，不访问真实设备。

```bash
pixi run sharpa-sim --feedback
```

终端二：只读查看器。

```bash
pixi run sharpa-viewer
```

查看器使用本机UDP 8766，显示收到的真实仿真姿态或SDK关节姿态；关闭查看器不停止控制进程。在终端一按`Ctrl+C`停止控制。可以给控制命令添加`--duration 10`定时退出，`--no-object`去掉固定抓握球。

用真实Motorhand控制虚拟Sharpa：

```bash
# 手套仅读取，不回零、不出力
pixi run sharpa-sim-teleop

# 开启手套反馈：会回零并使能Motorhand
pixi run sharpa-sim-teleop --feedback --execute
```

这条路径的MuJoCo步进、接触力投影、重定向及控制均在C++中；查看器仅用收到的qpos显示，不推进控制仿真。

## 真机遥操

当前验证设备为Sharpa右手WaveSE-R-01，固件3.0.10。先只读发现和检查：

```bash
pixi run sharpa-discover
pixi run sharpa-inspect --serial YOUR_SHARPA_SN
```

`inspect`读取22轴位置、速度和SDK报告力矩，不使能或改控制模式。有报警时明确拒绝，并给出错误码及关节编号；不会自动清错。

```bash
# Motorhand只读，Sharpa回零并开始跟随
pixi run sharpa-teleop --serial YOUR_SHARPA_SN --execute

# Motorhand与Sharpa依次回零，开启受阻位置阻抗反馈
pixi run sharpa-force-contact --serial YOUR_SHARPA_SN --execute

# SDK关节力矩的名义负载占比线性反馈
pixi run sharpa-force-linear --serial YOUR_SHARPA_SN --execute
```

默认手侧为`right`；另一侧须显式`--hand left`并使用该设备SN。可用`--expected-pn`和`--approved-firmware`指定已确认型号/版本。默认右手固件值3.0.10只用于核验，不是刷写固件。

Motorhand默认读取用户最近`zero-glove`更新的`hardware/encos_dexhand/configs/sim/glove_mapping.json`。可用`--mapping`指定其他标定。Wuji仍读取其原有FFB标定，两个文件不会静默同步。G4通常自动识别，也可设置`MOTORHAND_GLOVE_DRIVER=2`。

## 记录、回放和接管

任何控制命令都可增加`--output <新文件>`记录本会话。例如：

```bash
pixi run sharpa-teleop --serial YOUR_SHARPA_SN --execute \
  --output recordings/sharpa-follow.jsonl
```

单独只读记录（不接管已经被其他程序使能的设备）：

```bash
pixi run sharpa-record --serial YOUR_SHARPA_SN \
  --output recordings/sharpa-observation.jsonl --duration 10
```

```bash
# 回放控制Sharpa，Motorhand通过共同逆解跟随
pixi run sharpa-replay recordings/sharpa-follow.jsonl --serial YOUR_SHARPA_SN --execute

# 回放中按空格切换到人手接管，再按空格恢复；q或Ctrl+C停止
pixi run sharpa-takeover recordings/sharpa-follow.jsonl --serial YOUR_SHARPA_SN --execute
```

`takeover`启用手套反馈；回放阶段手套位置跟随，接管阶段由手套主导Sharpa并输出反馈。切换保留既有抓握偏移和平滑接入，原始100ms期限不因回放或切换而延长。

100ms约束的是输入时效和本地继续发令许可。独立监察线程在SDK调用阻塞期间也会撤销许可并标记停止待处理；SDK操作保持串行，阻塞调用返回后先失能，再拒绝后续目标。`Device::stop_status()`区分待处理、已确认和SDK忙；排队不代表硬件已失能。供应商SDK没有可中断调用或已确认的并发控制契约，因此网络/SDK不响应时不承诺100ms内物理失能，也不能把软件停止当成硬急停。

纯离线验证同一流程：

```bash
pixi run sharpa-sim --duration 6 --output recordings/sharpa-sim.jsonl
pixi run sharpa-takeover recordings/sharpa-sim.jsonl \
  --simulation --fake-glove --takeover-at 2 --duration 6
```

轨迹schema为2，包含型号、手侧、SN、22个语义关节名和单位。保存的`position_rad`保留实际观测；另存明确投影到合法命令范围的`target_rad`用于回放，以处理软限位微量超调，原始测量不被改写。有速度/负载时一并保存及标明来源。跨型号、错误手侧、错误关节数量/顺序和不合法目标被拒绝；已有文件不会被覆盖。

## 参数和来源

- `config/retargeting/sharpa/model.json`：Sharpa自身的任务定义与优化参数。当前统一位置尺度为本地映射参数，不是硬件标定。四指MCP侧摆、拇指MCP侧摆与小指CMC在手套正向重定向中固定零；真实反馈FK仍保留全部22轴。
- `config/feedback/sharpa-{right,left}.json`：独立Nm偏置、符号、接触阈值、名义负载参考和手套输出限额。URDF effort仅作名义归一化参考，不写入设备，不等同净接触力。
- `config/feedback/sharpa-driver-impedance.json`：手套驱动阻抗与摩擦参数。12轴名义上限均为10 Nm，原生阻抗保留0.85提交预算；MIT PD不是实体硬力矩限幅。
- 真机负载来源为`sharpa-sdk-reported-joint-torque`；仿真负载来源为`sharpa-mujoco-contact-joint-torque`。仿真只投影手与物体接触，不把驱动内力或关节限位反力混进接触源。
- 用户提供资产的`provenance.json`保留其历史来源；本轮实际运行实现为`native_control/hands/sharpa/`的C++代码，不导入其中记录的其他工程路径。

## SDK与构建

完整Sharpa SDK 5.0.11保留在本项目忽略的缓存目录。首次准备：

```bash
pixi run python tools/install_sharpa_sdk.py /absolute/path/sharpa_teleop_wave_resources-5.0.11-py3-none-linux_x86_64.whl
pixi run test-sharpa
```

安装器核对固定版本和归档摘要，保留完整许可，不覆盖不同安装。SDK二进制不提交Git。`build-sharpa-sim`无需Sharpa或Wuji硬件SDK；`build-sharpa`需要Sharpa SDK，不需要Wuji SDK。

## 验证边界

已提供SDK替身、真实模型几何、MuJoCo碰撞、缓存时间、期限、故障、回放接管及只读查看器回归。假手套只产生输入和记录名义输出，不模拟人体受力，因此离线测试不证明穿戴稳定性。真实设备目前只做身份/状态预检；现场反馈参数及运动效果须由使用者运行上述显式输出命令确认。
