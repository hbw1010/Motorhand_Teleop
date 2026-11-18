# Motorhand → Sharpa 遥操、反馈与接管

Sharpa与Wuji采用并列设备模块，共同C++会话负责主导方切换、逆解、Jacobian和手套输出。Sharpa使用22轴SDK关节力矩，**不采集、不使用触觉**。原有`wuji-*`命令保持兼容。

以下命令均在仓库根目录执行。首次启动自动构建；所有真实输出必须显式添加`--execute`。

## 先看仿真

一条指令同时启动假手套、Sharpa动力学仿真和并排查看窗口，不访问真实设备。默认去掉抓握球，便于观察跟随效果。

```bash
pixi run sharpa-sim
```

窗口左侧Motorhand、右侧Sharpa，无需另开查看器。Motorhand使用控制端68轴闭链姿态，Sharpa显示真实积分后的仿真姿态；显示副本保持原尺寸，只平移并排，不参与控制。关闭本次窗口或按`Ctrl+C`会停止本次启动的仿真。可添加`--duration 10`定时退出，`--headless`只运行原生控制；单独运行的`sharpa-viewer`仍是只读独立工具。需要接触实验时用`--with-object`恢复固定抓握球。

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

上述三个真机遥操入口默认一起打开并排窗口。左侧显示 Motorhand 闭链姿态，右侧显示 Sharpa 实测关节姿态。关闭本次窗口或按 `Ctrl+C` 会请求本次控制正常停止。添加 `--headless` 可从无桌面终端只启动控制。

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

`takeover`启用手套反馈。回放期间 Motorhand 的位置跟随使用 Sharpa 限速后提交给 SDK 的目标，保留物体受阻时的握紧余量。接管首帧保持该目标。手套相对最近一次求解的电机角未移动超过 0.002 rad 时保持接管目标，避免求解器继续收敛消耗抓握偏移。主动开合后，沿用按运动衰减的偏移，并约束偏移收敛不反向拉动手指。恢复回放时从最近的提交姿态接入，轨迹等到新鲜逆解后继续。原始 100 ms 期限仍按输入时间计算。

`sharpa-takeover`默认打开 Motorhand／Sharpa 并排窗口。空格在控制终端按，添加 `--headless` 可只启动控制进程。普通 `sharpa-force-contact` 运行手套主导的正解和力反馈。`sharpa-takeover` 使用既有的回放逆解与接管正解切换。两者默认使用与 `wuji-force-contact` 相同的 Motorhand 位置阻抗配置和 `0.3` 硬度。

100ms约束的是输入时效和本地继续发令许可。独立监察线程在SDK调用阻塞期间也会撤销许可并标记停止待处理；SDK操作保持串行，阻塞调用返回后先失能，再拒绝后续目标。`Device::stop_status()`区分待处理、已确认和SDK忙；排队不代表硬件已失能。供应商SDK没有可中断调用或已确认的并发控制契约，因此网络/SDK不响应时不承诺100ms内物理失能，也不能把软件停止当成硬急停。

使能调用本身必须在原始期限内返回。随后等待设备确认使能时，仅用新鲜关节原始包维持有限的确认阶段，重复包不刷新时效。这一阶段不下发位置目标，也不重发使能；确认后才进入回零。

若 SDK 给出相同时间戳和序号、内容却有变化的缓存快照，控制端丢弃它，并按上一帧的原始时间判断剩余有效期。一直收不到完整新包时，按原期限停止。

纯离线验证同一流程：

```bash
pixi run sharpa-sim --duration 6 --output recordings/sharpa-sim.jsonl
pixi run sharpa-takeover recordings/sharpa-sim.jsonl \
  --simulation --fake-glove --takeover-at 2 --duration 6
```

轨迹 schema 为 2，包含型号、手侧、SN、22 个语义关节名和单位。`position_rad`保存实际观测。控制会话的`target_rad`保存限速后实际提交的目标，独立只读录制则将实测角投影到合法命令范围作为回放目标。文件头的`target_source`标明两种来源。有速度或负载时一并保存及标明来源。旧轨迹仍可读取，已有文件不会被覆盖。

## 参数和来源

- `config/retargeting/sharpa/{right,left}.yaml`：Sharpa独立重定向配置。参照Wuji的自适应对指、Huber位置/方向混合、姿态先验和PIP/DIP耦合实现，代码仅位于`hands/sharpa/`；Wuji源码、配置和入口不变。四指MCP侧摆固定零；拇指额外MCP侧摆按训练对照固定−0.30 rad，消除冗余轴换支，小指CMC仍参与优化。这是模型目标配置，不是硬件安装角或标定。
- Sharpa的`norm_delta=0.4`由连续性回归选定；0.04、0.1、0.2在小幅闭链轨迹上分别出现0.301、0.201、0.150 rad跳变，0.4通过原有小于0.15 rad阈值。未改Wuji的0.04，也未放宽测试阈值。
- `geometry`保存从模型拟合的三轴比例与腕参考偏移，`segment_scaling`保存每指三个任务点比例，`scaling`用于对指位置目标。运行时原始手套canonical输入不变，对指触发仍按原始指尖距离；正逆求解共用归一化任务域。真实关节角、显示尺寸和物理FK不改变，毫米对指误差必须使用`physical_hand_points`，不能使用归一化任务距离冒充实物距离。
- `fit_report.json`与`simulation_validation.json`分别记录几何/比例拟合和实际MuJoCo积分结果；这些是仿真配置，不是穿戴或实物标定。
- 独立Sharpa遥操使用绝对几何目标，经原有0.5秒过渡进入；不再减去零位映射，否则会抵消已拟合的固定关节目标。过渡期间只在Sharpa反馈策略中卸力，完成后恢复反馈。回放接管仍保留原有抓握过渡语义。逆向拇指辅助中点与MCP共用同一比例，保证中点约束。
- `config/feedback/sharpa-{right,left}.json` 保存 Sharpa SDK 关节力矩的偏置、符号、名义负载参考和 Motorhand 各轴的名义力矩限额。接触模式读取仓库根目录下的 `config/feedback/wuji-contact-impedance-right.json`，使用相同的受阻判定与滤波参数。电流门槛除以 Wuji 参考电流得到负载比例，再乘 Sharpa 各轴参考力矩得到 Nm 门槛。
- 接触模式读取根目录下的 `config/feedback/wuji-driver-impedance-right.json`。Sharpa 入口使用 95% 的阻抗预算系数，Motorhand 的 12 轴接触增益统一为 2.7。默认硬度下，各轴接触刚度均为 58.05 Nm/rad。Motorhand 的 12 轴配置上限均为 10 Nm，对应每轴 9.5 Nm 的名义阻抗预算。预算用于估算提交时刻的阻抗指令力矩。
- Sharpa 阻挡目标在模型电机轴上以 0.25 rad/s 限速、5 rad/s² 限加速度。控制器按新鲜姿态时间差更新目标速度并积分角度，接触时从零速度渐入，撤力时清除阻挡目标和目标速度。
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

重定向验收采用左右各357帧真实Motorhand闭链模型输入，213帧训练、144帧独立验证。MuJoCo保留原碰撞/执行器/软限位，只写`ctrl`并积分，不覆盖`qpos`。当前各组能形成实体接触，但指定指腹中心仍有间距，特别是小指仍可先碰中节壳体；不能把优化器指令距离当成实际指腹已经完全对准。完整数值与接触几何对见两个JSON报告。

离线复现工具：`scripts/generate_retarget_fixtures.py`生成闭链输入，`scripts/fit_sharpa_retarget.py`拟合Sharpa几何与比例，`scripts/validate_sharpa_retarget.py`做真实积分验证。已有配置复核使用`--keep-fitted-geometry --thumb-aa-fixed -0.30 --norm-delta 0.4`，比例搜索只使用训练帧，验证帧不参与优化。
