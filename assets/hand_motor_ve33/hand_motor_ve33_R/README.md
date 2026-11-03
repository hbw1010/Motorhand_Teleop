# vE.3.3 电机外骨骼 · 12 维真实指节驱动

> 本文件为用户提供资产的原始说明。其中 `hand_controller`、历史网页预览和其他工程的命令不属于本独立项目；本项目可用入口、模型路径和依赖请以仓库根目录 README 与 `assets/training_models.json` 为准。

导出日期：2026-10-26。来源：`vE.3.3/手部电机外骨骼-右手.SLDASM`，默认配置。

`hand_motor_ve33_R/` 与 `hand_motor_ve33_L/` 为并列目录。右手保留原几何，左手由右手关于 `base_link` 的 XZ 平面镜像生成，不代表独立左手 CAD 导出。本文同时说明两侧模型。

## 先看哪里

- 本机闭链预览：右手 <http://localhost:8097/viewer/index.html>，左手 <http://localhost:8098/viewer/index.html>；先启动对应服务。输入为真实指节角，电机角和 DIP 联动角只读。
- 指节驱动开发：仓库 `hand_controller/ve33_kinematics.py` + 本侧模型、`config/joint_space.json` 和原闭链约束。URDF 树本身不改变。
- 在 URDF Studio 中查看：导入 `Studio导入包.zip`，打开其中 `hand_studio_preview.xml`。
- 后续接 MuJoCo：`hand_closed_chain.xml` 保存完整闭链约束，已通过 MuJoCo 加载和运动学坐标核对。

**普通 URDF 只能表达树。单独导入 URDF 不会自动保留联动闭链。** 从动角度需由配套求解器计算；Studio 使用带 equality/connect 的 MJCF 文件。

## 本版确定的结构

四指各 2 个电机、拇指 4 个电机，共 12 个物理执行器。**运动学独立输入改为 12 个真实指节角，电机转角由杆系逆解产生；电机仍是实物执行器。**

| 手指 | 独立指节输入 | 联动与执行器 |
| --- | --- | --- |
| 食指 / 中指 / 无名指 / 小指 | MCP 屈伸、PIP 屈伸，每指 2 维 | DIP 为真实杆系联动输出；MCP 和 PIP/DIP 两个电机角由闭环逆解确定 |
| 拇指 | CMC 屈伸、CMC 侧摆、MCP、IP，共 4 维 | 4 个电机角由同一闭环模型求出 |

四指侧摆和小指 CMC 固定在 CAD 装配位置。没有沿用旧 V1.4 的 21 自由度拓扑。

导出 93 个可见 CAD 零件实例，36 种零件/配置；排除 1 个隐藏轴承实例。相同零件的不同手指长度配置分别导出。
合并明确固定的电机外壳、安装件等后，共 73 个刚体、72 个树关节（4 个固定、12 个电机转轴、56 个物理从动转轴）。28 处闭合转轴用 56 条点连接约束表示。68 个运动树角均由指节输入求出；不是 68 个控制自由度。

## 命名、零位与正向

`config/joint_space.json` 定义 12 个独立坐标及 4 个 DIP 联动坐标的相邻父子刚体和正轴。`config/joint_roles.json` 的 `kinematic_independent` 是控制输入，`motor_actuators` 是物理执行器，`kinematic_solved` 是用于显示的全部 68 个运动树关节；`fixed` 为 4 个固定关节。URDF 中保留全部连接，未把连杆固定住或删除。

`hand.srdf` 将 12 个电机列入 `active_motors` 组，并把 56 个从动关节标记为 `passive_joint`。这是供支持 SRDF 的程序读取的语义文件，不能替代闭链求解器，也不等于已配置好 MoveIt 闭链规划。标记依据：[MoveIt SRDF 文档](https://moveit.picknik.ai/main/doc/examples/urdf_srdf/urdf_srdf_tutorial.html#passive-joints)。

本地预览只提供 12 个指节输入控件，电机和 DIP 只读。普通 URDF / Studio 查看器仍按物理树列出 joint；SRDF 的电机分组不代表新的运动学独立变量。

- 独立名称：`thumb_cmc_flex_joint`、`thumb_cmc_abd_joint`、`thumb_mcp_joint`、`thumb_ip_joint`；四指分别为 `*_mcp_flex_joint`、`*_pip_joint`。
- `*_motor_joint`：逆解得到的物理电机轴角，不是新指节接口的输入名称；`*_drive_link` 为电机杆，`*_connecting_rod_数字` 为从动杆。
- `*_output_joint` 是闭链树分解坐标，**不等于相邻指节真实角**。真实角由相邻刚体旋转定义，并通过树路径上的有向角度和精确表达。
- 角度单位 rad、长度 m；零位相对于当前 CAD 保存姿态，尚未对应编码器零点。
- 指节正轴从原物理轴取得，选择与 child 零位参考 +Z 同向的符号，不强制局部轴为 +Z。电机角不一定与指节角同号。网页箭头随对应 child 刚体，标记位置为 child 原点，不能据此当作虚拟指节的实测转轴中心。
- `12电机定义.csv` 和 `config/motor_definitions.json` 给出每个电机的父子刚体、零位轴心和轴向。`CAD零件对照.csv` 给出原 CAD 名称与导出名。

## 与上一版统一的坐标约定

本次实际参考 `hand_v14_urdf.zip` 中的 `R/hand_tree_only.urdf` 和 `L/hand_tree_only.urdf`。原 `assets/hand_v14_urdf` 目录仅剩缓存，因此只读使用下载目录的完整原始 ZIP；参考文件和来源 ZIP 的 SHA256 已记录在 `config/coordinate_alignment.json`。

- 对齐对象是 **CAD 零位下、以 base_link 表示的完整 X/Y/Z 朝向**，不是只对齐转轴或正运动方向。左右手各 73 个刚体的零位世界旋转，均与对应旧版同侧参考刚体一致。
- `base_link` 保持原基准和单位朝向；右手所有关节原点、实体位置、实际铰链轴及角度数值保持不变。左手位置按 `(x,y,z) → (x,-y,z)` 镜像。没有把新版关节中心搬到旧版位置。
- 四指固定安装座参考旧版侧摆刚体；MCP 输出参考旧版屈伸刚体；已有被动连杆按同名刚体对齐。15 个新增刚体没有旧版一一对应：MCP 主动杆参考相应 MCP 屈伸坐标，PIP/DIP 主动杆参考 PIP，拇指 IP 主动杆参考 IP，新增 MCP 连杆6参考其直接连接的旧版连杆1。逐项规则见 `刚体坐标对齐.csv`。
- **不再强制 `axis="0 0 1"`。** 刚体朝向与实际铰链方向是不同概念。完整朝向对齐后，例如右手食指 MCP 电机为局部 `-Z`，食指 MCP 被动连杆1为局部 `+Y`；拇指部分轴是斜轴。应读取每个 joint 的实际 `axis`，按该轴右手定则解释正角度。
- 所有网格局部变换同步重表达，右手 STL 内容不变；左手 STL 已真实镜像并修正三角形绕序与法线，不依赖负 scale。
- 左右对应姿态满足 `q_L = sign × q_R`；`config/mirror_joint_signs.json` 覆盖全部 68 个运动关节，仅拇指 CMC 侧摆电机为 `-1`，其余为 `+1`。该符号规则保持旧版左右手侧摆习惯。
- 对齐的是零位坐标朝向；运动后的坐标系随各自刚体运动，不要求新旧两种不同机构在任意姿态仍有相同朝向。保留新版 CAD 装配零位，未改为旧版角度或编码器零点。
- `与上一版关节对应.csv` 的 J 编号仅作功能对应；完整朝向参考和换基矩阵以 `刚体坐标对齐.csv`、`config/coordinate_alignment.json` 为准。本地 ±90° 滑块范围不是实物机械限位。

## 文件结构

```text
hand_motor_ve33_R/ 或 hand_motor_ve33_L/
├── hand_tree_only.urdf           标准 URDF 关节树
├── hand.srdf                    12 物理电机分组、56 从动关节标记
├── hand_closed_chain.xml        完整 MJCF 闭链模型
├── hand_studio_preview.xml      Studio 预览版（见限制说明）
├── Studio导入包.zip             本侧 Studio 模型、网格和配套配置
├── meshes/                     93 个零件实例的 STL，单位 m
├── config/
│   ├── loop_closures.yaml       原电机参数化、56从动轴与28闭合轴
│   ├── joint_space.json         12真实指节输入、4个DIP输出及正向
│   ├── assembly_branches.json   33 个装配分支角度约束
│   ├── cad_components.json      CAD 零件、配置与刚体映射
│   ├── motor_definitions.json   电机轴和正向定义
│   ├── joint_roles.json         指节输入 / 电机执行器 / 树解 / 固定
│   ├── coordinate_alignment.json 完整朝向参考与换基矩阵
│   ├── mirror_joint_signs.json   左右对应姿态的角度符号
│   └── source.json              来源、隐藏件及坐标变换
├── scripts/
│   ├── solve_closed_chain.py    原闭链残差/FK内核，供新模块复用
│   ├── viewer_server.py         共享指节网页服务的本侧入口
│   └── mujoco_preview.py        共享MuJoCo指节滑条的本侧入口
├── viewer/                     离线网页依赖和轴向查看界面
├── verification/               本次 frame_alignment.json；R另含历史归档
├── 12电机定义.csv
├── 关节分类.csv
├── 刚体坐标对齐.csv              73 个刚体的参考朝向及对应规则
├── examples/control_active.py   指节输入与电机逆解离线示例
├── CAD零件对照.csv
├── 启动预览.ps1
└── README.md                    本公共说明仅放在右手目录
```

## 运行与接入

运行新的预览和重定向需要**完整仓库的 `hand_controller/` 和本侧模型一起部署**；仅复制单侧目录不够。`Studio导入包.zip` 仅用于 Studio 几何/闭链预览，不包含新的 Python 重定向运行环境。Windows 可使用对应目录的 `启动预览.ps1`，Linux 可使用以下入口。

请通过上面的 `http://localhost:...` 地址进入预览，不要直接双击 `viewer/index.html` 使用 `file://` 打开，后者不能连接闭链接口。

```bash
# 从仓库根目录执行；左右手使用同一代码、各自模型。
python -m pip install -r assets/hand_motor_ve33/hand_motor_ve33_R/requirements.txt
python assets/hand_motor_ve33/hand_motor_ve33_R/scripts/viewer_server.py
python assets/hand_motor_ve33/hand_motor_ve33_L/scripts/viewer_server.py
```
服务按 `viewer/model_info.json` 自动选择右手 8097、左手 8098；可用 `--port` 覆盖。

本地 MuJoCo 原生窗口与中文指节滑条，在仓库根目录运行：

```bash
python -m hand_controller.ve33_preview --hand right
python -m hand_controller.ve33_preview --hand left
```

`requirements.txt` 已包含 MuJoCo；原生窗口还需要 Tkinter 和可用的图形桌面。本项目根目录下已有的 `.pixi/envs/default/bin/python` 环境可直接运行。脚本加载本侧 `hand_closed_chain.xml`，复用 URDF 闭链求解器计算完整姿态，再通过 MuJoCo 正向运动学显示；不进行动力学积分，不连接真实电机，预览过程不修改模型文件。默认手动模式也不连接手套。

控制面板提供 12 个真实指节滑块，单位 rad、步距 0.001 rad，显示 12 个电机逆解值和 4 个 DIP 联动值；支持全部回零。±1.57 rad 是预览范围，不是实物限位。不可达输入保留最后有效姿态；关闭窗口会等待 MuJoCo 渲染线程释放图形资源再退出。

实时数据手套入口复用原有 USB 采集、佩戴档案绑定与 C++ 人体骨架，不经过 WUJI 的20维关节输出：

```bash
python -m hand_controller --model motor_hand --hand left \
  --profile /path/to/left/personal-hand.json --confirm-same-wearing
```

实时入口先要求 `--confirm-same-wearing`，明确确认当前佩戴仍对应已有档案后才连接指定手套；不再由显示窗口按钮授权采集，不需要参考姿态配对。旧 `ve33_preview --live` 入口转交同一应用。更换手侧必须提供对应档案，不自动镜像。已有原始采集服务可用 `--capture-server` 复用；本地USB被其他进程占用时不抢占设备。

`hand_controller/hands/motor_hand.py` 统一封装既有重定向与闭链核心，输出12独立指节、12模型电机角和68树角；`simulation/motor_hand.py` 直接应用树角，不再次求解闭链。青色为统一掌坐标、按掌长缩放的人体21点目标，橙色为实际模型任务点。显示原始帧龄、拟合RMS和闭链误差，全部角度可由 `HandSample.pose` 读取；手动模式仍保留滑块及电机/DIP数值。

骨架源由顶层应用持有，各型号 `HandSession` 独立推进；可再指定 `--model wuji_hand2` 同时运行另一型号仿真。显示过期只冻结显示，不反向暂停处理线程；源失效、身份变化或求解故障由处理实例锁定，须显式恢复。关闭窗口不关闭共享源或其他实例；Ctrl+C或 `--duration` 退出整个应用。`--no-viewer` 不加载显示路径。motor hand没有真实驱动，本轮仅统一后处理和仿真接口，原人体骨架框架不变。

### 指节输入 → 杆系逆解 → 电机输出

```python
from hand_controller.ve33_kinematics import Ve33Kinematics

hand = Ve33Kinematics("right")  # 左手使用 "left"
pose = hand.solve({"index_mcp_flex_joint": 0.08, "index_pip_joint": 0.05})
joint_angles = pose.joint_positions       # 12 个真实指节输入
motor_angles = pose.motor_positions       # 12 个物理电机逆解输出
dip_angles = pose.dependent_positions     # 4 个 DIP 联动输出
tree_angles = pose.all_positions          # 68 个树角，整体用于 URDF/MuJoCo 显示
```

`solve()` 接受部分指节输入，未指定坐标保持原值；成功才提交。未知名称、电机名、非有限角、不可达或奇异姿态会报错并保留原状态。`evaluate(targets, seed=pose)` 只试算，不提交；`solve(targets, seed=pose)` 可沿用已连续试算成功的分支种子。`local=True` 限定0.04 rad邻域和求解次数，不递归长距离追踪；`reset()` 回 CAD 零位。单次变化大于4.8 rad要求分段，这只是求解资源保护，不是机械限位。

示例：`python assets/hand_motor_ve33/hand_motor_ve33_R/examples/control_active.py --joint index_pip_joint --angle 0.05`。左右手该输入均得到 PIP/DIP 电机约 `0.08774598 rad`、DIP 约 `0.01192639 rad`，说明指节角、电机角和树关节角不能互换。

### MediaPipe 21 点 → 12 维指节重定向

```python
from hand_controller.ve33 import Ve33Retargeter

retargeter = Ve33Retargeter("right")
pose = retargeter.step(points_m)          # 有限的 (21,3) 米制点，不需要参考帧
print(pose.motor_positions)
print(retargeter.last_diagnostics)
```

原 WUJI 的20维接口不变；不能把此12维结果交给原 `HandController` / WUJI 硬件输出。新模块复用已有 MediaPipe 掌坐标归一化，读取本侧 `config/retarget_targets.json` 的固定CAD任务点、掌旋转和骨长，优化位置、末节方向、捏合关系和四指真实MCP/PIP有向角；有向角项辅助保持深屈曲方向。解析雅可比包含杆系被动运动。

**不需要额外配对标定，不把首帧自动当零位。** 人体21点以腕为共同原点，按模型与人体的腕—中指MCP掌长比统一缩放；参考WUJI指尖位置任务保留共同原点和尺度，不再按机器人指根及各段骨长重建目标。输入指尖接触时目标也接触，非零指间矢量同样按统一尺度保留。机器人任务点和骨长仍固定，观测不会改写模型。每帧热启动并限制候选搜索；单指到达可达边界时搜索其余可行坐标，而不是锁住整手。每个接受的候选和最终提交都必须通过闭链校验，未放宽装配边界。

对指使用两级连续权重：原有宽范围alpha保留整手/末节任务混合；另以指尖距离相对0.15倍掌长的高斯衰减激活接触优先级，相对指尖矢量获得最高64的额外权重。这样不会把附近未接触的手指全部当作强对指任务。仍跟踪原始非零间距，不把目标强行投影到接触点，不按指名特判。接触状态下，阻尼随当前任务雅可比的曲率增加，抑制欠约束方向的大步；仅使用固定绝对阻尼或统一放大全部捏合项，可能使PIP落入反向局部解。接触未激活时保留原有阻尼。

当前显示直接设置闭链姿态并执行 `mj_forward`，不是动力学位置PD。跟随加速使用同状态闭链导数缓存、批量任务点雅可比和解析导数预测热启动；每次候选最多使用既有0.04 rad局部邻域，每帧最多两轮，最终仍检查装配边界、连杆跳变、闭链残差及奇异性。工作线程按20ms周期补足剩余等待，不再在求解后固定追加10ms。在本次对指目标修正之前，使用同一归档张手→深握阶跃，关节误差降至初始10%的时间由约2.94秒降至0.73秒，求解中位耗时由38.1ms降至26.4ms；实际原生窗口同场景约0.81秒。以上为此前记录输入测试，不是新目标构造的性能复验、USB端到端延迟、任意动作保证或硬实时承诺。

固定侧摆、DIP联动及人体/外骨骼差异会产生不可达目标。对指优先时允许绝对位置与末节方向拟合作出让步，整手RMS可能增大；不能只用整手RMS判断接触是否改善。各项仍为软任务，有限迭代也不保证全局最优或任意输入下精确接触。`task_rms_m` 是固定任务点的模型拟合残差，不是实测物理指尖精度。转换后的观测目标不再改形，不通过裁剪目标隐藏误差。

离线 CLI 输入 `.npy`，目标为 `(21,3)` 或 `(N,21,3)`：

```bash
python -m hand_controller.ve33 --hand left --points sequence.npy
```

逐帧输出 JSON，含指节角、电机角、DIP、闭链间隙和拟合残差。该流程不连接硬件；电机限位、编码器零点及实物闭环控制尚未标定。

本地滑块支持拖动时连续联动，步距 0.001 rad；只保留最新目标、逐小步求解，避免旧拖动指令排队。显示的每一帧均使用已通过闭链校验的姿态，不对连杆做独立角度插值。视角旋转采用阻尼缓动。求解器按手指分块，并使用解析雅可比减少重复计算。

## 检查结果与使用边界

当前回归覆盖左右手真实指节/电机/树角往返、不可达回滚、解析雅可比、无配对首帧、相机坐标不变性、四指对指目标保持、全骨架相对几何和尺寸不变性、实际模型四指连续对指及旧接触释放、单指边界不阻塞整手、深屈曲方向，以及实时原始期限、身份、乱序、暂停撤销在途结果和关闭来源；与原 WUJI、原生绑定、实时来源和仿真桥相关测试合计208项通过。此前已使用归档真实编码器数据经过正式C++骨架、Native25/21转换、新重定向和实际MuJoCo窗口进行回放；这是离线记录驱动的运行验证，**不是当前佩戴的真实USB遥操或实物验收**。此前双手网页和原生滑条已实际核对PIP、电机与DIP联动。

本次实际对指验收使用归档弯指骨架叠加合成接触，左右手分别连续执行食指→中指→无名指→小指→小幅张开→松开，每阶段120帧。实际模型任务点的对指间距分别约为1.03mm、1.62至1.65mm、2.27至2.28mm、3.22至3.24mm。小幅张开的目标间距4.83mm，实际约6.62至6.63mm，未被强制吸合；松开后不锁定旧接触。全序列最大闭链间隙小于5.34e-8m，独立关节单帧变化不超过0.08rad。另在原装配分支内求得无名指、小指的近零间距闭链接触对照解，说明此前残差不能直接归因于机构不可达。已重启修正后的实时仿真并核对实际显示；上述数值仍是合成回放结果，不是当前佩戴、CAD表面碰撞或实物接触精度保证。

- 坐标对齐报告仍为各侧 `verification/frame_alignment.json`：每侧 73 个零位坐标系、93 个 STL、51 个运动姿态通过独立验收；最大零位旋转矩阵元素差小于 `2.2e-15`，最大朝向角误差小于 `2.3e-15 rad`。这些是此前坐标对齐验收，本次未修改模型几何或重复冒充该验收。
- 运动验收含单电机 ±0.2 rad、归零及混合 ±0.1 rad 组合；小指 PIP/DIP 仅测正向及归零。右手与原模型、左手与镜像原模型的全部主动/被动角度吻合，最大偏差小于 `1.3e-14 rad`；最大闭链误差小于 `1.5e-11 m`。
- MuJoCo 3.12.0 分别加载两侧完整及 Studio 版 MJCF，逐姿态核验刚体世界变换、URDF FK 与闭合约束；同时渲染核对右手外形不变、左手镜像正确。本次没有重新运行 Studio 网页自身的求解器验收。
- **Studio 的小指 MCP 预览范围仍为 -0.45～+0.6 rad**，完整 URDF / MJCF 不增加该范围。这是既有数值预览保护，不是实物机械限位。
- 对齐前的几何、1,468 步运动和 Studio 624 步等历史报告已归档到右手 `verification/pre_alignment/`，不冒充本次或左手的新验证。该目录的 `input_right.zip` 保存对齐前模型及配置；右手网格未改变。
- CAD 源文件未参与编辑；当前核验使用导出模型、网格和元数据，不是加工公差验收。
- 当前交付用于几何和运动学核对。URDF 未配置实测惯量/碰撞，MJCF 使用统一密度 1000 kg/m³ 产生预览惯量、关闭接触碰撞、关闭重力；电机力矩、传动效率、实物限位和控制器尚未标定，未完成动力学验收。

源码提取工具：jsk-ros-pkg/solidworks_urdf_exporter2（sw2robot 0.4.4）。第三方许可见 `THIRD_PARTY_LICENSE` 与 `viewer/vendor/`。
