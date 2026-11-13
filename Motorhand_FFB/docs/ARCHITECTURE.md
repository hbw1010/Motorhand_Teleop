# 架构与运行边界

## 职责

| 位置 | 职责 |
| --- | --- |
| `domain.py`、`bidirectional.py` | 设备状态、任务空间、共享正逆解和单主导切换 |
| `ve33_kinematics.py`、`ve33_geometry.py`、`native/ve33_geometry.cpp` | vE.3.3 闭链及装配分支、模型电机/真实指节换算；C++ 批量状态、几何和点雅可比计算 |
| `retargeting.py`、`landmarks.py` | vE.3.3 任务点与官方 WujiRetarget 适配；不建立直接关节映射 |
| `real_hand/wuji_hand2.py`、`real_hand/motor_hand.py` | 并列的设备生命周期、身份/固件、单位、标定、限位和发送边界 |
| `real_hand/control.py` | 推理/回放与手套接管的单主导控制周期 |
| `real_hand/runtime.py` | 双设备遥操作和单侧跟随入口 |
| `real_hand/inference.py` | 交给模型调用方的会话、观测、执行、主导方切换与释放 |
| `real_hand/force_*.py`、`real_hand/contact_feedback.py` | 受阻估计、阻抗、重定向雅可比、力矩投影及曲线输出；不反写位置求解 |
| `real_hand/trajectory.py`、`real_hand/playback.py` | 实测状态异步录制、文件校验、按源时间插值、回放和按键接管 |
| `real_hand/native/` | 设备 C ABI、官方 SDK 运输层、期限/故障边界及离线 C/C++ 自检 |
| `../hardware/encos_dexhand/` | 共用的 12 轴 Encos SDK、协议、CAN 工作线程和分离的现场标定 |
| `../assets/hand_motor_ve33/`、`assets/wuji-hand2/`、`vendor/wuji-retargeting/` | 用户提供的闭链/Hand2 模型和固定版本官方优化器、MJCF |

## 数据与控制权

手套编码器 raw degree 经 `zero/sign/scale` 得模型电机 m12 rad，再用闭链 FK 获得真实指节 q12。q12 经 21 点与官方优化器求 Wuji q20；逆向从 Wuji 的实测 q20 通过同一 canonical 点空间求 q12、完整 CAD 树及 m12。电机角不等于指节角；关节与电机目标不能按名称或数组下标直接替代。正向四指侧摆固定零；真实 Wuji 侧摆仍用于实测 FK。

计算调度及逆解约束仍在 Python；数值优化使用 SciPy SLSQP 与 NumPy/LAPACK 原生计算核，并非整个实时链路都已迁为 C++。

推理与回放模式中，Wuji 接收模型/轨迹 q20；手套根据 Wuji 实测姿态的异步逆解跟随。接管模式中，手套停止位置跟随，由实际手套 q12 主导 Wuji，并可通过接触估计力矩前馈反馈到手套。`HandoverController` 对位置命令及手套模型电机轨迹分别平滑接入；两设备的 SDK 无跨设备原子提交能力。

力反馈的 `effort` 是电流 A，不是净外力；`contact_feedback` 的接触状态由误差、速度、电流持续条件估计，输出为软件虚拟反力。既有完整重定向雅可比转置与 CAD 传动投影负责坐标转换；手套原始轴名义 Nm 不冒充电机实测值。`force_feedback` 的局部雅可比由独立进程更新，姿态或时效不匹配时撤力、重新计算并渐入；不会改写位置输出。

原生独立力反馈命令（`force-contact` / `force-linear`）启用 `zero_aligned_start`：先分别回零手套和Wuji，再发布 `clamp(retarget(glove) - retarget(glove_zero))`，启动使用既有五次平滑过渡。零位参考在无输出时用同一优化器收敛30次，仅作为固定角度偏移，不修改模型资产或设备硬件原点；旧的随运动衰减偏置仅保留给推理/回放接管。反馈Jacobian的常数偏移导数不变，但被最终关节限位截断的行置零，其余乘同帧启动过渡系数，使用副本不续期、不累计缩放。

## 生命周期与时效

- 只读发现/连接不获得运动输出权。显式 `start()`/`--execute` 前检查身份、固件、在线状态、标定和轨迹限位；首次手套输出先伺服回零。
- Wuji 启动清错后，连续观察全部关节失能且无停机故障至少 100 ms，并使用该窗口之后的新鲜位置与诊断帧初始化保持目标，再请求使能。固件 2.13.1 实测中，清错后 1–2 ms 的新鲜 `Ready` 帧不足以保证后续使能生效；这个间隔只用于启动，仍须逐轴确认使能，不延长正常控制的反馈或命令期限。
- 独立力反馈启动新增Wuji显式回零，实测容差0.03 rad，失败时停止两侧。CLI通过无锁原子信号标志和回零停止线程传播取消，Wuji回零可及时中断；手套既有同步回零仍只在调用前后检查取消。
- 真实反馈使用其接收时刻；命令保留原始 100 ms 期限。可选推理输入时间戳进一步收紧期限，不可用较新的设备帧续期旧模型目标。
- 回放逆解 worker 只计算，不访问硬件。一个仍在原始期限及发送余量内的结果可以跨周期使用；失效时保持 Wuji 已发目标、暂停回放与过渡、等待新鲜结果，超时停止。重新接入从手套实测电机姿态锚定。
- 故障或超期停止本会话取得的输出；硬件/通信故障可能导致停止请求无法确认，仍须实体急停或断电。可视化通过非阻塞 UDP 旁路，不访问控制权限。

## 验证边界

MuJoCo 模型、录制重放和带机械滞后的数值设备用于验证数学、时序和切换；它们不是电机扭矩标定，也不证明真实接触力、人体安全或真机时延。入口及资产许可见根目录 `README.md` 与 `THIRD_PARTY_NOTICES.md`。
