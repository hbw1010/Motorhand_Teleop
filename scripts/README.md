脚本入口保持为空，避免把真机控制逻辑散落在 shell 脚本中。

离线入口使用：

```bash
pixi run demo
pixi run validate-assets
pixi run strict-validation
pixi run mujoco-force-feedback
pixi run mujoco-viewer

`mujoco-force-feedback` 加载 `vendor/wuji-retargeting` 子模块中的官方 Hand2 Beta1
MJCF，使用官方控制指南中的 `kp=3.0`、`kd=0.05`、`effort_limit=1.5 A` 示例参数。
其中电流—力矩常数和外力施加方式是仿真假设，不是真机标定。

`mujoco-viewer` 同时加载官方 Hand2 和真实 vE3.3 闭链 CAD/MJCF；两套模型在 MuJoCo 中按腕部
基座重合显示。vE3.3 先使用资产内固定的 `canonical_to_model` 换回 canonical，再应用当前安装
标定的 `canonical_to_optimizer_row`（canonical 到优化器固定 -90° 列旋转，等价于行向量 +90°），
然后使用 Hand2 MJCF 与重定向 URDF 之间明确的 `Rx(pi)` 基座换向，并在上一版 `-90°` 的基础上
按用户要求再绕掌心指向手指的轴追加 `180°`，当前总装配补偿为 `+90°`；零位腕点/
五指尖只用于计算几何残差，不参与旋转拟合，避免屈伸姿态污染 wrist 方向。后续运动仍由
`WujiRetarget` 输出的 Hand2 关节驱动，不使用手写关节映射。反馈力矩通过小幅视觉偏转叠加，
仅用于观察链路，不等同于真机动力学。

viewer 中 Hand2 使用官方 MJCF 的 `position` actuator，因此 `data.ctrl` 写入的是 20 维关节
目标角，而不是外部计算的力矩；当前按关节语义查找 actuator 并将 WujiRetarget 输出直接写入
ctrl，避免把 `kp*位置误差-kd*速度` 错当成 position actuator 的目标导致 Hand2 几乎不动。
演示轨迹同时驱动四指 MCP/PIP 屈伸，幅度仅用于 MuJoCo 观察，不代表真机安全参数。

Viewer 为了在被渲染的组合场景中跟上扩大后的演示轨迹，将 Hand2 position actuator 的可视化
跟随增益设为 `kp=10.0, kd=0.15`；这不是官网真机控制参数，真机仍须使用 SDK 的安全限幅和标定参数。

由于双模型 viewer 的 GUI 同步会显著低于 MuJoCo 物理步频，viewer 每次刷新推进 100 个
`0.001 s` 物理步，避免窗口看起来“手套在变但 Hand2 不动”；这只是可视化实时化策略。

