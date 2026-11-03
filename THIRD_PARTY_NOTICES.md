# 第三方与用户提供资产

- `assets/wuji-hand2/body/`：来自用户提供的 WujiHand2 模型资产。模型和网格的官方来源、版本风险与官方链接见 `docs/research/wujihand2-official.md`；本项目不重新声明其许可证。
- `assets/hand_motor_ve33/`：来自用户提供的 vE.3.3 手套模型资产。目录内 `THIRD_PARTY_LICENSE` 记录闭链求解器依赖的 Apache-2.0 来源；各模型目录的 README 记录资产结构与当前验证边界。
- `src/wujihand2_force_glove/ve33_kinematics.py`：从当前工作区已有 vE.3.3 运动学实现导入，作为用户提供的内部工程代码使用；其依赖的模型闭链求解器来源和许可见上条资产目录说明。

- `vendor/wuji-retargeting/`：固定官方 Wuji Retargeting 上游 `v2026.8.17`（commit `531f6ed4250b475d2e9231f54e988fc9b1c5b4ea`）用于离线 Hand2 重定向。其上游许可证和依赖声明以该目录内文件为准；本项目只通过配置和适配器调用，不修改官方优化器语义。
- `vendor/wuji-retargeting/wuji_retargeting/wuji-description/`：官方模型子模块，当前检出 commit `8271644a78d69ed9a4adcf9165d882c64ad33dfa`，仿真使用其中 Hand2 Beta1 `mjcf/right.xml`；模型版本不等于用户真机版本。
- 研究报告中的官方网页、官方仓库和 SDK 资料属于其原作者所有。本项目仅保存引用和工程约束，不复制 SDK 实现。
