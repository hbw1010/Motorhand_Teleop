# 第三方与用户提供资产

- `assets/wuji-hand2/body/`：来自用户提供的 WujiHand2 模型资产。模型和网格的官方来源、版本风险与官方链接见 `docs/research/wujihand2-official.md`；本项目不重新声明其许可证。
- `assets/hand_motor_ve33/`：来自用户提供的 vE.3.3 手套模型资产。目录内 `THIRD_PARTY_LICENSE` 记录闭链求解器依赖的 Apache-2.0 来源；各模型目录的 README 记录资产结构与当前验证边界。
- `src/motorhand_teleop/ve33_kinematics.py`：从当前工作区已有 vE.3.3 运动学实现导入，作为用户提供的内部工程代码使用；其依赖的模型闭链求解器来源和许可见上条资产目录说明。

- `vendor/wuji-retargeting/`：固定官方 Wuji Retargeting 上游 `v2026.8.17`（commit `531f6ed4250b475d2e9231f54e988fc9b1c5b4ea`）用于离线 Hand2 重定向。其上游许可证和依赖声明以该目录内文件为准；本项目只通过配置和适配器调用，不修改官方优化器语义。
- `vendor/wuji-retargeting/wuji_retargeting/wuji-description/`：官方模型子模块，当前检出 commit `8271644a78d69ed9a4adcf9165d882c64ad33dfa`，仿真使用其中 Hand2 Beta1 `mjcf/right.xml`；模型版本不等于用户真机版本。
- 研究报告中的官方网页、官方仓库和 SDK 资料属于其原作者所有。本项目仅保存引用和工程约束，不复制 SDK 实现。

## 离线训练模型清单与边界

`assets/training_models.json` 是仓库根目录相对路径的入口清单，包含左右手、格式和入口文件 SHA-256；哈希仅标识入口 XML/URDF，不覆盖网格或全部配置，不能当成整个资产包的完整性签名。使用时保持原目录结构，连同网格、配置和许可文件一起复制。用户原始 `assets/assets.zip` 本地保留但不随当前分支分发；核心模型及其网格已独立跟踪，历史网页预览依赖不作为本项目运行入口。

| 资产 | 入口及版本 | 用途和限制 |
| --- | --- | --- |
| 用户提供的 Hand2 URDF | `assets/wuji-hand2/body/urdf/{left,right}.urdf`；具体上游版本未确认，以清单中的文件哈希识别 | 每侧 26 个 STL，visual/collision 共 52 次引用；颜色内联。模型可用于 URDF 几何/关节树消费，不能仅凭文件名判定对应真机版本。 |
| 官方 Hand2 Beta1 MJCF | `vendor/wuji-retargeting/wuji_retargeting/wuji-description/hand2/hand2_beta1/body/mjcf/{left,right}.xml`，子模块 commit `8271644a78d69ed9a4adcf9165d882c64ad33dfa` | 仿真训练模型；模型版本与真机版本不一定相同，实机映射、限位及参数须逐台核验。 |
| vE.3.3 右/左 | `assets/hand_motor_ve33/hand_motor_ve33_{R,L}/hand_closed_chain.xml` 和 `hand_tree_only.urdf`，右手 CAD 导出日期 2026-10-26；左手为镜像而非独立 CAD 导出 | 每侧 93 个 STL。MJCF 有 56 条闭合点约束；URDF 只有树，不能单独表达闭链。Studio 版 `hand_studio_preview.xml` 另有预览范围限制，不应代替完整模型。 |

静态清点时上述用户资产的模型网格引用均存在；URDF 使用内联颜色，vE.3.3 MJCF 直接使用 geom 颜色而非外部材质文件。完整性与加载仍应在接收环境重新验证。vE.3.3 的 `config/joint_space.json`、`joint_roles.json`、`loop_closures.yaml`、`assembly_branches.json` 和 `retarget_targets.json` 与模型一起使用：12 个真实指节坐标不是 12 个电机轴角，也不是 68 个树角；驱动树姿态需闭链求解及装配分支检查。固定四指侧摆、无小指 CMC 自由度；零位不是编码器零位。

vE.3.3 交付仅用于几何和运动学核对：URDF 未配置实测惯量/碰撞，MJCF 使用预览密度、关闭重力和接触碰撞，尚无经实测验证的动力学、传动效率、力矩及物理限位。直接加载不意味着可用于物理可信的接触/力反馈训练或部署；必须另行建模、标定和验收。模型本身不是轨迹、训练数据或物理性能保证，也不能替代设备身份、固件、原始命令期限和人工安全过渡。

许可范围：官方 `wuji-description/LICENSE` 是 Wuji Technology 的 MIT 许可证，分发须保留其版权及许可声明；这不自动证明用户提供的 Hand2 文件与该版本相同或适用同一许可。vE.3.3 模型为用户提供，其几何资产的再分发/训练许可未确认，需向权利人确认；各侧 `THIRD_PARTY_LICENSE` 中的 Apache-2.0 文件及 viewer 内第三方许可仅适用于各自相关的软件，不自动授予 CAD/网格许可。

离线加载（已安装 MuJoCo 的 Python 环境，仓库根目录；不连接设备）：

```sh
python -c 'import json,mujoco; m=json.load(open("assets/training_models.json")); [print(x["id"], mujoco.MjModel.from_xml_path(x["path"]).nq) for x in m["models"] if x["format"] == "mjcf"]'
python -m pytest tests/test_training_assets.py
```

测试检查清单入口哈希、模型内网格引用、URDF 树链接与材质引用，并用 MuJoCo 加载所列 MJCF；它不证明网格内容未改变、URDF 被特定下游加载器成功导入或动力学真实性。下游 URDF 加载器应从原路径加载并自行验证闭链、材质与单位；此处不宣称任意加载器兼容。
