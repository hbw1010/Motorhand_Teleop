"""监视曲线的数据定义；只描述现有实现，不参与控制或数值换算。"""

METRICS: dict[str, dict[str, str]] = {
    "glove_projected_torque_nm": {
        "name": "增益后投影目标（保护前）",
        "unit": "Nm",
        "kind": "算法",
        "meaning": "反馈算法投影到手套电机轴的名义力矩目标，已包含配置增益，尚未经过控制器启动渐入、输出限幅、变化率限制和电机软限位保护。",
        "formula": "接触模式：τ投影 = gain × reflection_gain × M × τ虚拟。M 为原始电机轴换算后的 B⁻ᵀJ_effᵀ；B=∂模型电机角/∂模型指节角，J_eff=∂最终Wuji目标/∂模型指节角的有效局部近似。线性模式：每轴 τ投影 = C × clamp(-gain × Σ(Mij × lj)/Σ|Mij|, -1, 1)，lj=clamp((Ij-biasj)×effort_signj/Imaxj, -1, 1)，C为该原始轴扣除量化余量的配置上限；矩阵整行为零时结果为零。",
        "coordinates": "12维，按原始电机J1–J12排列；模型电机力矩按映射关系重排并乘 sign×scale，正负沿原始电机轴。接触模式的有效矩阵已包含零位对齐模式中的过渡系数及Wuji限位行置零。",
        "time_source": "glove_output_source_ns：对应已完成的异步输出计算所用Wuji接收时刻与手套最旧轴接收时刻的较早者；不是当前遥测帧的计算时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "这是软件目标，不是测得的接触力矩；接触模式与线性模式的物理含义不同。几何无效时该曲线缺失，不能把缺失当零。",
    },
    "glove_requested_torque_nm": {
        "name": "控制器保护后请求（编码前）",
        "unit": "Nm",
        "kind": "指令",
        "meaning": "控制器准备交给手套驱动的名义力矩，已完成启动渐入、幅值限制、变化率限制和模型电机软限位方向检查。驱动随后仍会再次保护并量化。",
        "formula": "先将投影目标乘五次平滑渐入系数r(x)=x³(10-15x+6x²)，再依次限幅、按已确认输出历史限制变化率、在软限位处屏蔽向外力矩；最后重排到原始轴并乘sign×scale。接触模式换向先置零，线性模式按双向变化率限幅。",
        "coordinates": "12维原始电机J1–J12轴，与同图SDK名义指令、电流换算力矩使用相同通道顺序和方向。",
        "time_source": "glove_output_source_ns，表示产生该请求的异步计算输入源时刻；不是驱动提交时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "不能称最终输出或实际力矩；驱动层按最新电机角执行的软限位保护和12位量化还未计入。",
    },
    "glove_target": {
        "name": "已提交名义力矩（SDK量化值）",
        "unit": "Nm",
        "kind": "指令",
        "meaning": "驱动实际采用的MIT力矩编码对应的名义数值，包含驱动层限幅、软限位方向修正和量化；仅在glove_target_unit为Nm时显示。",
        "formula": "τ名义=-10+code×20/4095，code为最终12位MIT力矩编码。控制器请求先受原始轴配置上限及半LSB余量约束，再取整并执行编码边界检查；遥测已换回原始轴数值。",
        "coordinates": "12维原始电机J1–J12轴；单位是SDK协议定义的名义Nm，不是经独立传感器标定的轴输出力矩。",
        "time_source": "glove_submit_ns：本次12轴命令中最后一个提交时刻。驱动1是主机串口写入完成时刻；驱动2是交给SDK的时刻，SDK在后续5ms周期发送。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "提交成功不证明电机执行或达到该力矩。12位编码无精确零点，零请求可能显示约±0.002442Nm；位置模式下本曲线不显示。",
    },
    "glove_estimated_torque_nm": {
        "name": "手套电流换算力矩（参考估算）",
        "unit": "Nm",
        "kind": "估算",
        "meaning": "把手套驱动回读电流乘监视器配置的每轴Nm/A系数，作为观察负载变化的参考值。",
        "formula": "τ估算,i=glove_current_a[i]×glove_nm_per_a[i]；当前默认系数为5.1Nm/A。这里不扣除偏置，也不补偿摩擦、惯性或传动效率。",
        "coordinates": "12维原始电机J1–J12轴；电流保留驱动原始正负，换算系数的符号决定估算力矩方向。",
        "time_source": "glove_motor_received_ns[i]：该原始电机最新反馈的主机接收时刻；驱动2为发现新SDK快照的时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "5.1Nm/A来自空载参考，未证明所有电机在当前工况的真实力矩常数；不是力矩传感器测量，也不是人体接触力。系数或电流缺失时不显示。",
    },
    "wuji_reflected_torque_nm": {
        "name": "Wuji负载反射估算（未乘增益）",
        "unit": "Nm",
        "kind": "估算",
        "meaning": "将Wuji各轴电流估算的执行器力矩，经本次反馈计算的有效几何映射反向投影到手套原始电机轴，供同坐标对比。",
        "formula": "τ反射,i=-Σj(Mij×I配对,j×Kt,j)。M=wuji_to_glove_torque_matrix，是按原始电机轴重排并乘sign×scale的B⁻ᵀJ_effᵀ；不乘gain或reflection_gain。I配对取wuji_projection_current_a，不能用当前帧wuji_current_a替代。",
        "coordinates": "12维原始电机J1–J12轴。负号用于向主手反射从手执行器负载；有效矩阵包含零位对齐模式中的过渡系数及Wuji限位行置零，固定零位相减本身不改变局部导数。",
        "time_source": "wuji_projection_received_ns：与矩阵一同封装在已完成异步结果中的Wuji电流接收时刻。矩阵采用的重定向雅可比可能来自较早的有效快照，并非各输入同刻采样；旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "当前Wuji默认Kt全为未知，依赖未知Kt的轴保持缺失；矩阵整行零时可为零。该估算未扣偏置、摩擦和惯性，不能当作净接触力矩，也不能要求与接触阻抗算法目标重合。",
    },
    "glove_friction_torque_nm": {
        "name": "自由空间摩擦补偿前馈",
        "unit": "Nm",
        "kind": "指令",
        "meaning": "driver_impedance模式在有效自由空间提供的摩擦补偿前馈；旧纯力矩路径没有此分量。",
        "formula": "由规划器配置的速度死区、库仑和黏性项生成，受预算与渐入限制，发布实际编码后的自由轴分量。",
        "coordinates": "12维原始电机J1–J12轴，与提交名义总输入同方向。",
        "time_source": "glove_driver_velocity_received_ns[i]记录规划所用位置源时刻，实际提交时刻另见glove_submit_ns。",
        "limitations": "这是补偿指令，不是测得的摩擦。接触轴的协议零码残差不算摩擦补偿；旧模式缺失不代表实测摩擦为零。",
    },
    "wuji_target_rad": {
        "name": "Wuji已提交目标角",
        "unit": "rad（旧兼容曲线可显示°，新验证页保留rad）",
        "kind": "指令",
        "meaning": "本控制周期成功提交给Wuji发送接口的最终20轴位置目标，包含当前模式的重定向、过渡和关节范围处理。",
        "formula": "零位对齐接管模式：q目标=(1-b)×q接管锚点+b×clamp(retarget(手套模型关键点)-q模型零位参考, 下限, 上限)。零位参考固定相减，不随累计运动衰减；其他模式使用各自的过渡目标。旧兼容曲线可乘180/π；新六组验证页保留rad。",
        "coordinates": "Wuji原生20关节坐标，按拇指、食指、中指、无名指、小指各4轴排列；使用Wuji设备零位和关节方向。",
        "time_source": "wuji_submit_ns：本周期Wuji发送接口成功返回后的主机单调时刻；不是设备开始执行或到达目标的时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "目标角不保证机械手已经达到；它不等于手套原始电机角。图中目标与反馈分别按提交和接收时间绘制，不能把水平间隔直接当作纯机械响应延迟。",
    },
    "wuji_position_rad": {
        "name": "Wuji反馈关节角",
        "unit": "rad（旧兼容曲线可显示°，新验证页保留rad）",
        "kind": "实测",
        "meaning": "Wuji反馈帧中的20轴关节位置，是与目标角比较的设备反馈值。",
        "formula": "直接取Wuji反馈positions；反馈原值为rad；旧兼容角度曲线可乘180/π，新六组验证页保留rad。",
        "coordinates": "Wuji原生20关节坐标，与wuji_target_rad同序、同零位、同方向。",
        "time_source": "wuji_received_ns：Wuji反馈帧的主机单调接收时刻，读取状态时不会刷新。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "主机接收时间包含通信影响，不等于设备内部采样时刻；反馈角不直接表示接触、力或跟踪性能合格。",
    },
    "glove_raw_position_deg": {
        "name": "手套原始电机反馈角",
        "unit": "°",
        "kind": "实测",
        "meaning": "手套驱动回读的12个原始电机角，尚未减去现场校准零位，也未乘模型映射的sign和scale。",
        "formula": "直接取驱动反馈angle_deg；MIT反馈时由16位编码解码为(-5+code×10/65535)×180/π，伺服反馈使用设备返回角度。",
        "coordinates": "原始电机J1–J12顺序及原始编码器角度坐标；含当前现场零位偏移。中末节驱动电机角不等于中节指节角。",
        "time_source": "glove_motor_received_ns[i]：各电机各自的反馈主机接收时刻；驱动2为发现新SDK快照的时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "12轴并非严格同步采样，且电机之间零位/方向可能不同；不能直接要求数值等于Wuji关节角或手套模型指节角。",
    },
    "glove_position_rad": {
        "name": "手套闭链模型指节角",
        "unit": "rad（旧兼容曲线可显示°，新验证页保留rad）",
        "kind": "算法",
        "meaning": "根据手套电机反馈及校准映射求解闭链模型后，得到的12个相邻指节角；它是运动学重建值。",
        "formula": "模型电机角=(原始电机角-校准zero_deg)×sign×scale×π/180；以这些电机角求解68维闭链树，再按模型关节定义提取12维joint_rad，旧兼容曲线可乘180/π；新六组验证页保留原始rad或rad/s。",
        "coordinates": "模型关节顺序：拇指4轴；其余四指各为指根屈伸、中节屈伸。这里使用模型关节坐标，不能套用原始J1–J12电机角的物理含义。",
        "time_source": "glove_received_ns：用于该模型姿态的12轴反馈中最早的主机接收时刻；不是模型求解完成时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "依赖校准零位、CAD几何和闭链装配分支；不是独立关节传感器实测值，求解失败后的姿态不会作为有效新帧发布。",
    },
    "glove_current_a": {
        "name": "手套驱动反馈电流",
        "unit": "A",
        "kind": "实测",
        "meaning": "手套各电机驱动回读的带正负电流，用于观察执行器负载变化。",
        "formula": "直接取驱动解码电流；SLCAN伺服反馈为有符号16位值/100，MIT反馈为-2+12位编码×4/4095；G4路径使用SDK快照current_a。",
        "coordinates": "12维原始电机J1–J12顺序，正负遵循各驱动原始电机轴；未做模型sign/scale映射，也未扣电流偏置。",
        "time_source": "glove_motor_received_ns[i]：该轴反馈的主机接收时刻；驱动2为发现新SDK快照的时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "电流不是接触力或直接力矩；摩擦、加速、保持和外载都会影响读数，分辨率取决于反馈协议。",
    },
    "wuji_current_a": {
        "name": "Wuji驱动反馈电流",
        "unit": "A",
        "kind": "实测",
        "meaning": "Wuji反馈帧提供的20轴带正负电流。虽然原生字段称effort_a，这里量纲仍是安培。",
        "formula": "直接取Wuji反馈effort_a；不乘Kt、不扣偏置、不取绝对值、不低通。",
        "coordinates": "Wuji原生20轴顺序，电流正负遵循设备报告；监视器未自动套用控制配置中的wuji_effort_sign。",
        "time_source": "wuji_received_ns：当前Wuji反馈帧的主机接收时刻。它可能晚于已完成异步反馈计算所使用的配对电流；旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "不能由软件电流上限反推真实Nm，也不能把有符号原始电流与滤波绝对负载电流混为一谈。缺少电流反馈时不显示。",
    },
    "wuji_velocity_rad_s": {
        "name": "Wuji反馈关节速度",
        "unit": "rad/s（旧兼容曲线可显示°/s，新验证页保留rad/s）",
        "kind": "实测",
        "meaning": "Wuji反馈帧报告的20轴关节速度，用于观察方向、运动快慢及是否接近停滞。",
        "formula": "直接取反馈velocity_rad_s；旧兼容曲线可乘180/π；新六组验证页保留原始rad或rad/s。监视器不以相邻位置帧差分补齐这条曲线。",
        "coordinates": "Wuji原生20关节坐标，正向与对应关节角增大方向一致。",
        "time_source": "wuji_received_ns：Wuji反馈帧的主机接收时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "SDK未提供速度时本曲线缺失；控制器内部可能用位置差分作为接触判断的备用速度，两者不能默认相同。接近零也不单独证明接触。",
    },
    "wuji_tracking_error_rad": {
        "name": "Wuji目标减反馈角差",
        "unit": "rad（旧兼容曲线可显示°，新验证页保留rad）",
        "kind": "算法",
        "meaning": "同一遥测周期中的已提交目标角减去本周期读取的Wuji反馈角，反映两者当时的数值差。",
        "formula": "e[i]=wuji_target_rad[i]-wuji_position_rad[i]；旧兼容曲线可乘180/π；新六组验证页保留原始rad或rad/s。正值表示目标沿该关节正方向领先于反馈。",
        "coordinates": "Wuji原生20关节坐标，与目标和反馈角使用相同零位、顺序和方向。",
        "time_source": "曲线横轴使用wuji_submit_ns；减数的反馈来自wuji_received_ns，通常早于该目标提交，未进行时间插值对齐。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "这是跨两个时刻的角差，不能直接当作纯稳态误差或时延；运动、接触受阻、软限位和响应滞后都可能使角差增大，不能一概理解为越小越安全。",
    },
    "wuji_estimated_torque_nm": {
        "name": "Wuji电流换算力矩（未标定则缺失）",
        "unit": "Nm",
        "kind": "估算",
        "meaning": "仅在配置有效每轴Nm/A系数时，才把当前Wuji反馈电流换算成该关节的参考执行器力矩。",
        "formula": "τ估算,i=wuji_current_a[i]×wuji_nm_per_a[i]；当前默认20轴Kt均为null，因此均不显示。这里不扣除偏置，也不使用软件限额猜测Kt。",
        "coordinates": "Wuji原生20关节顺序；Kt必须包含正确量纲与符号，使结果对应Wuji关节坐标。尚未投影到手套电机轴。",
        "time_source": "wuji_received_ns：当前Wuji电流反馈帧的主机接收时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "系数未知或电流缺失时保留缺失，不能填零；即使Kt已知也未分离摩擦、惯性和外部接触，不能视为实测接触力矩。",
    },
    "virtual_hand_torque_nm": {
        "name": "Wuji坐标虚拟阻抗力矩",
        "unit": "Nm",
        "kind": "算法",
        "meaning": "接触阻抗模式根据受阻估计产生的20轴虚拟反作用力矩，已经过算法输出低通，尚未乘反馈总增益，也未投影到手套电机。",
        "formula": "受阻有效时：e=q目标-q反馈，n为锁定的受阻方向，p=max(n×e-error_off,0)，v为限幅后的n×(目标速度-反馈速度)。原始请求τ=-n×max(w载荷×K×p+w阻尼×D×v,0)，再按α=1-exp(-2π×output_filter_hz×Δt)低通；权重由滤波负载与速度计算。释放、回退或不可观测时按算法置零。",
        "coordinates": "Wuji原生20关节坐标；反作用方向由跟踪误差锁定，而不是直接照搬电流正负。随后才通过局部雅可比映射到手套原始电机轴。",
        "time_source": "glove_output_source_ns：已完成异步反馈计算所用输入中的较早主机接收时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "这是人为设定刚度、阻尼与权重产生的Nm量纲软件输出，不是Wuji实际输出力矩；线性反馈或非接管模式下不提供。几何无效时也可能保留算法内部结果，但不会据此产生有效投影输出。",
    },
    "contact_active_mask": {
        "name": "受阻估计锁存状态",
        "unit": "0 / 1",
        "kind": "状态",
        "meaning": "1表示该Wuji关节的受阻估计已激活且尚未完成释放去抖；0表示未激活。这里显示每轴状态，不是整只手的接触总状态。",
        "formula": "激活需关节在有效映射中可观测，|目标-反馈|达到error_on、滤波负载电流达到current_on、|速度|不超过stalled_speed，并持续dwell；释放依据error_off/current_off及release_dwell滞回。",
        "coordinates": "20个Wuji关节布尔值，按Wuji原生轴顺序；无力矩方向或大小含义。",
        "time_source": "glove_output_source_ns：对应已完成异步估计的输入源时刻。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "不是触觉传感器确认的接触。释放等待期间可能仍为1而虚拟力矩已为0；映射无效时不能只凭本掩码判断正在输出，须结合feedback_state/contact_state。",
    },
    "wuji_filtered_current_a": {
        "name": "受阻估计使用的滤波负载电流",
        "unit": "A",
        "kind": "算法",
        "meaning": "接触阻抗算法用于受阻触发和载荷权重的电流幅值，先扣偏置、取绝对值，再做低通；与原始带符号电流不同。",
        "formula": "load=|I-wuji_current_bias_a|；初次有效输入直接初始化，后续I滤波←I滤波+[1-exp(-2π×current_filter_hz×Δt)]×(load-I滤波)。Δt来自新的配对输入源时刻之差。",
        "coordinates": "Wuji原生20轴顺序；数值非负，不再包含电流或力矩方向，未乘Kt。",
        "time_source": "glove_output_source_ns：已完成异步反馈计算的输入源时刻；原始I来自该计算所用Wuji帧，不一定是当前遥测帧电流。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "低通会产生滞后，阈值只用于受阻估计；当前仅在接触模式且反馈几何有效时发布，缺失不能解释为零负载。",
    },
    "timing_ms": {
        "name": "发布时帧龄与控制周期耗时",
        "unit": "ms",
        "kind": "状态",
        "meaning": "四条线依次表示发布时Wuji反馈帧龄、手套最旧轴反馈帧龄、主控制周期耗时，以及已完成手套输出所用输入源的帧龄。",
        "formula": "[published_ns-wuji_received_ns, published_ns-glove_received_ns, cycle_completed_ns-cycle_started_ns, published_ns-glove_output_source_ns]÷10⁶。时间缺失或顺序倒置时该项缺失。",
        "coordinates": "索引0=Wuji帧龄，1=手套帧龄，2=主控制周期耗时，3=手套输出源帧龄；都使用同一主机单调时钟。",
        "time_source": "曲线横轴为published_ns，即原生遥测组包发布时刻；帧龄也固定在该时刻计算。旧兼容曲线缺时间可回退timestamp_ns；新六组验证页缺原始时间就不绘制该点。",
        "limitations": "不包含发布后的UDP接收、排队与屏幕绘制延迟；主周期耗时不是纯计算耗时，也不包含整个异步手套输出过程。输出源帧龄不是单独的发送耗时或设备响应时间。",
    },
}

# 新驱动阻抗模式的独立字段；不把估算的总PD当作实测力矩。
METRICS.update({
    'glove_driver_target_rad': {
        'name':'驱动器阻抗最终目标角','unit':'原始电机rad（含现场zero）','kind':'指令',
        'meaning':'实际提交给驱动器的绝对电机目标角，含现场零位。',
        'formula':'q_raw = zero_deg×π/180 + q_model_encoded/(sign×scale)。q_model_encoded已包含驱动保护与编码量化；新六页不转成°。',
        'coordinates':'原始J1–J12电机轴，不能与模型指节角直接等同。',
        'time_source':'glove_submit_ns，表示主机串口或G4 SDK提交，不是硬件开始运动时刻。',
        'limitations':'这是最终下发目标，不是glove_proxy_surface_rad阻挡参考。预算或方向保护可能使两者不同；也不代表实际达到的角度。'},
    'glove_driver_kp': {
        'name':'驱动器位置刚度kp','unit':'Nm/rad','kind':'指令',
        'meaning':'编码提交后的MIT位置弹簧系数，与20维从手虚拟刚度不同。',
        'formula':'kp_raw=(sign×scale)²×kp_model，再按协议量化。',
        'coordinates':'原始J1–J12电机轴。','time_source':'glove_submit_ns，SDK/串口提交时间。',
        'limitations':'kp本身不是力矩；位置误差变化会改变驱动器输出，不能据此证明实体硬力矩上限。'},
    'glove_driver_kd': {
        'name':'驱动器速度阻尼kd','unit':'Nm·s/rad','kind':'指令',
        'meaning':'编码提交后的MIT速度阻尼系数。撤离、预算不足或自由空间可能归零。',
        'formula':'kd_raw=(sign×scale)²×kd_model，再按协议量化。',
        'coordinates':'原始J1–J12电机轴。','time_source':'glove_submit_ns，SDK/串口提交时间。',
        'limitations':'阻尼贡献取决于速度误差；不是实测阻尼力。'},
    'glove_driver_pd_torque_nm': {
        'name':'发送时弹簧与阻尼力矩估算（不含前馈）','unit':'名义Nm','kind':'估算',
        'meaning':'用编码后的目标/kp/kd与反馈状态推算弹簧和阻尼部分，不含前馈。',
        'formula':'τPD=kp×(q目标编码−q反馈)+kd×(速度目标编码−dq滤波)。速度零码有量化残差，不假装精确为零。',
        'coordinates':'原始J1–J12电机轴，与同图电流×Kt估算使用同一轴方向。',
        'time_source':'glove_driver_estimate_received_ns为各轴位置反馈的主机收包时刻；速度另有glove_driver_velocity_received_ns。',
        'limitations':'位置与速度可能来自不同收包时刻；驱动内部状态随后变化会改变输出。不是力传感器实测，也不是实体硬上限。'},
})
METRICS['glove_friction_torque_nm'].update(
    name='自由空间摩擦补偿前馈',
    meaning='driver_impedance模式在有效自由空间生成的摩擦补偿前馈分量；接触/释放等待关联轴及失效时撤除。',
    formula="规划器按配置的速度死区、tanh库仑项与黏性项计算补偿，执行摩擦上限、名义预算与渐入；SDK编码后按原始电机轴发布。此字段是自由轴实际编码的前馈，接触轴的协议零码残差不算摩擦。",
    coordinates='12维原始电机J1–J12轴；力矩按sign×scale转换，和提交名义总输入同方向。',
    time_source='glove_driver_velocity_received_ns：各轴速度估计使用的真实新位置反馈主机收包时间；重复包不更新也不续期。',
    limitations='这是补偿指令分量，不是测得的摩擦；纯torque模式没有此分量。协议零码残差不算摩擦补偿。')
for key in ('glove_current_a','glove_estimated_torque_nm'):
    METRICS[key]['time_source']='glove_current_received_ns[i]：各轴电流反馈的主机收包时刻，与位置时间分别保留。新六组验证页缺该原始时间就不画；仅旧兼容曲线可回退位置/整帧时间，不能用于精确时序验收。'


def metric_details(key,frame=None):
    info=dict(METRICS[key])
    if frame and frame.get('feedback_control_mode')=='driver_impedance':
        if key=='glove_target':
            info.update(METRICS['glove_driver_pd_torque_nm'])
            info['name']='SDK提交时总PD名义估算'
            info['meaning']='已编码的弹簧、阻尼和前馈总和在提交准备时状态上的名义估算；不是独立实测力矩。'
            info['formula']='τ总名义=kp×(q目标编码−q反馈)+kd×(速度目标编码−dq滤波)+τff编码。'
            info['time_source']='横轴用glove_submit_ns（主机提交时刻）；参与推算的位置与速度时间分别保存在glove_driver_estimate_received_ns和glove_driver_velocity_received_ns。新六组验证页缺原始提交时刻就不绘制该点。'
        elif key=='glove_projected_torque_nm':
            info.update(name='阻抗规划名义估算（兼容字段）',kind='算法',
                        meaning='新模式该兼容字段与控制器请求字段相同，都是经过规划器名义预算后的结果，尚未驱动编码；不是额外的限幅前阶段。UI只画一次。',
                        formula='基于有界局部受阻代理的弹簧、阻尼及自由空间摩擦分量组合；接触关联轴不叠加摩擦。',
                        limitations='受阻代理是估计，参数中的硬度为人工设置；总目标不代表实测接触力。')
        elif key=='glove_requested_torque_nm':
            info.update(name='阻抗规划名义值（编码前）',kind='算法',
                        meaning='经过规划器总名义预算和单侧撤离约束后的力矩估算；驱动还会量化并按最新位置再次检查。',
                        formula='用规划后的target/kp/kd/tau_ff在规划输入位置和速度上计算总PD；驱动实际编码值另见SDK提交曲线。',
                        limitations='不是发送完成或实测输出；驱动量化和更新的反馈位置会改变发送时估算。')
    return info

# 控制链端点直接来自同一个异步输出请求，不用UI发布时的最新反馈冒充。
for field,name,unit,meaning in (
    ('wuji_feedback_current_a','Wuji控制链输入电流','A','实际用于产生本次手套输出的Wuji原始反馈电流，不需要Kt。'),
    ('wuji_feedback_position_rad','Wuji控制链输入角度','rad','产生本次手套输出时所使用的Wuji反馈角；新验证页直接显示rad。'),
    ('wuji_feedback_velocity_rad_s','Wuji控制链输入速度','rad/s','产生本次手套输出时所使用的Wuji同帧反馈速度；新验证页直接显示rad/s。'),
    ('wuji_feedback_error_rad','Wuji控制链目标减反馈角差','rad','本次请求中的最终Wuji目标减去请求所用反馈角，参与受阻判据和局部代理求解；新验证页不转成°。'),
):
    METRICS[field]={'name':name,'unit':unit,'kind':'实测' if field!='wuji_feedback_error_rad' else '算法',
        'meaning':meaning,'formula':'由GloveOutputWorker原请求快照携带，feedback_request_id与手套提交一致；角差项为该请求的target−hand.position。',
        'coordinates':'Wuji原生20关节，按所选手指分组，不与手套12电机逐轴等同。',
        'time_source':'wuji_feedback_received_ns：该次请求所用Wuji反馈的主机接收时刻，不是最新UI帧的wuji_received_ns。',
        'limitations':'电流不是力矩；反馈帧主机时间也不是设备内部采样时刻。缺少配对输入时保持空值，不回填更新的状态。'}
METRICS['glove_feedforward_torque_nm']={
    'name':'手套实际编码前馈输入τff','unit':'协议名义Nm','kind':'指令',
    'meaning':'交给手套驱动的实际MIT前馈编码对应数值。阻抗模式还同时提交目标角、kp、kd，因此此值不是总输出。',
    'formula':'取同一次已提交输出中的feedforward参数，按原始电机轴重排及方向换算。旧纯力矩模式等于总力矩命令。',
    'coordinates':'原始J1–J12电机轴，方向与其他手套输入曲线一致。',
    'time_source':'glove_submit_ns：本次输出最后一轴的主机串口/G4 SDK提交时刻，不是电机执行回执。',
    'limitations':'MIT模式包含协议零码量化残差；完全撤力时实际使用0A命令，图上记为零前馈。它不等同于摩擦补偿分量，也不等同于实测力矩。'}

# 六组验证页直接使用rad与各字段原时间；下列条目也覆盖显示端派生量。
METRICS['contact_active_mask'].update(
    time_source='新验证页按contact_evaluated_ns（ForceController::step返回时刻）绘制；缺失不画。旧兼容曲线曾按glove_output_source_ns绘制，两者不能当作同一种事件时间。',
    limitations='这是软件受阻锁存，不是接触传感器结果。释放迟滞时可保持1但已不参与阻抗；须与实际used mask即impedance_contact、参考有效性和保护原因分别查看。⑤状态按原事件时间阶跃，不用斜线暗示提前改变。')
METRICS['glove_estimated_torque_nm']['limitations']+=' 该旧诊断字段不用于新六页的验证力矩、Nm/rad拟合或Nm误差；这些功能只接受glove_validated_torque_nm及独立标定证据。'
METRICS.update({
    'wuji_feedback_target_rad': {
        'name':'① 同一请求的Wuji目标角','unit':'rad','kind':'显示派生',
        'meaning':'与本次手套输出绑定的Wuji目标，和同一请求实际角叠加；不是UI当前周期的最新目标。',
        'formula':'wuji_feedback_position_rad + wuji_feedback_error_rad，逐轴相加恢复该请求原target；任一项缺失则结果缺失，不插值。',
        'coordinates':'20维Wuji原生关节轴；与本次实际角同零位、同方向，新页面保留rad。',
        'time_source':'wuji_feedback_received_ns，用于标识该目标对应的反馈输入快照，不表示该目标实际提交的时间；原时刻缺失不画。',
        'limitations':'受阻后的目标与实际角差可能正是反馈依据，不能单凭角差判为故障。feedback_request_id绑定该输入与后续手套输出。'},
    'glove_raw_position_rad': {
        'name':'② 手套原始绝对电机实际角','unit':'原始电机rad（含现场zero）','kind':'显示派生',
        'meaning':'把手套原始编码器度数换成rad，用于和阻挡参考、最终下发目标对齐。',
        'formula':'glove_raw_position_deg×π/180；仅作单位换算，不减zero、不乘sign或scale、不解闭链。',
        'coordinates':'12维原始J1–J12轴，和proxy/driver target均为含现场zero的绝对电机角；不是模型指节角。',
        'time_source':'glove_motor_received_ns[i]：本轴位置原收包时刻，缺失不画；12轴不假定同时采样。',
        'limitations':'曲线实际角可能来自比规划更新的反馈；规划当时使用的位置另见glove_proxy_input_rad，不能互相替代。'},
    'glove_proxy_surface_rad': {
        'name':'② 受阻代理参考角','unit':'原始电机rad（含现场zero）','kind':'算法观测',
        'meaning':'局部逆解及既有限幅后，规划器实际保留的本轴阻挡表面；不是最终下发目标。',
        'formula':'取completed请求保存的surface，按zero_deg×π/180 + surface_model/(sign×scale)转换；glove_proxy_valid=false时发布null，UI也隐藏无效参考，不沿用之前的角度。',
        'coordinates':'12维原始J1–J12绝对电机轴。预算、撤力或编码保护可使最终目标偏离参考，二者分别记录。',
        'time_source':'曲线用impedance_planned_ns标记规划完成；每轴原始输入时刻另存glove_proxy_source_ns[i]，保留同一feedback_request_id；缺规划完成时间不画。',
        'limitations':'不是实测物体表面。有效参考仍可能伴随保护撤力；不能仅凭参考有效就认为正在输出，也不能用最终目标代替参考计算压入量。'},
    'glove_proxy_input_rad': {
        'name':'代理规划当时的实际输入角','unit':'原始电机rad（含现场zero）','kind':'输入快照',
        'meaning':'该次代理规划实际读取的手套位置，随completed快照保存，不用当前UI位置覆盖。',
        'formula':'zero_deg×π/180 + input_model/(sign×scale)，输入来自该请求GloveFrame.motor_rad。',
        'coordinates':'12维原始J1–J12绝对电机轴；和代理参考、最终目标同坐标。',
        'time_source':'glove_proxy_source_ns[i]：该请求本轴原位置收包时刻；不是规划完成或遥测发布时刻。',
        'limitations':'几何失效但已完成撤力时也可保留该输入；没有completed时为null，不能借最新帧补值。'},
    'glove_proxy_source_ns': {
        'name':'代理规划原始位置源时刻','unit':'主机单调ns','kind':'时间来源',
        'meaning':'本次规划实际使用的各轴位置原收包时刻。',
        'formula':'从原请求GloveFrame.received_ns逐轴复制，经completed与SessionTick传递，按raw ID重排。',
        'coordinates':'12项对应原始J1–J12；与glove_proxy_input_rad逐项配对。',
        'time_source':'其本身就是原始源时刻，不用published_ns、timestamp_ns或UI当前轴时间替代。',
        'limitations':'主链路栏显示本轴参考源→提交毫秒差，悬停可看本轴源与提交的原始ns；不是物理接触或执行延迟。全请求有效期使用glove_output_source_ns，即Wuji输入与最旧手套输入的较早者，不是只看这个单轴时刻。'},
    'glove_filtered_velocity_rad_s': {
        'name':'手套规划滤波速度','unit':'原始电机rad/s','kind':'算法观测',
        'meaning':'本次规划的电机速度估计器输出，供按入/回开/保持分类；保留规划值而不拿撤力命令中的零值代替。',
        'formula':'按真实新位置收包间隔差分及既有低通估计，模型速度除以sign×scale换到raw方向；重复包不更新速度或源时间。',
        'coordinates':'12维原始J1–J12，沿阻挡方向速度为glove_blocking_direction×本值。',
        'time_source':'glove_driver_velocity_received_ns[i]：估计器本次所用位置源时刻；分析仅接受不晚于提交、且距响应电流不超过20ms的速度。',
        'limitations':'不是独立速度传感器；缺失或超期时分类unknown，不猜测人手意图。默认>0.02rad/s为按入，<-0.02为回开，中间为低速保持。'},
    'glove_blocking_direction': {
        'name':'手套原始轴按入方向','unit':'-1 / 0 / 1','kind':'算法状态',
        'meaning':'规划器实际得到的受阻方向；0表示没有有效受阻方向。',
        'formula':'direction_raw=direction_model×sign(sign×scale)；沿参考超出量x=direction_raw×(actual_raw−proxy_raw)。',
        'coordinates':'12维原始J1–J12；正负依据局部逆解，不按电流符号或UI当前运动推测。',
        'time_source':'同一completed请求的impedance_planned_ns；输入与参考来源另有glove_proxy_source_ns。',
        'limitations':'分析要求参考有效且方向为±1；direction=0不是保持。按入/回开还需有效滤波速度，且只描述测得运动方向。'},
    'glove_proxy_valid': {
        'name':'② 有效受阻参考状态','unit':'0 / 1','kind':'算法状态',
        'meaning':'1表示本次规划存在本轴有效受阻参考及±1方向；自由轴、无几何或规划失败为0。',
        'formula':'从规划器真实方向和规划结果记录，free方向0必须false；无效参考角为null。',
        'coordinates':'12维原始J1–J12，不与20维Wuji受阻锁存逐轴等同。',
        'time_source':'impedance_planned_ns；新验证页缺少该原始完成时刻就不画。',
        'limitations':'保留参考但预算/预测方向等保护撤力时仍可为1；拟合还要求已提交kp>0、无释放原因且三类限幅明确false。无completed时false只表示没有有效参考，原因仍为unknown。'},
    'impedance_contact': {
        'name':'① 实际参与局部逆解的Wuji行','unit':'0 / 1','kind':'算法状态',
        'meaning':'实际进入本次局部逆解normal/rhs的Wuji行，即used mask；不是上游候选mask或contact锁存的复制。',
        'formula':'仅在该行通过实际方向检查并加入局部求解时置true；整次规划失败则清空全部行。',
        'coordinates':'20维Wuji关节顺序；行参与不保证某个手套轴一定产生非零PD。',
        'time_source':'impedance_planned_ns：规划完成时刻，缺失不画；与contact_active_mask的判据完成时刻分开。',
        'limitations':'释放迟滞期间contact_active_mask可为1而本值为0。⑤状态按原事件时间阶跃。没有completed时整字段null，已完成释放规划时为20项false，未知与已知未参与不能混淆。'},
    'glove_validated_torque_nm': {
        'name':'③ 独立标定电流力矩估计','unit':'Nm','kind':'标定估算',
        'meaning':'仅在显式独立标定声明、非空证据ID及本轴有效Kt存在时显示；默认5.1Nm/A空载参考不会进入此字段。',
        'formula':'τ=(glove_current_a−glove_bias_a)×glove_nm_per_a；未提供偏置时用0。glove_calibrated=true且glove_calibration_id非空，本轴Kt须非零有限，否则缺失。',
        'coordinates':'12维原始J1–J12；Kt包含方向与Nm/A量纲。新③/⑥页与名义Nm同图，电流A另图。',
        'time_source':'glove_current_received_ns[i]：电流独立原收包时刻；缺失不画，不借位置时间。',
        'limitations':'仍是电流标定估计，不是独立力矩传感器实测。散点整轴快照只要有真实电流样本缺完整标定，就统一退回A/A·rad⁻¹并关闭Nm误差，不混合证据等级。'},
    'validation_timing_ms': {
        'name':'⑤ 软件阶段耗时','unit':'ms','kind':'显示派生',
        'meaning':'本次请求从Wuji主机收包，经判据、规划到SDK提交的四段软件耗时。',
        'formula':'依次为(contact_evaluated_ns−wuji_feedback_received_ns)、(impedance_planned_ns−contact_evaluated_ns)、(glove_submit_ns−impedance_planned_ns)、(glove_submit_ns−wuji_feedback_received_ns)，各除10⁶；时间缺失或倒序则对应项null。',
        'coordinates':'4项依次为收包→判据、判据→规划、规划→提交、收包→提交；同一feedback_request_id。',
        'time_source':'横轴glove_submit_ns。contact_evaluated_ns与impedance_planned_ns是函数返回后的主机时刻；缺提交时刻不画。',
        'limitations':'不等于物理接触→完整触感响应。判据去抖可能跨多个请求；未知设备采样、SDK内部排队、驱动执行和机械响应不在这些软件阶段值中。'},
    'validation_stiffness_analysis': {
        'name':'④ 响应—参考偏差与刚度拟合','unit':'默认A/rad；独立标定完整时Nm/rad','kind':'选区分析（非遥测字段）',
        'meaning':'分析电机的沿阻挡方向超出参考量与有符号阻力响应关系。按入、回开、低速保持、方向未知分别着色；名义输入散点单独作为算法参照。',
        'formula':'x=direction×(实际角−代理参考)；y=−direction×实测电流，独立标定完整时改为−direction×标定力矩。带截距线性拟合至少8点且有效角跨度≥0.005rad；仅用有效参考、已提交kp>0、非负x、|速度|≤0.05rad/s的按入/保持点，排除释放原因及任一限幅或限幅未知。',
        'coordinates':'原始单电机坐标，不混合J轴；抵抗按入方向为正，保留负响应以暴露方向问题。默认保持死区为沿阻挡方向速度±0.02rad/s。',
        'time_source':'以独立电流收包为响应时钟，取不晚于它的最近真实位置且相差≤20ms；速度源不晚于提交且距电流≤20ms。命令取同会话连续序号段最近已提交请求，原glove_output_source_ns+100ms起失效；不使用未来值或插值。',
        'limitations':'20ms不是命令提交年龄上限；配对时差、命令年龄、输入年龄、速度年龄分开。回开及限幅点可供观察但不拟合；名义线不能证明实际阻力准确，两侧电流比例不是验收标准。'},
    'validation_stability_analysis': {
        'name':'⑥ 保持统计与独立力矩误差','unit':'rad/s、A；标定误差为Nm','kind':'选区分析（非遥测字段）',
        'meaning':'对冻结快照的选区计算保持角漂移、电流峰峰值/RMS/标准差，并在独立标定完整时计算标定力矩减名义输入的偏差/RMSE/峰值绝对误差。',
        'formula':'保持统计只取满足刚度拟合筛选的hold点；电流峰峰值=max−min，RMS=√mean(I²)，std为去均值RMS。角漂移对唯一真实位置点按秒做线性回归。Nm误差使用有效因果配对的标定力矩−名义输入，独立于保持/刚度样本筛选。',
        'coordinates':'选定原始J1–J12单轴；电流保留原始符号。原始位置源时刻去重，漂移不跨会话或序号断档。',
        'time_source':'与④相同的电流响应时钟和20ms位置/速度配对；选区按电流响应时刻取闭区间，允许使用选区之前仍有效的已提交命令。',
        'limitations':'未知/超期速度不按保持处理；拟合不足、角跨度不足或跨分段时保留不可用。界面显示简短摘要，悬停查看完整筛选和配对规则。冻结、分析与导出使用同一份数据，不覆盖暂停快照、不平滑或补造缺失。'},
})

for field,name,meaning,formula,stamp in (
    ('glove_torque_limited','② 名义力矩/协议预算受限','实际命中弹簧/阻尼总预算、摩擦上限、SDK增益范围、前馈或量化后预算缩放。','规划与实际SDK提交分支标志取或；不是根据波形平台反推。','glove_submit_ns'),
    ('glove_motion_limited','② 电机软限位保护','规划或提交目标碰到位置范围限制，或当前位置位于软限位且名义输出朝外而撤力。','记录实际目标截断（含编码可表示范围）或限位撤力分支。','glove_submit_ns'),
    ('glove_proxy_limited','② 代理位移/移动速率受限','局部代理偏移或参考表面移动速率被既有限制截断；位置软限位另列。','仅在proxy_offset或surface_rate实际受限时置true。','impedance_planned_ns'),
    ('glove_any_limited','三类限制合并状态','只读显示端合并力矩、软限位、代理限制。','三类标志均为明确bool时取逻辑或，否则null；分析仍分别检查三项。','glove_submit_ns'),
):
    METRICS[field]={'name':name,'unit':'0 / 1','kind':'保护状态','meaning':meaning,'formula':formula,
        'coordinates':'12维原始J1–J12电机轴，不改变控制条件或参数。',
        'time_source':stamp+'：同一completed请求的原始阶段时刻；缺少原始时刻不绘制。',
        'limitations':'限幅点可保留在散点中观察，但不进入刚度或保持统计。无completed时原生false并伴随unknown，不能把它当作已知无保护的有效控制；仍需有效参考、源时间和请求ID。'}

for field,name,meaning in (
    ('contact_evaluated_ns','判据计算完成时刻','ForceController::step返回后记录的主机时刻，包含已完成但几何无效的判据结果。'),
    ('impedance_planned_ns','代理规划完成时刻','DriverImpedancePlanner::step返回后记录的主机时刻，包含已完成的安全撤力规划。'),
):
    METRICS[field]={'name':name,'unit':'主机单调ns','kind':'软件事件','meaning':meaning,
        'formula':'在原生实际阶段完成处记录，随同一completed/request_id传递；不由UI发布时间或曲线转折估计。',
        'coordinates':'每个完成请求一个标量；和各轴原收包ns使用同一主机时钟。',
        'time_source':'字段本身为原始软件事件时刻。新验证曲线缺该时刻则不绘制相应点。',
        'limitations':'不是设备内部采样或物理接触时刻。失败且无completed时不制造轴级原因；现有终止事件另行提示。'}

METRICS['glove_control_reason']={
    'name':'② 实际控制分支与保护原因','unit':'原因字符串（可用|连接）','kind':'分支观测',
    'meaning':'本次规划与提交实际执行分支的并集，共25种已定义原因；unknown表示没有可用记录。固定大小位图由遥测线程转字符串，原因不会参与控制。',
    'formula':'inactive_geometry=上游几何未激活；invalid_input=位置或源时间无效；invalid_geometry=矩阵非有限；unobservable_geometry=Jacobian总范数不足；singular_transmission=传动矩阵不可逆/条件不足；invalid_contact_row=候选行无效导致整次释放；inverse_failed=逆解结果非有限；contact_pd=规划形成PD；free_axis=无受阻且无接触耦合；coupled_release=无逆解方向但仍耦合锁存接触，含局部相互抵消；friction_disabled=自由摩擦配置关闭；outside_proxy=未超过参考；zero_stiffness=刚度为零；planner_release_guard=作用距离/当前或预测输出触发撤力；proxy_offset_limited=代理偏移限制；proxy_rate_limited=参考速率限制；soft_limit_target=目标限位；soft_limit_release=软限位撤力；torque_budget_limited=名义总预算限制；friction_limited=自由摩擦上限；sdk_gain_limited=MIT增益范围；sdk_zero_output=精确零电流分支；sdk_direction_release=量化后当前/预测方向保护；sdk_budget_release=预算缩放尝试耗尽；sdk_output_accepted=量化MIT命令获接纳。',
    'coordinates':'12项对应原始J1–J12，同轴可同时有规划contact_pd及后续SDK撤力原因；原始原因字符串保留。',
    'time_source':'同一completed请求，规划事件看impedance_planned_ns，提交事件看glove_submit_ns；原始输入时间另有逐轴字段。',
    'limitations':'不能从UI角差猜原因；coupled_release和planner_release_guard不证明人手主动回开。无completed时unknown，不伪造轴级故障。分析排除包含release/retract/withdrawal的原因，且另查kp、三类限幅、有效参考与速度。'}
