"""各页曲线定义：每条线的量纲、通道和时间来源明确。"""
from dataclasses import dataclass
import math
from .quantities import joint_label,finite

STATE_FIELDS=('contact_active_mask','impedance_contact','glove_proxy_valid','glove_torque_limited','glove_motion_limited','glove_proxy_limited','glove_any_limited')

COLORS=('#54d6cf','#f4b45a','#ad98ff','#f18b9b','#73b9f7','#b8d778')
GLOVE_GROUPS=(range(4),range(4,6),range(6,8),range(8,10),range(10,12),range(12))


def axis_range(values):
    import numpy as np
    # 数值缓存直接走批量计算；Python输入仍按原来的有限数规则验证。
    if isinstance(values,np.ndarray) and values.dtype.kind in 'fiu':
        data=values.astype(np.float64,copy=False);data=data[np.isfinite(data)]
    else:data=np.asarray([v for v in values if finite(v)],dtype=float)
    scale=float(np.max(np.abs(data),initial=0)) or 1.
    normalized=data/scale
    low=float(np.min(normalized,initial=0));high=float(np.max(normalized,initial=0))
    margin=(high-low)*.12 or .1
    return low-margin,high+margin,scale


@dataclass(frozen=True)
class Trace:
    field:str
    index:int
    label:str
    color:str
    factor:float=1.
    style:int=1
    time_field:str='timestamp_ns'
    strict_time:bool=False

    def display_label(self,frame):
        if self.field=='glove_target' and self.label.startswith('J'):return self.label
        if frame and frame.get('feedback_control_mode')=='driver_impedance':
            return {'glove_projected_torque_nm':'阻抗规划估算（兼容字段）',
                    'glove_requested_torque_nm':'阻抗规划（编码前）',
                    'glove_target':'发送时总PD名义估算'}.get(self.field,self.label)
        return self.label

    def value(self,frame):
        if self.strict_time:
            stamp=frame.get(self.time_field)
            if isinstance(stamp,list):stamp=stamp[self.index] if self.index<len(stamp) else None
            if type(stamp) is not int or stamp<=0:return None
        if self.field=='glove_proxy_surface_rad':
            valid=frame.get('glove_proxy_valid')
            if not isinstance(valid,(list,tuple)) or self.index>=len(valid) or valid[self.index] is not True:return None
        values=frame.get(self.field)
        if not isinstance(values,(list,tuple)) or self.index>=len(values):return None
        v=values[self.index]
        if self.field in STATE_FIELDS and isinstance(v,bool):return float(v)
        if self.field=='glove_target' and frame.get('glove_target_unit')!='Nm':return None
        return v*self.factor if finite(v) else None

    def stamp(self,frame):
        t=frame.get(self.time_field)
        if t is None and self.time_field=='glove_current_received_ns':
            t=frame.get('glove_motor_received_ns')
        if isinstance(t,list):t=t[self.index] if self.index<len(t) else None
        return t if type(t) is int and t>0 else frame['timestamp_ns']


@dataclass(frozen=True)
class PlotSpec:
    title:str
    description:str
    traces:tuple
    states:tuple=()
    kind:str='time'
    variant:str='response'


def channels(field,indices,hand,factor=1.):
    time_field={
        'glove_current_a':'glove_current_received_ns', 'glove_raw_position_deg':'glove_motor_received_ns',
        'glove_position_rad':'glove_received_ns', 'wuji_current_a':'wuji_received_ns',
        'wuji_position_rad':'wuji_received_ns', 'wuji_velocity_rad_s':'wuji_received_ns',
        'wuji_estimated_torque_nm':'wuji_received_ns', 'wuji_tracking_error_rad':'wuji_submit_ns',
        'virtual_hand_torque_nm':'glove_output_source_ns', 'contact_active_mask':'glove_output_source_ns',
        'wuji_filtered_current_a':'glove_output_source_ns',
        'glove_driver_pd_torque_nm':'glove_driver_estimate_received_ns',
        'glove_driver_target_rad':'glove_submit_ns','glove_driver_kp':'glove_submit_ns','glove_driver_kd':'glove_submit_ns',
        'wuji_feedback_current_a':'wuji_feedback_received_ns','wuji_feedback_position_rad':'wuji_feedback_received_ns',
        'wuji_feedback_velocity_rad_s':'wuji_feedback_received_ns','wuji_feedback_error_rad':'wuji_feedback_received_ns',
    }.get(field,'timestamp_ns')
    return tuple(Trace(field,i,joint_label(hand,i),COLORS[j%len(COLORS)],factor,time_field=time_field) for j,i in enumerate(indices))


def build_pages(finger,control_mode='driver_impedance',selected_axis=None):
    glove=list(GLOVE_GROUPS[finger]);wuji=list(range(20) if finger==5 else range(finger*4,finger*4+4))
    axis=selected_axis if type(selected_axis) is int and selected_axis in glove else glove[0]
    def tr(field,i,label,color=0,stamp='timestamp_ns',style=1):
        return Trace(field,i,label,COLORS[color%len(COLORS)],style=style,time_field=stamp,strict_time=True)
    def group(field,indices,hand,stamp):
        return tuple(tr(field,i,joint_label(hand,i),j,stamp) for j,i in enumerate(indices))
    obstruct=[]
    for i in wuji:
        states=(tr('contact_active_mask',i,'受阻锁存',1,'contact_evaluated_ns'),
                tr('impedance_contact',i,'参与求解',0,'impedance_planned_ns'))
        obstruct.append(PlotSpec(joint_label('wuji',i)+' · rad',
            '目标与实际角叠加受阻状态。受阻后的角差是正常反馈依据，不能直接判为控制失败。',
            (tr('wuji_feedback_target_rad',i,'本次目标',1,'wuji_feedback_received_ns',2),
             tr('wuji_feedback_position_rad',i,'本次实际角',0,'wuji_feedback_received_ns')),states))
    obstruct += [PlotSpec('Wuji速度 · rad/s','与同次请求角度、电流对齐，观察真正受阻时是否停滞。',group('wuji_feedback_velocity_rad_s',wuji,'wuji','wuji_feedback_received_ns')),
                 PlotSpec('Wuji电流 · A','电流是受阻判据之一，不把它称为已标定接触力矩。',group('wuji_feedback_current_a',wuji,'wuji','wuji_feedback_received_ns'))]
    blocking=[];output=[]
    for i in glove:
        states=(tr('glove_proxy_valid',i,'参考有效',0,'impedance_planned_ns'),
                tr('glove_torque_limited',i,'预算限幅',1,'glove_submit_ns'),
                tr('glove_motion_limited',i,'软限位',3,'glove_submit_ns'),
                tr('glove_proxy_limited',i,'参考限幅',2,'impedance_planned_ns'))
        blocking.append(PlotSpec(joint_label('glove',i)+' · 原始电机rad',
            '实际角、求解参考、最终目标均含现场零位。预算可能移动最终目标，因此参考与目标分别记录。',(
             tr('glove_raw_position_rad',i,'电机实际角',0,'glove_motor_received_ns'),
             tr('glove_proxy_surface_rad',i,'阻挡参考角',2,'impedance_planned_ns',2),
             tr('glove_driver_target_rad',i,'最终下发目标',1,'glove_submit_ns')),states))
        output.append(PlotSpec(joint_label('glove',i)+' · Nm',
            '名义总输入与有独立标定依据的电流力矩估计对比；默认空载参考不进入准确性评价。',(
             tr('glove_target',i,'提交名义总输入',1,'glove_submit_ns'),
             tr('glove_validated_torque_nm',i,'独立标定电流估计',3,'glove_current_received_ns'))))
        output.append(PlotSpec(joint_label('glove',i)+' · A',
            '独立回读电流，用于观察方向、响应和波动；A与Nm不混在同一纵轴。',(
             tr('glove_current_a',i,'实测电流',0,'glove_current_received_ns'),)))
    stiffness=[PlotSpec(joint_label('glove',axis)+' · 响应—参考偏差',
        '按入/回开分色；未独立标定时为A/rad。只在低速、有效受阻、未限幅区间拟合。橙圈为限幅点。',(),kind='scatter'),
        PlotSpec('名义输入—参考偏差（理论参照）','观察预算平台；这条关系由控制算法产生，不能证明实际阻力准确。',(),kind='scatter',variant='nominal')]
    timing=[PlotSpec('Wuji受阻状态与求解状态 · 0/1',
        '这些是算法状态时间，不是物理接触时刻；释放等待期间锁存可能仍为1。',
        tuple(tr(field,i,joint_label('wuji',i)+suffix,j,stamp,style) for j,i in enumerate(wuji)
              for field,suffix,stamp,style in (('contact_active_mask','·锁存','contact_evaluated_ns',2),
                                             ('impedance_contact','·求解','impedance_planned_ns',1)))),
        PlotSpec('软件阶段耗时 · ms','仅统计原始反馈收包→判据完成→规划完成→SDK提交；不是机械执行或完整触觉响应时间。',
        tuple(tr('validation_timing_ms',i,label,i,'glove_submit_ns') for i,label in enumerate(('收包→判据','判据→规划','规划→提交','收包→提交')))),
        PlotSpec(joint_label('glove',axis)+' · 命令与响应时序','名义Nm按提交时间绘制；结合下一图电流的独立收包时间，用A/B观察建立与撤除。',(
            tr('glove_target',axis,'总名义输入',1,'glove_submit_ns'),)),
        PlotSpec(joint_label('glove',axis)+' · 响应电流A','采样/通讯造成的时间差不能直接当成电机内部延迟；缺失值保持空白。',(
            tr('glove_current_a',axis,'实测电流',0,'glove_current_received_ns'),))]
    stability=[PlotSpec(joint_label('glove',axis)+' · 保持角度rad','冻结并框选保持区间，比较实际角和阻挡参考是否持续漂移。',(
            tr('glove_raw_position_rad',axis,'实际角',0,'glove_motor_received_ns'),
            tr('glove_proxy_surface_rad',axis,'参考角',2,'impedance_planned_ns'),
            tr('glove_driver_target_rad',axis,'下发目标',1,'glove_submit_ns'))),
        PlotSpec('保持时名义输入 · Nm','只代表提交参数对应的名义量，不用同公式重算来宣称执行准确。',(
            tr('glove_target',axis,'名义输入',1,'glove_submit_ns'),
            tr('glove_validated_torque_nm',axis,'标定电流估计',3,'glove_current_received_ns'))),
        PlotSpec('保持时电流 · A','观察抖动和撤力后的残留；统计只使用匹配有效的低速保持样本。',(
            tr('glove_current_a',axis,'实测电流',0,'glove_current_received_ns'),)),
        PlotSpec('按入与回开回线','按滤波速度相对阻挡方向分类，保留有符号响应以揭示错误方向；灰点为低速保持。',(),kind='scatter')]
    return {'① Wuji受阻':obstruct,'② 手套阻挡':blocking,'③ 输出响应':output,
            '④ 刚度关系':stiffness,'⑤ 接触/释放时序':timing,'⑥ 保持/往返':stability}
