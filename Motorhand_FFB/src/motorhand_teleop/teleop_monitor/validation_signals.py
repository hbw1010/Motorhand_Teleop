"""六组验证视图的显示字段；不修改遥测原值、不参与控制。"""
import math
from .quantities import finite

REASONS={
 'unknown':'未提供','inactive_geometry':'几何未就绪','invalid_input':'输入无效',
 'invalid_geometry':'几何无效','unobservable_geometry':'不可观测','singular_transmission':'传动奇异',
 'invalid_contact_row':'受阻行无效','inverse_failed':'局部逆解无效','contact_pd':'接触阻抗',
 'free_axis':'自由轴','coupled_release':'耦合/释放抑制','friction_disabled':'摩擦补偿关闭',
 'outside_proxy':'已退出参考','zero_stiffness':'零刚度','planner_release_guard':'规划撤离保护',
 'proxy_offset_limited':'代理偏移受限','proxy_rate_limited':'代理移动速率受限',
 'soft_limit_target':'目标软限位','soft_limit_release':'软限位撤力',
 'torque_budget_limited':'名义力矩预算限制','friction_limited':'摩擦预算限制',
 'sdk_gain_limited':'协议增益上限','sdk_zero_output':'零电流撤力',
 'sdk_direction_release':'提交方向保护撤力','sdk_budget_release':'提交预算撤力','sdk_output_accepted':'提交检查通过',
}

def reason_text(value):
    if not isinstance(value,str):return '未提供'
    return ' / '.join(REASONS.get(part,part) for part in value.split('|'))


def validate_extensions(p):
    for key in ('glove_proxy_surface_rad','glove_proxy_input_rad','glove_filtered_velocity_rad_s'):
        values=p.get(key)
        if values is not None and (not isinstance(values,list) or len(values)!=12 or
                any(v is not None and not finite(v) for v in values)):raise ValueError(f'{key}必须为12维有限量或缺失值')
    for key,n in (('glove_proxy_valid',12),('glove_torque_limited',12),('glove_motion_limited',12),
                  ('glove_proxy_limited',12),('impedance_contact',20)):
        values=p.get(key)
        if values is not None and (not isinstance(values,list) or len(values)!=n or
                any(v is not None and type(v) is not bool for v in values)):raise ValueError(f'{key}布尔状态无效')
    values=p.get('glove_blocking_direction')
    if values is not None and (not isinstance(values,list) or len(values)!=12 or
            any(v is not None and (type(v) is not int or v not in (-1,0,1)) for v in values)):
        raise ValueError('阻挡方向无效')
    values=p.get('glove_control_reason')
    if values is not None and (not isinstance(values,list) or len(values)!=12 or
            any(not isinstance(v,str) or not 0<len(v)<=1024 for v in values)):
        raise ValueError('保护原因无效')


def prepare_signals(packet):
    p=dict(packet)
    def get(key,n):
        value=p.get(key)
        return value if isinstance(value,(list,tuple)) and len(value)==n else [None]*n
    p['glove_raw_position_rad']=[math.radians(v) if finite(v) else None for v in get('glove_raw_position_deg',12)]
    p['wuji_feedback_target_rad']=[a+b if finite(a) and finite(b) and finite(a+b) else None
        for a,b in zip(get('wuji_feedback_position_rad',20),get('wuji_feedback_error_rad',20))]
    groups=[get(key,12) for key in ('glove_torque_limited','glove_motion_limited','glove_proxy_limited')]
    p['glove_any_limited']=[any(v[i] for v in groups) if all(type(v[i]) is bool for v in groups) else None for i in range(12)]
    def elapsed(a,b):
        x,y=p.get(a),p.get(b)
        return (x-y)/1e6 if type(x) is int and type(y) is int and 0<y<=x else None
    p['validation_timing_ms']=[elapsed('contact_evaluated_ns','wuji_feedback_received_ns'),
        elapsed('impedance_planned_ns','contact_evaluated_ns'),elapsed('glove_submit_ns','impedance_planned_ns'),
        elapsed('glove_submit_ns','wuji_feedback_received_ns')]
    return p
