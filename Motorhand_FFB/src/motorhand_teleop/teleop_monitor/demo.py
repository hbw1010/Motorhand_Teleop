"""离线演示：合成角度、电流和反馈；无真实设备输出。"""
import math
import numpy as np
from ..ve33_kinematics import Ve33Kinematics
from ..real_hand.force_curves import make_demo_sample
from .scene import GLOVE


class DemoSource:
    def __init__(self):
        self.sequence=0
        model=Ve33Kinematics('right',GLOVE)
        self.poses=[]
        for phase in np.linspace(0,1,25):
            angles={name:float(phase)*(.13 if 'mcp' in name or 'cmc' in name else .32) for name in model.joint_names}
            angles['thumb_cmc_abd_joint']=0.
            pose=model.solve(angles)
            self.poses.append(([pose.all_positions[n] for n in model.tree_names],
                               [pose.joint_positions[n] for n in model.joint_names],pose.max_gap_m))
    def sample(self,t,now):
        p=validation_sample(t,now);self.sequence+=1
        phase=(1-math.cos(t*.8))/2
        index=min(24,int(phase*24));tree,joints,gap=self.poses[index]
        p.update(sequence=self.sequence,hand='right',
            glove_tree_order='ve33_closure_independent_dependent',glove_tree_rad=tree,
            glove_position_rad=joints,glove_closure_gap_m=gap)
        return p


def validation_sample(t,now):
    """18秒合成实验，仅用于验收显示/时间配对，不代表真机物理响应。"""
    p=make_demo_sample(t,now)
    stage=t%18
    contact=2<=stage<15
    if stage<2:delta=-.03*(2-stage);speed=.03
    elif stage<7:delta=.03*(stage-2);speed=.03
    elif stage<10:delta=.15;speed=0.
    elif stage<15:delta=.15-.03*(stage-10);speed=-.03
    else:delta=-.02*(stage-15);speed=-.02
    # 明确构造原始电机轴，包含正反向与不同零偏；不读取现场标定文件。
    directions=[-1,1,-1,1,-1,1,1,1,-1,1,-1,1]
    proxy=[.1+i*.05 for i in range(12)]
    actual=[v+d*delta for v,d in zip(proxy,directions)]
    velocity=[d*speed for d in directions]
    nominal=[];targets=[];kp=[];kd=[];limited=[];reasons=[];current=[]
    for i,d in enumerate(directions):
        k=21.5*([1.8,1.2,1.2,1.2]+[1.]*8)[i] if contact and speed>=0 else 0.
        damping=.008*k
        requested=-d*(k*max(delta,0)+damping*speed)
        cap=1.7 if i==1 else 3.4
        torque=max(-cap,min(cap,requested))
        is_limited=abs(requested)>cap
        nominal.append(torque);kp.append(k);kd.append(damping);limited.append(is_limited)
        targets.append(actual[i]+(torque+damping*velocity[i])/k if k else actual[i])
        reason='contact_pd' if k else 'planner_release_guard' if contact else 'free_axis'
        reasons.append(reason+('|torque_budget_limited' if is_limited else '')+'|sdk_output_accepted')
        # 显式合成带微小波动的响应，绝不标作独立力矩标定。
        response=abs(torque)/5.+(.025*max(delta,0) if contact and speed<0 else 0.)
        current.append(-d*(response+.001*math.sin(t*13+i)))
    measured=[(.3 if contact else .3+delta) if i%4!=1 else 0. for i in range(20)]
    errors=[(.08+max(delta,0)) if contact and i%4!=1 else 0. for i in range(20)]
    target=[a+b for a,b in zip(measured,errors)]
    mask=[contact and i%4!=1 for i in range(20)]
    wuji_current=[.2+max(delta,0) if mask[i] else .02 for i in range(20)]
    wuji_velocity=[0. if contact or i%4==1 else speed for i in range(20)]
    p.update(session_id='monitor-offline-demo',feedback_control_mode='driver_impedance',
        glove_target_semantics='nominal_total_pd_at_submission',feedback_state='active',
        contact_state='contact' if contact and speed>=0 else 'releasing' if contact else 'free',
        glove_raw_position_deg=[math.degrees(v) for v in actual],glove_current_a=current,
        glove_target=nominal,glove_driver_target_rad=targets,glove_driver_kp=kp,glove_driver_kd=kd,
        glove_driver_pd_torque_nm=nominal.copy(),glove_friction_torque_nm=[0.]*12,
        glove_feedforward_torque_nm=[0.]*12,glove_projected_torque_nm=nominal.copy(),
        glove_requested_torque_nm=nominal.copy(),glove_proxy_surface_rad=proxy if contact else [None]*12,
        glove_proxy_input_rad=actual.copy(),glove_filtered_velocity_rad_s=velocity,
        glove_blocking_direction=directions if contact else [0]*12,glove_proxy_valid=[contact]*12,
        glove_torque_limited=limited,glove_motion_limited=[False]*12,glove_proxy_limited=[False]*12,
        glove_control_reason=reasons,impedance_contact=mask,contact_active_mask=mask,
        wuji_position_rad=measured,wuji_target_rad=target,wuji_tracking_error_rad=errors,
        wuji_current_a=wuji_current,wuji_velocity_rad_s=wuji_velocity,
        wuji_feedback_position_rad=measured.copy(),wuji_feedback_error_rad=errors.copy(),
        wuji_feedback_current_a=wuji_current.copy(),wuji_feedback_velocity_rad_s=wuji_velocity.copy(),
        wuji_filtered_current_a=[abs(v) for v in wuji_current],feedback_hardness=.3,
        feedback_request_id=p['sequence']+1,feedback_hardness_source='manual_setting',
        wuji_feedback_received_ns=now-4_000_000,wuji_received_ns=now-4_000_000,
        contact_evaluated_ns=now-3_800_000,impedance_planned_ns=now-3_500_000,
        glove_submit_ns=now-3_000_000,wuji_submit_ns=now-3_000_000,
        glove_proxy_source_ns=[now-4_000_000]*12,glove_output_source_ns=now-4_000_000,
        glove_driver_velocity_received_ns=[now-4_000_000]*12,
        glove_driver_estimate_received_ns=[now-4_000_000]*12,
        glove_motor_received_ns=[now-2_000_000]*12,glove_received_ns=now-2_000_000,
        glove_current_received_ns=[now-1_000_000]*12,
        cycle_started_ns=now-4_000_000,cycle_completed_ns=now-2_500_000,published_ns=now)
    return p
