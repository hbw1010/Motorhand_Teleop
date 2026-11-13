import json
import math
import pytest
from motorhand_teleop.real_hand.force_curves import make_demo_sample
from motorhand_teleop.teleop_monitor.plots import build_pages
from motorhand_teleop.teleop_monitor.validation_signals import prepare_signals
from motorhand_teleop.teleop_monitor.state import decode_packet


def test_six_validation_groups_have_unambiguous_units_and_sources():
    pages=build_pages(1,'driver_impedance')
    assert list(pages)==['① Wuji受阻','② 手套阻挡','③ 输出响应','④ 刚度关系','⑤ 接触/释放时序','⑥ 保持/往返']
    assert any(s.kind=='scatter' for s in pages['④ 刚度关系'])
    assert any(s.states for s in pages['① Wuji受阻'])
    response={t.field for spec in pages['③ 输出响应'] for t in spec.traces}
    assert 'glove_current_a' in response and 'glove_target' in response
    assert 'glove_estimated_torque_nm' not in response
    assert 'glove_validated_torque_nm' in response


def test_angle_reconstruction_is_explicit_not_joint_motor_mix():
    p=prepare_signals({'glove_raw_position_deg':[180.]*12,'wuji_feedback_position_rad':[.2]*20,
                       'wuji_feedback_error_rad':[.1]*20})
    assert p['glove_raw_position_rad']==pytest.approx([math.pi]*12)
    assert p['wuji_feedback_target_rad']==pytest.approx([.3]*20)
    assert prepare_signals({})['glove_raw_position_rad']==[None]*12


def test_invalid_extended_observations_rejected():
    p=make_demo_sample(1,1_000_000_000)
    for field,value in (('glove_proxy_surface_rad',[0.]*11),('impedance_contact',[1]*20),
                        ('glove_blocking_direction',[2]*12),('glove_control_reason',{'bad':1})):
        with pytest.raises(ValueError):decode_packet(json.dumps({**p,field:value}).encode())


def test_full_demo_cycle_has_causal_samples_and_no_fake_torque_calibration():
    from motorhand_teleop.teleop_monitor.demo import validation_sample
    from motorhand_teleop.teleop_monitor.state import MonitorState
    from motorhand_teleop.teleop_monitor.validation_analysis import analyze_axis
    state=MonitorState()
    for i in range(900):
        now=60_000_000_000+i*20_000_000
        packet=decode_packet(json.dumps(validation_sample(i*.02,now)).encode())
        state.ingest(packet,now)
    result=analyze_axis(state.frames,4)
    assert {point['phase'] for point in result['points']}=={'press','hold','release'}
    assert result['response_unit']=='A' and result['slope_unit']=='A/rad'
    assert result['fit']['slope']==pytest.approx(4.3,abs=.02)
    assert result['stability']['count']>100
    assert result['torque_error'] is None
    assert all(0<=point['alignment_ms']<=20 for point in result['points'])
    limited=analyze_axis(state.frames,1)
    assert limited['counts']['limited']>0
    assert all(not point['fit_eligible'] for point in limited['points'] if point['limited'])
    assert all(v is None for v in state.latest['glove_validated_torque_nm'])
