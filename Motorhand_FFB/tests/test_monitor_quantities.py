import pytest
from motorhand_teleop.teleop_monitor.quantities import TorqueConversion, joint_label, enrich


def test_conversion_preserves_units_missing_values_and_sources():
    cfg=TorqueConversion([5.1]*12,[None]*20,'手套参考','Wuji未标定')
    p=enrich({'glove_current_a':[.2,None]+[0.]*10,'wuji_current_a':[1.]*20},cfg)
    assert p['glove_estimated_torque_nm'][0]==pytest.approx(1.02)
    assert p['glove_estimated_torque_nm'][1] is None
    assert p['wuji_estimated_torque_nm']==[None]*20
    assert p['torque_conversion']['glove_source']=='手套参考'


def test_known_wuji_signed_coefficient_and_missing_channels():
    p=enrich({'glove_current_a':None,'wuji_current_a':[.5]*20},TorqueConversion([None]*12,[-.2]*20,'未标定','用户参考'))
    assert p['wuji_estimated_torque_nm']==pytest.approx([-.1]*20)
    assert p['glove_estimated_torque_nm']==[None]*12


def test_joint_labels_are_readable_and_distinguish_thumb():
    assert joint_label('wuji',4)=='食指·指根屈伸'
    assert joint_label('wuji',5)=='食指·指根侧摆'
    assert joint_label('wuji',7)=='食指·末节屈伸'
    assert joint_label('wuji',1)=='拇指·掌根外展'
    assert 'J2' in joint_label('glove',1) and '外展' in joint_label('glove',1)


def test_reject_invalid_conversion_coefficients():
    for value in (True,0,float('nan'),10**400):
        with pytest.raises(ValueError):TorqueConversion([value]*12,[None]*20,'x','y')


def test_wuji_load_is_compared_in_glove_coordinates_without_feedback_gain():
    matrix=[0.]*240
    matrix[0]=2.;matrix[20+1]=-3.
    cfg=TorqueConversion([5.1]*12,[.2]*20,'参考','参考')
    p=enrich({'glove_current_a':[0.]*12,'wuji_current_a':[.5]*20,
              'wuji_to_glove_torque_matrix':matrix,'wuji_projection_current_a':[.5]*20,
              'wuji_projection_received_ns':80},cfg)
    assert p['wuji_reflected_torque_nm'][0]==pytest.approx(-.2)
    assert p['wuji_reflected_torque_nm'][1]==pytest.approx(.3)
    cfg.wuji=[None]*20
    assert enrich(p,cfg)['wuji_reflected_torque_nm'][:2]==[None]*2


def test_plot_comparison_uses_actual_signal_timestamps_and_readable_names():
    from motorhand_teleop.teleop_monitor.plots import build_pages
    spec=build_pages(1)['③ 输出响应'][0]
    traces={t.field:t for t in spec.traces}
    assert '食指' in spec.title and 'J5' in spec.title
    p={'timestamp_ns':100,'glove_submit_ns':95,'glove_current_received_ns':[92]*12}
    assert traces['glove_target'].stamp(p)==95
    assert traces['glove_validated_torque_nm'].stamp(p)==92
    assert traces['glove_validated_torque_nm'].value(p) is None


def test_other_pages_keep_async_output_and_measurement_times():
    from motorhand_teleop.teleop_monitor.plots import build_pages
    traces=[t for specs in build_pages(1).values() for spec in specs for t in (*spec.traces,*spec.states)]
    stamps={t.field:t.time_field for t in traces}
    assert stamps['glove_proxy_surface_rad']=='impedance_planned_ns'
    assert stamps['contact_active_mask']=='contact_evaluated_ns'
    assert stamps['impedance_contact']=='impedance_planned_ns'
    assert stamps['glove_current_a']=='glove_current_received_ns'
    assert stamps['wuji_feedback_current_a']=='wuji_feedback_received_ns'
    assert stamps['validation_timing_ms']=='glove_submit_ns'
    assert all(t.strict_time for t in traces)


def test_projection_uses_paired_current_and_only_requires_nonzero_dependencies():
    matrix=[0.]*240;matrix[0]=1.;matrix[21]=1.
    cfg=TorqueConversion([5.1]*12,[.2]+[None]*19,'参考','部分标定')
    p={'wuji_current_a':[10.]*20,'wuji_projection_current_a':[1.]*20,
       'wuji_projection_received_ns':80,'wuji_to_glove_torque_matrix':matrix}
    out=enrich(p,cfg)
    assert out['wuji_reflected_torque_nm'][0]==pytest.approx(-.2)
    assert out['wuji_reflected_torque_nm'][1] is None
    assert out['wuji_reflected_torque_nm'][2]==0
    p.pop('wuji_projection_current_a')
    assert enrich(p,cfg)['wuji_reflected_torque_nm']==[None]*12


def test_friction_payload_rejected_and_trace_defensive():
    import json
    from motorhand_teleop.real_hand.force_curves import make_demo_sample
    from motorhand_teleop.teleop_monitor.state import decode_packet
    from motorhand_teleop.teleop_monitor.plots import Trace
    p=make_demo_sample(1,100);p['glove_friction_torque_nm']={'x':1}
    with pytest.raises(ValueError):decode_packet(json.dumps(p).encode())
    assert Trace('glove_friction_torque_nm',0,'摩擦','#fff').value(p) is None


def test_axis_range_does_not_overflow_for_valid_extreme_numbers():
    import math
    from motorhand_teleop.teleop_monitor.plots import axis_range
    low,high,scale=axis_range([-1.7e308,1.7e308])
    assert all(math.isfinite(x) for x in (low,high,scale))
    assert low < -1 and high > 1
    assert math.isfinite((1.7e308/scale-low)/(high-low))
