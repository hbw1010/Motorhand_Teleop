"""用独立数值和异步时间验证证据等级、因果匹配与选区统计。"""
from copy import deepcopy
import math

import pytest

from motorhand_teleop.teleop_monitor.quantities import TorqueConversion, enrich


MS = 1_000_000


def conversion(calibrated=False, constants=None, bias=None):
    return TorqueConversion(
        constants if constants is not None else [2.] * 12, [None] * 20,
        '独立台架标定' if calibrated else '空载参考', '未提供',
        glove_calibrated=calibrated,
        glove_calibration_id='bench-2026-11-13' if calibrated else '',
        glove_bias_a=bias,
    )


def frame(sequence, *, angle=.1, current=-.2, direction=1, velocity=.03,
          calibrated=False, session='session-a'):
    now = (1000 + sequence * 10) * MS
    packet = dict(
        session_id=session, sequence=sequence, timestamp_ns=now,
        published_ns=now, feedback_request_id=sequence + 1,
        glove_submit_ns=now - 3 * MS,
        glove_output_source_ns=now - 5 * MS,
        glove_proxy_source_ns=[now - 5 * MS] * 12,
        glove_proxy_surface_rad=[0.] * 12,
        glove_driver_target_rad=[-.01] * 12,
        glove_driver_kp=[20.] * 12,
        glove_proxy_input_rad=[angle] * 12,
        glove_proxy_valid=[True] * 12,
        glove_blocking_direction=[direction] * 12,
        glove_raw_position_deg=[math.degrees(angle)] * 12,
        glove_motor_received_ns=[now - 2 * MS] * 12,
        glove_current_a=[current] * 12,
        glove_current_received_ns=[now - MS] * 12,
        glove_filtered_velocity_rad_s=[velocity] * 12,
        glove_driver_velocity_received_ns=[now - 4 * MS] * 12,
        glove_torque_limited=[False] * 12,
        glove_motion_limited=[False] * 12,
        glove_proxy_limited=[False] * 12,
        glove_control_reason=['blocked'] * 12,
        impedance_contact=[True] * 20,
        glove_target=[current * 2] * 12,
        glove_target_unit='Nm',
        glove_target_semantics='nominal_total_pd_at_submission',
    )
    return enrich(packet, conversion(calibrated))


def analyze(frames, **kwargs):
    from motorhand_teleop.teleop_monitor.validation_analysis import analyze_axis
    return analyze_axis(frames, 0, **kwargs)


def test_default_reference_never_becomes_validation_torque():
    cfg = TorqueConversion([5.1] * 12, [None] * 20, '空载参考', '未提供')
    result = enrich({'glove_current_a': [.2] * 12}, cfg)
    assert result['glove_estimated_torque_nm'][0] == pytest.approx(1.02)
    assert result['glove_validated_torque_nm'] == [None] * 12
    assert result['torque_conversion']['glove_calibrated'] is False
    assert TorqueConversion.load().glove_calibrated is False


def test_independent_calibration_applies_bias_and_retains_evidence():
    cfg = conversion(True, [2., None] + [-3.] * 10, [.1] * 12)
    packet = {'glove_current_a': [.3] * 12}
    result = enrich(packet, cfg)
    assert result['glove_validated_torque_nm'][:3] == [pytest.approx(.4), None, pytest.approx(-.6)]
    assert result['torque_conversion']['glove_calibration_id'] == 'bench-2026-11-13'
    assert result['torque_conversion']['glove_bias_a'] == [.1] * 12
    assert 'glove_validated_torque_nm' not in packet


@pytest.mark.parametrize('kwargs', [
    {'glove_calibration_id': ''}, {'glove_calibration_id': '   '},
    {'glove_calibration_id': True}, {'glove_calibrated': 1},
    {'glove_bias_a': [0.] * 11}, {'glove_bias_a': [float('nan')] * 12},
])
def test_calibration_rejects_missing_evidence_or_invalid_bias(kwargs):
    options = dict(glove_calibrated=True, glove_calibration_id='bench-id')
    options.update(kwargs)
    with pytest.raises(ValueError):
        TorqueConversion([2.] * 12, [None] * 20, '台架', '未提供', **options)


def test_calibrated_flag_requires_at_least_one_coefficient():
    with pytest.raises(ValueError):
        conversion(True, [None] * 12)


@pytest.mark.parametrize(('direction', 'angle', 'current', 'velocity'), [
    (1, .03, -.12, .03), (-1, -.03, .12, -.03),
])
def test_signed_axis_retains_resistance_and_physical_motion_phase(direction, angle, current, velocity):
    result = analyze([frame(0, angle=angle, current=current, direction=direction, velocity=velocity)])
    point = result['points'][0]
    assert point['angle_excess_rad'] == pytest.approx(.03)
    assert point['response'] == pytest.approx(.12)
    assert point['phase'] == 'press'
    assert point['fit_eligible'] is True
    assert point['alignment_ms'] == 1.
    assert result['response_unit'] == 'A'
    assert result['slope_unit'] == 'A/rad'
    assert result['torque_error'] is None


def test_reverse_force_is_not_hidden_by_absolute_value():
    result = analyze([frame(0, current=.4)])
    assert result['points'][0]['response'] == -.4


def test_fit_uses_low_speed_unlimited_press_or_hold_and_reports_units():
    frames = [frame(i, angle=i * .01, current=-(.1 + i * .03)) for i in range(8)]
    frames += [frame(8, angle=.2, current=-50., velocity=-.03),
               frame(9, angle=.3, current=-50., velocity=.08),
               frame(10, angle=.4, current=-50., velocity=0.)]
    frames[-1]['glove_torque_limited'][0] = True
    result = analyze(frames)
    assert result['fit']['slope'] == pytest.approx(3.)
    assert result['fit']['intercept'] == pytest.approx(.1)
    assert result['fit']['r_squared'] == pytest.approx(1.)
    assert result['fit']['count'] == 8
    assert [p['phase'] for p in result['points'][-3:]] == ['release', 'press', 'hold']
    assert all(not p['fit_eligible'] for p in result['points'][-3:])
    assert result['points'][-1]['limited'] is True


def test_missing_velocity_is_unknown_and_not_used_as_hold():
    p = frame(0)
    p['glove_filtered_velocity_rad_s'][0] = None
    point = analyze([p])['points'][0]
    assert point['phase'] == 'unknown'
    assert point['fit_eligible'] is False


def test_fit_needs_sufficient_count_and_angle_span():
    assert analyze([frame(i) for i in range(7)])['fit'] is None
    result = analyze([frame(i, angle=.1 + i * .0001) for i in range(8)])
    assert result['fit'] is None
    assert result['counts']['fit_eligible'] == 8


def test_calibrated_fit_and_error_use_current_evidence_not_pd_recalculation():
    frames = [frame(i, angle=i * .01, current=-(.1 + i * .03), calibrated=True) for i in range(8)]
    for p in frames:
        p['glove_target'][0] += .1
        p['glove_driver_pd_torque_nm'] = [987.] * 12
    result = analyze(frames)
    assert result['response_unit'] == 'Nm'
    assert result['slope_unit'] == 'Nm/rad'
    assert result['fit']['slope'] == pytest.approx(6.)
    assert result['torque_error'] == {
        'mean_nm': pytest.approx(-.1), 'rmse_nm': pytest.approx(.1),
        'peak_abs_nm': pytest.approx(.1), 'count': 8,
    }


def test_partial_or_undeclared_calibration_keeps_one_axis_in_amperes():
    frames = [frame(0, calibrated=True), frame(1)]
    result = analyze(frames, time_range=(1000 * MS, 1000 * MS))
    assert result['response_unit'] == 'A'
    assert analyze(frames)['torque_error'] is None
    p = frame(0, calibrated=True)
    p['torque_conversion']['glove_calibration_id'] = ''
    assert analyze([p])['response_unit'] == 'A'


def test_current_sample_uses_latest_past_position_and_command_not_same_frame_future_values():
    prior = frame(0, angle=.02, current=-.1)
    later = frame(1, angle=99., current=-.2)
    later['glove_current_received_ns'][0] = 1001 * MS
    later['glove_raw_position_deg'][0] = math.degrees(99.)
    later['glove_proxy_surface_rad'][0] = 50.
    result = analyze([prior, later])
    assert result['points'][-1]['angle_excess_rad'] == pytest.approx(.02)
    assert result['points'][-1]['response'] == .2
    assert result['points'][-1]['alignment_ms'] == 3.


@pytest.mark.parametrize('change', ['future_position', 'old_position', 'missing_position',
                                    'future_command', 'expired_source', 'invalid_proxy'])
def test_missing_future_or_expired_sources_are_not_filled(change):
    p = frame(0)
    if change == 'future_position': p['glove_motor_received_ns'][0] += 3 * MS
    if change == 'old_position': p['glove_motor_received_ns'][0] -= 20 * MS
    if change == 'missing_position': p['glove_motor_received_ns'][0] = None
    if change == 'future_command': p['glove_submit_ns'] += 5 * MS
    if change == 'expired_source': p['glove_output_source_ns'] -= 100 * MS
    if change == 'invalid_proxy': p['glove_proxy_valid'][0] = False
    assert analyze([p])['points'] == []


def test_unordered_input_and_repeated_current_stamps_do_not_change_result():
    frames = [frame(i, angle=.01 * i, current=-.1 * i) for i in range(4)]
    duplicate = deepcopy(frames[-1])
    duplicate['sequence'] += 1
    duplicate['timestamp_ns'] += MS
    assert analyze(list(reversed(frames)) + [duplicate])['points'] == analyze(frames)['points']


@pytest.mark.parametrize('boundary', ['session', 'gap'])
def test_causal_matching_cannot_reuse_position_or_command_across_boundary(boundary):
    prior = frame(0)
    after = frame(1, session='session-b' if boundary == 'session' else 'session-a')
    if boundary == 'gap': after['sequence'] = 9
    after['glove_motor_received_ns'][0] = None
    after['glove_submit_ns'] = 0
    assert len(analyze([prior, after])['points']) == 1


def test_selected_range_keeps_prior_sources_and_does_not_mutate_snapshot():
    prior = frame(0, angle=.02)
    after = frame(1, angle=.04, current=-.4)
    after['glove_motor_received_ns'][0] = None
    after['glove_submit_ns'] = 0
    frozen = [prior, after]
    original = deepcopy(frozen)
    result = analyze(frozen, time_range=(1008 * MS, 1010 * MS))
    assert len(result['points']) == 1
    assert result['points'][0]['angle_excess_rad'] == pytest.approx(.02)
    assert result['counts']['selected_currents'] == 1
    assert frozen == original


def test_hold_statistics_report_drift_current_peak_to_peak_and_rms():
    frames = [frame(i, angle=.1 + i * .001, current=-.2 - i * .1, velocity=0.) for i in range(3)]
    result = analyze(frames)
    stability = result['stability']
    assert stability['angle_drift_rad_s'] == pytest.approx(.1)
    assert stability['current_peak_to_peak_a'] == pytest.approx(.2)
    assert stability['current_rms_a'] == pytest.approx(math.sqrt(.29 / 3))
    assert stability['count'] == 3


def test_scatter_reference_fields_come_from_the_matched_command():
    prior, later = frame(0), frame(1)
    later['glove_current_received_ns'][0] = 1001 * MS
    later['glove_target'][0] = 99.
    later['glove_proxy_surface_rad'][0] = 99.
    later['glove_driver_target_rad'][0] = 99.
    point = analyze([prior, later])['points'][-1]
    assert point['nominal_nm'] == -.4
    assert point['proxy_rad'] == 0.
    assert point['target_rad'] == -.01
    assert point['velocity_rad_s'] == .03
    assert point['blocking_direction'] == 1


@pytest.mark.parametrize('field', ['glove_torque_limited', 'glove_motion_limited', 'glove_proxy_limited'])
def test_each_limit_excludes_fitting_without_hiding_the_point(field):
    p = frame(0)
    p[field][0] = True
    point = analyze([p])['points'][0]
    assert point['limited'] is True
    assert point['fit_eligible'] is False


def test_missing_limit_state_is_not_assumed_unlimited():
    p = frame(0)
    p.pop('glove_motion_limited')
    result = analyze([p])
    assert result['points'][0]['fit_eligible'] is False
    assert result['counts']['unknown_limits'] == 1


@pytest.mark.parametrize('velocity_ns', [970 * MS, 1000 * MS, None])
def test_velocity_needs_a_real_causal_fresh_stamp(velocity_ns):
    p = frame(0, velocity=0.)
    p['glove_driver_velocity_received_ns'][0] = velocity_ns
    assert analyze([p])['points'][0]['phase'] == 'unknown'


def test_original_request_deadline_cannot_be_extended_by_fresh_axis_source():
    p = frame(0)
    p['glove_output_source_ns'] = 898 * MS
    assert analyze([p])['points'] == []


def test_original_request_is_expired_at_exact_deadline():
    p = frame(0)
    p['glove_output_source_ns'] = 899 * MS
    assert analyze([p])['points'] == []
    p['glove_output_source_ns'] = None
    assert analyze([p])['points'] == []


def test_new_invalid_command_cancels_old_valid_reference():
    frames = [frame(0), frame(1)]
    frames[-1]['glove_proxy_valid'][0] = False
    assert len(analyze(frames)['points']) == 1


def test_gap_does_not_accept_cached_command_with_old_timestamp():
    prior = frame(0)
    later = frame(9)
    later['glove_submit_ns'] = prior['glove_submit_ns']
    later['glove_output_source_ns'] = prior['glove_output_source_ns']
    assert len(analyze([prior, later])['points']) == 1


def test_null_publish_timestamp_does_not_disable_gap_isolation():
    prior, later = frame(0), frame(1)
    prior['published_ns'] = None
    later['sequence'] = 9
    for field in ('glove_submit_ns', 'glove_output_source_ns', 'glove_proxy_source_ns',
                  'glove_filtered_velocity_rad_s', 'glove_driver_velocity_received_ns', 'feedback_request_id'):
        later[field] = prior[field]
    assert len(analyze([prior, later])['points']) == 1


def test_conflicting_same_current_stamp_is_discarded():
    prior, later = frame(0), frame(1)
    later['glove_current_received_ns'][0] = prior['glove_current_received_ns'][0]
    later['glove_current_a'][0] = 999.
    result = analyze([prior, later])
    assert result['points'] == []
    assert result['counts']['conflicting_currents'] == 1


def test_stability_does_not_regress_drift_across_session_boundary():
    frames = [frame(0, velocity=0.), frame(1, angle=9., velocity=0., session='session-b')]
    result = analyze(frames)
    assert result['stability']['count'] == 2
    assert result['stability']['angle_drift_rad_s'] is None


@pytest.mark.parametrize('scale', [1e200, 1e308])
def test_extreme_finite_telemetry_does_not_crash_or_return_infinite_fit(scale):
    frames = [frame(i, angle=.01 * i, current=-scale if i % 2 else scale) for i in range(8)]
    result = analyze(frames)
    assert result['fit'] is None


def test_overflowing_covariance_is_unavailable_instead_of_crashing():
    frames = [frame(i, angle=1e153 if i % 2 else .01,
                    current=1.7e308 if i % 4 < 2 else -1.7e308) for i in range(8)]
    assert analyze(frames)['fit'] is None


def test_hold_statistics_remain_finite_when_peak_to_peak_exceeds_float_range():
    frames = [frame(i, angle=.01 * i, current=1.7e308 if i % 2 else -1.7e308,
                    velocity=0.) for i in range(8)]
    result = analyze(frames)['stability']
    assert result['current_peak_to_peak_a'] is None
    assert result['current_rms_a'] == pytest.approx(1.7e308)
    assert result['current_std_a'] == pytest.approx(1.7e308)


@pytest.mark.parametrize('reason', ['release', 'safe_retract', 'withdrawal_guard'])
def test_release_or_withdrawal_reference_is_displayed_but_not_fitted(reason):
    p = frame(0)
    p['glove_control_reason'][0] = reason
    assert analyze([p])['points'][0]['fit_eligible'] is False


@pytest.mark.parametrize('kp', [0., -1., None])
def test_absent_or_disabled_driver_spring_cannot_prove_stiffness(kp):
    p = frame(0)
    p['glove_driver_kp'][0] = kp
    assert analyze([p])['points'][0]['fit_eligible'] is False
