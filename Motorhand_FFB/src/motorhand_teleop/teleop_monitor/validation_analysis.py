"""只读验证分析：用真实接收时刻因果匹配，不插值或复用未来样本。"""
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import math

from .quantities import finite


@dataclass(frozen=True)
class ValidationOptions:
    max_pair_age_ns: int = 20_000_000
    low_speed_rad_s: float = .05
    motion_deadband_rad_s: float = .02
    min_fit_samples: int = 8
    min_angle_span_rad: float = .005

    def __post_init__(self):
        if type(self.max_pair_age_ns) is not int or self.max_pair_age_ns <= 0:
            raise ValueError('允许配对时差必须为正整数纳秒')
        if type(self.min_fit_samples) is not int or self.min_fit_samples < 2:
            raise ValueError('拟合至少需要两个样本')
        if (not finite(self.low_speed_rad_s) or self.low_speed_rad_s <= 0
                or not finite(self.motion_deadband_rad_s) or self.motion_deadband_rad_s < 0
                or self.motion_deadband_rad_s > self.low_speed_rad_s
                or not finite(self.min_angle_span_rad) or self.min_angle_span_rad <= 0):
            raise ValueError('速度阈值或最小角跨度无效')


def _axis(frame, field, index):
    values = frame.get(field)
    return values[index] if isinstance(values, (list, tuple)) and index < len(values) else None


def _stamp(value):
    return type(value) is int and 0 < value < 2**64


def _calibration(frame, index):
    """验证字段仍需携带声明、对应轴Kt和独立证据ID。"""
    metadata = frame.get('torque_conversion')
    if not isinstance(metadata, dict) or metadata.get('glove_calibrated') is not True:
        return None, None
    evidence = metadata.get('glove_calibration_id')
    coefficient = _axis(metadata, 'glove_nm_per_a', index)
    torque = _axis(frame, 'glove_validated_torque_nm', index)
    if (not isinstance(evidence, str) or not evidence.strip()
            or not finite(coefficient) or coefficient == 0 or not finite(torque)):
        return None, None
    return torque, evidence


def _segments(frames, counts):
    """按遥测发布时间整理乱序输入；序号断档后重新建立来源索引。"""
    ordered = []
    for frame in frames:
        if (not isinstance(frame, dict) or frame.get('type') == 'teleop_status'
                or not isinstance(frame.get('session_id'), str)
                or not frame['session_id'] or not _stamp(frame.get('timestamp_ns'))
                or type(frame.get('sequence')) is not int or frame['sequence'] < 0):
            counts['invalid_frames'] += 1
            continue
        ordered.append(frame)
    ordered.sort(key=lambda p: (p['timestamp_ns'], p['session_id'], p['sequence']))
    previous = None
    seen = set()
    segments = []
    for frame in ordered:
        identity = frame['session_id'], frame['sequence']
        if identity in seen:
            counts['duplicate_frames'] += 1
            continue
        seen.add(identity)
        new_session = previous is None or frame['session_id'] != previous['session_id']
        gap = not new_session and frame['sequence'] != previous['sequence'] + 1
        if new_session or gap:
            # 断档后的缓存值若在断档前已存在，不能证明仍属当前命令。
            cutoff = None
            if gap:
                published = previous.get('published_ns')
                cutoff = published if _stamp(published) else previous['timestamp_ns']
            segments.append((cutoff, []))
            if gap:
                counts['sequence_gaps'] += 1
        segments[-1][1].append(frame)
        previous = frame
    return segments


def _record(index, stamp, payload, kind, counts):
    if stamp not in index:
        index[stamp] = payload
    else:
        counts['duplicate_' + kind] += 1
        if index[stamp] != payload:
            # 同一真实源时刻出现矛盾内容时，不任取一个值。
            index[stamp] = None
            counts['conflicting_' + kind] += 1


def _command(frame, index):
    return dict(
        source_ns=frame.get('glove_output_source_ns'),
        proxy_source_ns=_axis(frame, 'glove_proxy_source_ns', index),
        proxy=_axis(frame, 'glove_proxy_surface_rad', index),
        target=_axis(frame, 'glove_driver_target_rad', index),
        kp=_axis(frame, 'glove_driver_kp', index),
        valid=_axis(frame, 'glove_proxy_valid', index),
        direction=_axis(frame, 'glove_blocking_direction', index),
        limits=tuple(_axis(frame, field, index) for field in (
            'glove_torque_limited', 'glove_motion_limited', 'glove_proxy_limited')),
        velocity=_axis(frame, 'glove_filtered_velocity_rad_s', index),
        velocity_ns=_axis(frame, 'glove_driver_velocity_received_ns', index),
        nominal=_axis(frame, 'glove_target', index) if frame.get('glove_target_unit') == 'Nm' else None,
        request_id=frame.get('feedback_request_id'),
        reason=_axis(frame, 'glove_control_reason', index),
    )


def _indexes(frames, index, cutoff, counts):
    positions, currents, commands = {}, {}, {}
    for frame in frames:
        position_ns = _axis(frame, 'glove_motor_received_ns', index)
        position = _axis(frame, 'glove_raw_position_deg', index)
        if _stamp(position_ns) and finite(position) and (cutoff is None or position_ns > cutoff):
            _record(positions, position_ns, math.radians(position), 'positions', counts)
        current_ns = _axis(frame, 'glove_current_received_ns', index)
        current = _axis(frame, 'glove_current_a', index)
        if _stamp(current_ns) and finite(current) and (cutoff is None or current_ns > cutoff):
            torque, evidence = _calibration(frame, index)
            _record(currents, current_ns, (current, torque, evidence), 'currents', counts)
        else:
            counts['missing_current'] += 1
        submit_ns = frame.get('glove_submit_ns')
        if _stamp(submit_ns) and (cutoff is None or submit_ns > cutoff):
            _record(commands, submit_ns, _command(frame, index), 'commands', counts)
    return positions, currents, commands


def _latest(stamps, values, time_ns):
    index = bisect_right(stamps, time_ns) - 1
    if index < 0:
        return None, None
    stamp = stamps[index]
    return stamp, values[stamp]


def _linear_fit(xs, ys):
    """中心化以避免纳秒时钟的绝对量级损害回归精度。"""
    if len(xs) < 2:
        return None
    try:
        mean_x = math.fsum(x / len(xs) for x in xs)
        mean_y = math.fsum(y / len(ys) for y in ys)
        centered_x = [x - mean_x for x in xs]
        centered_y = [y - mean_y for y in ys]
        xx = math.fsum(x * x for x in centered_x)
        if not finite(xx) or xx <= 0:
            return None
        slope = math.fsum(x * y for x, y in zip(centered_x, centered_y)) / xx
        intercept = mean_y - slope * mean_x
        yy = math.fsum(y * y for y in centered_y)
        if not all(finite(v) for v in (slope, intercept, yy)):
            return None
        residual = math.fsum((y - slope * x) ** 2 for x, y in zip(centered_x, centered_y))
        r_squared = 1. - residual / yy if yy > 0 else 1.
    except (OverflowError, ValueError):
        # 解码允许有限浮点数；极端数据的中间乘积仍可能溢出。
        return None
    if not all(finite(v) for v in (slope, intercept, r_squared)):
        return None
    return dict(slope=slope, intercept=intercept, r_squared=r_squared, count=len(xs))


def _rms(values):
    scale = max(abs(value) for value in values)
    if scale == 0:
        return 0.
    return scale * math.sqrt(math.fsum((value / scale) ** 2 / len(values) for value in values))


def _stability(points):
    held = [p for p in points if p['phase'] == 'hold' and p['fit_eligible']]
    result = dict(angle_drift_rad_s=None, current_peak_to_peak_a=None,
                  current_rms_a=None, current_std_a=None, count=len(held), angle_count=0)
    if not held:
        return result
    currents = [p['current_a'] for p in held]
    peak_to_peak = max(currents) - min(currents)
    result['current_peak_to_peak_a'] = peak_to_peak if finite(peak_to_peak) else None
    result['current_rms_a'] = _rms(currents)
    scale = max(abs(value) for value in currents)
    normalized = [value / scale for value in currents] if scale else currents
    mean = math.fsum(value / len(currents) for value in normalized)
    result['current_std_a'] = scale * _rms([value - mean for value in normalized])
    positions = {(p['segment_index'], p['position_time_ns']): p['actual_angle_rad'] for p in held}
    result['angle_count'] = len(positions)
    if len({segment for segment, _ in positions}) == 1:
        stamps = [stamp for _, stamp in positions]
        start = min(stamps)
        fit = _linear_fit([(stamp - start) / 1e9 for stamp in stamps], list(positions.values()))
        if fit is not None:
            result['angle_drift_rad_s'] = fit['slope']
    return result


def analyze_axis(frames, raw_index, time_range=None, options=None):
    """分析raw J1–J12中的一轴，time_range为闭区间的绝对纳秒。

    先在全部帧上建立分段因果索引，再筛选电流响应时刻；因此选区不会丢掉
    区间之前有效的命令。一个分析快照固定一种证据等级，任何真实电流样本
    缺少独立标定时整轴使用A。输入帧及其数组均不写入。
    """
    if type(raw_index) is not int or not 0 <= raw_index < 12:
        raise ValueError('raw轴索引必须为0至11')
    options = options or ValidationOptions()
    if time_range is not None:
        if (not isinstance(time_range, (list, tuple)) or len(time_range) != 2
                or any(type(t) is not int or t < 0 for t in time_range)
                or time_range[0] > time_range[1]):
            raise ValueError('选区必须为有序的起止纳秒')
    counts = Counter({key: 0 for key in (
        'selected_currents', 'matched', 'fit_eligible', 'missing_position', 'stale_position',
        'missing_command', 'invalid_command', 'expired_command', 'invalid_proxy',
        'unknown_velocity', 'limited', 'unknown_limits', 'duplicate_currents',
        'sequence_gaps', 'invalid_frames', 'duplicate_frames', 'missing_current',
        'inactive_spring', 'release_or_withdrawal')})
    segments = _segments(frames, counts)
    indexed = [_indexes(segment, raw_index, cutoff, counts) for cutoff, segment in segments]
    all_currents = [sample for _, currents, _ in indexed for sample in currents.values() if sample is not None]
    calibrated = bool(all_currents) and all(sample[1] is not None for sample in all_currents)
    points = []
    errors = []
    for segment_index, (positions, currents, commands) in enumerate(indexed):
        position_stamps, command_stamps = sorted(positions), sorted(commands)
        for current_ns, sample in sorted(currents.items()):
            if sample is None or (time_range is not None and not time_range[0] <= current_ns <= time_range[1]):
                continue
            counts['selected_currents'] += 1
            current, torque, _ = sample
            position_ns, angle = _latest(position_stamps, positions, current_ns)
            if angle is None:
                counts['missing_position'] += 1
                continue
            if current_ns - position_ns > options.max_pair_age_ns:
                counts['stale_position'] += 1
                continue
            submit_ns, command = _latest(command_stamps, commands, current_ns)
            if command is None:
                counts['missing_command'] += 1
                continue
            source_ns, proxy_source_ns = command['source_ns'], command['proxy_source_ns']
            if (not _stamp(source_ns) or not _stamp(proxy_source_ns)
                    or source_ns > submit_ns or proxy_source_ns > submit_ns
                    or not _stamp(command['request_id'])):
                counts['invalid_command'] += 1
                continue
            if current_ns - source_ns >= 100_000_000:
                counts['expired_command'] += 1
                continue
            direction = command['direction']
            if (command['valid'] is not True or type(direction) is not int
                    or direction not in (-1, 1) or not finite(command['proxy'])):
                counts['invalid_proxy'] += 1
                continue
            velocity, velocity_ns = command['velocity'], command['velocity_ns']
            velocity_valid = (finite(velocity) and _stamp(velocity_ns) and velocity_ns <= submit_ns
                              and 0 <= current_ns - velocity_ns <= options.max_pair_age_ns)
            directed_velocity = direction * velocity if velocity_valid else None
            phase = ('unknown' if directed_velocity is None else
                     'press' if directed_velocity > options.motion_deadband_rad_s else
                     'release' if directed_velocity < -options.motion_deadband_rad_s else 'hold')
            limits_known = all(type(value) is bool for value in command['limits'])
            limited = any(value is True for value in command['limits'])
            spring_active = finite(command['kp']) and command['kp'] > 0
            reason = command['reason']
            releasing = isinstance(reason, str) and any(
                word in reason.lower() for word in ('release', 'retract', 'withdrawal'))
            excess = direction * (angle - command['proxy'])
            eligible = (spring_active and not releasing and limits_known and not limited and phase in ('press', 'hold')
                        and abs(directed_velocity) <= options.low_speed_rad_s and excess >= 0)
            point = dict(time_ns=current_ns, angle_excess_rad=excess,
                         response=-direction * (torque if calibrated else current), phase=phase,
                         fit_eligible=eligible, limited=limited,
                         alignment_ms=(current_ns - position_ns) / 1e6,
                         command_age_ms=(current_ns - submit_ns) / 1e6,
                         input_age_ms=(current_ns - source_ns) / 1e6,
                         velocity_age_ms=(current_ns - velocity_ns) / 1e6 if velocity_valid else None,
                         actual_angle_rad=angle, proxy_rad=command['proxy'],
                         blocking_direction=direction,
                         target_rad=command['target'] if finite(command['target']) else None,
                         current_a=current, velocity_rad_s=velocity if velocity_valid else None,
                         position_time_ns=position_ns,
                         command_submit_ns=submit_ns, reference_source_ns=proxy_source_ns,
                         feedback_request_id=command['request_id'],
                         nominal_nm=command['nominal'] if finite(command['nominal']) else None,
                         segment_index=segment_index, control_reason=command['reason'])
            points.append(point)
            counts['unknown_velocity'] += not velocity_valid
            counts['limited'] += limited
            counts['unknown_limits'] += not limits_known
            counts['inactive_spring'] += not spring_active
            counts['release_or_withdrawal'] += releasing
            if calibrated and finite(command['nominal']):
                errors.append(torque - command['nominal'])
    points.sort(key=lambda point: (point['time_ns'], point['segment_index']))
    counts['matched'] = len(points)
    eligible = [p for p in points if p['fit_eligible']]
    counts['fit_eligible'] = len(eligible)
    angle_span = (max(p['angle_excess_rad'] for p in eligible) - min(p['angle_excess_rad'] for p in eligible)
                  if eligible else 0.)
    fit = None
    if len(eligible) >= options.min_fit_samples and angle_span >= options.min_angle_span_rad:
        fit = _linear_fit([p['angle_excess_rad'] for p in eligible], [p['response'] for p in eligible])
    notes = [
        '电流为响应时钟；实际角只取不晚于电流且相差不超过'
        f'{options.max_pair_age_ns / 1e6:g}ms的真实接收样本，不插值。',
        '命令取同会话连续序号段内最近已提交请求；原输入时刻+100ms后失效。',
        'alignment_ms为电流与实际角接收时差；命令、输入和速度年龄另列。',
        '拟合只用有效代理、已提交正Kp、低速按入或保持、非负超出角、无释放/撤离原因且三类限幅明确为false的样本。',
        f'低速阈值≤{options.low_speed_rad_s:g}rad/s；保持速度死区±{options.motion_deadband_rad_s:g}rad/s。',
        '保持统计只用上述可拟合条件中的hold样本；角漂移不跨会话或序号断档。',
        '缺失速度标为unknown，不猜测按入、回开或保持。',
    ]
    if calibrated:
        notes.append('Nm来自声明的独立电流标定估计，未提供直接力矩传感器测量。')
    else:
        notes.append('本轴整份快照缺少完整独立标定证据，统一使用A及A/rad；不计算力矩误差。')
    if len(eligible) < options.min_fit_samples:
        notes.append(f'拟合样本不足：{len(eligible)}/{options.min_fit_samples}。')
    elif angle_span < options.min_angle_span_rad:
        notes.append(f'有效角跨度不足：{angle_span:g}rad < {options.min_angle_span_rad:g}rad。')
    torque_error = None
    if errors:
        torque_error = dict(mean_nm=math.fsum(e / len(errors) for e in errors),
                            rmse_nm=_rms(errors),
                            peak_abs_nm=max(abs(e) for e in errors), count=len(errors))
    return dict(points=points, response_unit='Nm' if calibrated else 'A',
                slope_unit='Nm/rad' if calibrated else 'A/rad', fit=fit,
                stability=_stability(points), torque_error=torque_error,
                counts=dict(counts), notes=notes, angle_span_rad=angle_span,
                calibration_ids=sorted({sample[2] for sample in all_currents}) if calibrated else [])
