#!/usr/bin/env /usr/bin/python3
"""单机只读UDP到ROS桥；不导入项目SDK、不连接设备。

启动：source /opt/ros/humble/setup.bash; /usr/bin/python3 scripts/plotjuggler_bridge.py
离线演示：另开终端运行 /usr/bin/python3 scripts/plotjuggler_bridge.py --demo
真实遥测将生产者曲线端口设为8766；--forward-port 8765 可旁路转发原包。
PlotJuggler选择ROS订阅 /glove_bridge/*，务必勾选使用消息header时间戳。
真正的关节角topic用 sensor_msgs/JointState.position（rad）；
电流、名义力矩、误差和诊断量用 plotjuggler_msgs/StatisticsValues.values，
各数值topic使用 /values 和 /names 配对，符合PlotJuggler统计消息解析规则。
单位由topic后缀定义，不把电流或力矩伪装成JointState.position。
原始本机monotonic时刻换算成wall时间，缺失轴为NaN，不补零；不重发旧数据。
手套q12由异步电机帧组合，header为最老帧时间，不表示同时采样。
手套原始角度/电流按电机拆分topic并使用各电机帧时间；旧包缺该时间则不发布。
诊断量是一次UDP到达时的年龄或时间差，跟随滞后不是网络/接触延迟。
"""
import argparse
from bisect import bisect_left
from collections import deque
import json
import math
import socket
import time

MAX_PACKET = 16384
MAX_AGE_NS = 500_000_000
NS = 1_000_000_000

POSITION_TOPICS = frozenset((
    'wuji_position_rad', 'wuji_target_rad',
    'glove_position_rad', 'glove_target_position_rad',
))
NUMERIC_TOPICS = (
    'wuji_current_a', 'wuji_tracking_error_rad', 'wuji_velocity_rad_s',
    'wuji_load_fraction', 'virtual_hand_torque_nm', 'glove_projected_nominal_torque_nm',
    'glove_nominal_feedforward_a', 'glove_nominal_feedforward_nm',
    'wuji_frame_age_s', 'glove_frame_age_s', 'telemetry_age_at_bridge_s',
    'jacobian_reported_age_s', 'glove_sample_to_wuji_submit_s',
    'wuji_sample_to_glove_submit_s', 'control_cycle_duration_s',
    'glove_motor_sample_spread_s', 'wuji_following_lag_estimate_s',
    'stream_alive', 'wuji_lag_estimate_valid', 'telemetry_source_code',
    'feedback_projection_code', 'contact_state_code',
    *(f'{kind}_motor_{i}' for kind in ('glove_raw_position_deg', 'glove_current_a') for i in range(1, 13)),
)


def valid_time(value, now, max_age=MAX_AGE_NS):
    return type(value) is int and 0 < value <= now and now - value <= max_age


def wall_stamp(event_ns, now_ns, wall_ns):
    if not valid_time(event_ns, now_ns):
        return None
    stamp = wall_ns - (now_ns - event_ns)
    return stamp if stamp > 0 else None


def numbers(sample, key, count):
    values = sample.get(key)
    if not isinstance(values, list) or len(values) != count:
        return None
    try:
        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in values):
            return None
    except OverflowError:
        return None
    return values


def names(sample, key, count):
    value = sample.get(key)
    if not isinstance(value, list) or len(value) != count or any(not isinstance(n, str) or not n for n in value):
        return None
    return value


class SessionGate:
    """仅接收更新的会话；退役会话不能重入，序号不能回退。"""
    def __init__(self):
        self.session = None
        self.sequence = -1
        self.last_time = -1
        self.retired = set()

    def accept(self, sample, now):
        if not isinstance(sample, dict) or type(sample.get('schema_version')) is not int or sample['schema_version'] != 1:
            return False
        session, sequence = sample.get('session_id'), sample.get('sequence')
        source_time = sample.get('published_ns', sample.get('timestamp_ns'))
        if (not isinstance(session, str) or not 0 < len(session) <= 128 or
                type(sequence) is not int or sequence < 0 or not valid_time(source_time, now) or
                sample.get('source') not in ('hardware', 'demo') or
                sample.get('mode') not in ('glove-to-hand2', 'hand2-to-glove')):
            return False
        # 出现不可能属于本机的可选时间时整包拒绝，而不是改变时钟域。
        for key in ('timestamp_ns', 'published_ns', 'wuji_received_ns', 'glove_received_ns',
                    'cycle_started_ns', 'cycle_completed_ns', 'wuji_submit_ns', 'glove_submit_ns'):
            value = sample.get(key)
            if value is not None and (type(value) is not int or value <= 0 or value > now):
                return False
        motor_times = sample.get('glove_motor_received_ns')
        if motor_times is not None and (not isinstance(motor_times, list) or len(motor_times) != 12 or
                any(t is not None and (type(t) is not int or t <= 0 or t > now) for t in motor_times)):
            return False
        if session in self.retired:
            return False
        if session == self.session:
            if sequence <= self.sequence or source_time < self.last_time:
                return False
        else:
            if self.session is not None:
                if source_time <= self.last_time:
                    return False
                self.retired.add(self.session)
            self.session = session
        self.sequence, self.last_time = sequence, source_time
        return True


def interpolate(points, t):
    # 输入时间严格递增；二分避免50Hz流上执行二次复杂度的线性扫描。
    if len(points) < 2:
        return None
    index = max(1, bisect_left(points, (t, -math.inf)))
    if index >= len(points):
        return None
    t0, v0 = points[index - 1]
    t1, v1 = points[index]
    if t0 <= t <= t1 and 0 < t1 - t0 <= 80_000_000:
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return None


class LagEstimator:
    """目标到实测的单轴跟随时移；低激励、模糊或缺数据时不输出。"""
    def __init__(self):
        self.target = deque()
        self.actual = deque()

    def clear(self):
        self.target.clear()
        self.actual.clear()

    def add(self, target_time, target, actual_time, actual):
        if ((self.target and target_time - self.target[-1][0] > 80_000_000) or
                (self.actual and actual_time - self.actual[-1][0] > 80_000_000)):
            self.clear()
        for series, t, value in ((self.target, target_time, target), (self.actual, actual_time, actual)):
            if not series or t > series[-1][0]:
                series.append((t, value))
            while series and t - series[0][0] > 3 * NS:
                series.popleft()

    def estimate(self):
        target, actual = list(self.target), list(self.actual)
        if len(target) < 30 or len(actual) < 30 or actual[-1][0] - actual[0][0] < NS:
            return None
        scores = []
        for lag in range(0, 501, 10):
            pairs = [(interpolate(target, t - lag * 1_000_000), v) for t, v in actual]
            pairs = [(x, y) for x, y in pairs if x is not None]
            if len(pairs) < 30 or len(pairs) < len(actual) * 0.65:
                continue
            mx = sum(x for x, _ in pairs) / len(pairs)
            my = sum(y for _, y in pairs) / len(pairs)
            xx = sum((x - mx) ** 2 for x, _ in pairs)
            yy = sum((y - my) ** 2 for _, y in pairs)
            if min(xx, yy) / len(pairs) < 0.0001:
                continue
            score = sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(xx * yy)
            scores.append((score, lag))
        if not scores:
            return None
        score, lag = max(scores)
        alternatives = [s for s, other in scores if abs(other - lag) >= 100]
        if score < 0.85 or lag in (0, 500) or (alternatives and score - max(alternatives) < 0.02):
            return None
        return lag / 1000.0


def fields(sample, now):
    """返回 (topic, 轴名, 数值, 原始事件时间)，缺值/过期数据不输出。"""
    result = []
    wn = names(sample, 'wuji_joint_names', 20)
    gn = names(sample, 'glove_position_names', 12)
    motors = [f'motor_{i}' for i in range(1, 13)]
    # 目标只使用提交时间，提交只表示SDK调用完成/入队，不代表物理发送。
    specs = [
        ('wuji_position_rad', 20, wn, 'wuji_received_ns', 'wuji_position_rad'),
        ('wuji_target_rad', 20, wn, 'wuji_submit_ns', 'wuji_target_rad'),
        ('glove_position_rad', 12, gn, 'glove_received_ns', 'glove_position_rad'),
        ('glove_target_position_rad', 12, gn, 'glove_submit_ns', 'glove_target_position_rad'),
        ('wuji_current_a', 20, wn, 'wuji_received_ns', 'wuji_current_a'),
        ('wuji_tracking_error_rad', 20, wn, 'wuji_submit_ns', 'wuji_tracking_error_rad'),
        ('wuji_velocity_rad_s', 20, wn, 'wuji_received_ns', 'wuji_velocity_rad_s'),
        ('wuji_load_fraction', 20, wn, 'wuji_received_ns', 'wuji_load_fraction'),
        ('virtual_hand_torque_nm', 20, wn, 'glove_submit_ns', 'virtual_hand_torque_nm'),
        ('glove_projected_torque_nm', 12, motors, 'glove_submit_ns', 'glove_projected_nominal_torque_nm'),
    ]
    unit = sample.get('glove_target_unit')
    if unit in ('A', 'Nm'):
        specs.append(('glove_target', 12, motors, 'glove_submit_ns',
                      'glove_nominal_feedforward_' + ('a' if unit == 'A' else 'nm')))
    for key, count, axis_names, time_key, topic in specs:
        if key == 'wuji_target_rad' and sample.get('mode') != 'glove-to-hand2':
            continue
        if key == 'glove_target_position_rad' and sample.get('mode') != 'hand2-to-glove':
            continue
        values, stamp = numbers(sample, key, count), sample.get(time_key)
        if axis_names is None or values is None or not valid_time(stamp, now):
            continue
        if any(v is not None for v in values):
            result.append((topic, axis_names, [float(v) if v is not None else float('nan') for v in values], stamp))
    motor_times = sample.get('glove_motor_received_ns')
    if isinstance(motor_times, list) and len(motor_times) == 12:
        for key in ('glove_raw_position_deg', 'glove_current_a'):
            values = numbers(sample, key, 12)
            if values is None:
                continue
            for i, (value, stamp) in enumerate(zip(values, motor_times)):
                if value is not None and valid_time(stamp, now):
                    result.append((f'{key}_motor_{i+1}', [motors[i]], [float(value)], stamp))
    return result


def diagnostics(sample, now):
    result = []
    def age(name, key):
        t = sample.get(key)
        if valid_time(t, now):
            result.append((name, (now - t) / NS))
    age('wuji_frame_age_s', 'wuji_received_ns')
    age('glove_frame_age_s', 'glove_received_ns')
    age('telemetry_age_at_bridge_s', 'published_ns')
    jac = sample.get('jacobian_age_ns')
    pub = sample.get('published_ns')
    if type(jac) is int and jac >= 0 and valid_time(pub, now) and jac <= pub:
        result.append(('jacobian_reported_age_s', jac / NS))
    for source, submit, name in [('glove_received_ns', 'wuji_submit_ns', 'glove_sample_to_wuji_submit_s'),
                                 ('wuji_received_ns', 'glove_submit_ns', 'wuji_sample_to_glove_submit_s')]:
        start, end = sample.get(source), sample.get(submit)
        if valid_time(start, now) and valid_time(end, now) and start <= end:
            result.append((name, (end - start) / NS))
    start, end = sample.get('cycle_started_ns'), sample.get('cycle_completed_ns')
    if valid_time(start, now) and valid_time(end, now) and start <= end:
        result.append(('control_cycle_duration_s', (end - start) / NS))
    stamps = sample.get('glove_motor_received_ns')
    if isinstance(stamps, list) and len(stamps) == 12 and all(valid_time(t, now) for t in stamps):
        result.append(('glove_motor_sample_spread_s', (max(stamps) - min(stamps)) / NS))
    projection = {'disabled': 0, 'active': 1, 'refreshing_pose': 2, 'refreshing_age': 3}
    contact = {'not_configured': 0, 'free': 1, 'arming': 2, 'contact': 3,
               'releasing': 4, 'waiting_geometry': 5}
    if sample.get('feedback_state') in projection:
        result.append(('feedback_projection_code', float(projection[sample['feedback_state']])))
    if sample.get('contact_state') in contact:
        result.append(('contact_state_code', float(contact[sample['contact_state']])))
    if sample.get('source') in ('hardware', 'demo'):
        result.append(('telemetry_source_code', 1.0 if sample['source'] == 'hardware' else 0.0))
    return result


def decode(payload):
    if not 0 < len(payload) <= MAX_PACKET:
        return None
    try:
        def reject(value):
            raise ValueError(value)
        value = json.loads(payload, parse_constant=reject)
        return value if isinstance(value, dict) else None
    except (ValueError, UnicodeError, RecursionError):
        return None


def demo(port):
    import uuid
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    session, start, seq = uuid.uuid4().hex, time.monotonic_ns(), 0
    print('仅发送合成数据到本机UDP，Ctrl-C结束。', flush=True)
    def signal(t):
        return 0.25 * math.sin(6.1*t) + 0.15 * math.sin(13.7*t) + 0.09 * math.sin(23.3*t)
    try:
        while True:
            now = time.monotonic_ns()
            t = (now - start) / NS
            seq += 1
            contact = math.sin(1.5*t) > 0.3
            target, actual = signal(t), signal(t-0.15)
            sample = dict(schema_version=1, source='demo', mode='glove-to-hand2', session_id=session,
                          sequence=seq, executed=True, timestamp_ns=now-1_500_000, published_ns=now,
                          wuji_received_ns=now-1_500_000, glove_received_ns=now-2_000_000,
                          cycle_started_ns=now-4_000_000, cycle_completed_ns=now-250_000,
                          wuji_submit_ns=now-1_000_000, glove_submit_ns=now-500_000,
                          glove_motor_received_ns=[now-2_000_000] * 12,
                          wuji_joint_names=[f'wuji_{i}' for i in range(20)],
                          glove_position_names=[f'glove_q_{i}' for i in range(12)],
                          glove_joint_names=[f'motor_{i+1}' for i in range(12)],
                          wuji_target_rad=[target] * 20, wuji_position_rad=[actual] * 20,
                          glove_position_rad=[target] * 12, glove_raw_position_deg=[target*180/math.pi] * 12,
                          wuji_tracking_error_rad=[target-actual] * 20,
                          wuji_velocity_rad_s=[(signal(t-0.149)-signal(t-0.151))/0.002] * 20,
                          wuji_current_a=[0.3*math.sin(t+i/5) for i in range(20)],
                          wuji_load_fraction=[0.15*math.sin(t+i/5) for i in range(20)],
                          glove_current_a=[0.2*math.sin(t+i/5) for i in range(12)],
                          glove_target_unit='Nm', glove_target=[0.10 if contact else 0.0] * 12,
                          glove_projected_torque_nm=[0.15 if contact else 0.0] * 12,
                          virtual_hand_torque_nm=[-0.01 if contact else 0.0] * 20,
                          contact_active_mask=[contact] * 20, contact_reflection_gain=1.0,
                          mapping_mode='contact_impedance_estimate', feedback_state='active',
                          contact_state='contact' if contact else 'free', jacobian_age_ns=20_000_000)
            try:
                sock.sendto(json.dumps(sample).encode(), ('127.0.0.1', port))
            except OSError:
                pass
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def run(args):
    # 只在真正运行ROS入口时导入，纯函数测试与demo无需ROS或项目SDK。
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import JointState
    from plotjuggler_msgs.msg import StatisticsNames, StatisticsValues

    class Bridge(Node):
        def __init__(self):
            super().__init__('glove_plotjuggler_bridge')
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            self.sock.bind(('127.0.0.1', args.port))
            self.forward = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.forward_port else None
            if self.forward:
                self.forward.setblocking(False)
            self.gate, self.lag = SessionGate(), LagEstimator()
            self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.VOLATILE)
            self.name_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                       durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.pubs = {
                topic: self.create_publisher(
                    JointState if topic in POSITION_TOPICS else StatisticsValues,
                    '/glove_bridge/' + topic + ('' if topic in POSITION_TOPICS else '/values'), self.qos,
                )
                for topic in (*POSITION_TOPICS, *NUMERIC_TOPICS)
            }
            # PlotJuggler按同一前缀的 /names 与 /values 配对统计量。
            # 提前创建names publisher，让用户能在硬件数据出现前订阅。
            self.name_pubs = {
                topic: self.create_publisher(StatisticsNames, '/glove_bridge/' + topic + '/names', self.name_qos)
                for topic in NUMERIC_TOPICS
            }
            self.names = {}
            self.last_lag = 0
            self.last_valid_lag_ns = None
            self.last_packet_ns = None
            self.create_timer(0.005, self.drain)
            self.create_timer(0.2, self.heartbeat)

        def publish_value(self, topic, axes, values, timestamp, now, wall):
            stamp = wall_stamp(timestamp, now, wall)
            if stamp is None:
                return
            if topic not in self.pubs:
                return
            if topic in POSITION_TOPICS:
                msg = JointState()
                msg.name, msg.position = axes, values
            else:
                previous = self.names.get(topic)
                if previous is None or previous[0] != tuple(axes):
                    version = 1 if previous is None else previous[1] + 1
                    labels = StatisticsNames()
                    labels.header.stamp.sec, labels.header.stamp.nanosec = divmod(stamp, NS)
                    labels.names, labels.names_version = axes, version
                    self.name_pubs[topic].publish(labels)
                    self.names[topic] = (tuple(axes), version)
                msg = StatisticsValues()
                msg.names_version, msg.values = self.names[topic][1], values
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, NS)
            self.pubs[topic].publish(msg)

        def heartbeat(self):
            now, wall = time.monotonic_ns(), time.time_ns()
            alive = self.last_packet_ns is not None and now - self.last_packet_ns <= MAX_AGE_NS
            self.publish_value('stream_alive', ['stream_alive'], [float(alive)], now, now, wall)
            lag_fresh = alive and self.last_valid_lag_ns is not None and now - self.last_valid_lag_ns <= MAX_AGE_NS
            self.publish_value('wuji_lag_estimate_valid', ['valid'], [float(lag_fresh)], now, now, wall)

        def drain(self):
            # 有界批量，避免UDP洪泛阻塞ROS执行器。
            for _ in range(100):
                try:
                    payload, _ = self.sock.recvfrom(MAX_PACKET + 1)
                except BlockingIOError:
                    break
                now, wall = time.monotonic_ns(), time.time_ns()
                sample = decode(payload)
                old_session = self.gate.session
                if sample is None or not self.gate.accept(sample, now):
                    continue
                self.last_packet_ns = now
                if old_session != self.gate.session:
                    self.lag.clear()
                    self.last_valid_lag_ns = None
                if self.forward:
                    try:
                        self.forward.sendto(payload, ('127.0.0.1', args.forward_port))
                    except OSError:
                        pass
                entries = fields(sample, now)
                for topic, axes, values, stamp in entries:
                    self.publish_value(topic, axes, values, stamp, now, wall)
                for topic, value in diagnostics(sample, now):
                    self.publish_value(topic, [topic], [value], now, now, wall)
                target = numbers(sample, 'wuji_target_rad', 20)
                actual = numbers(sample, 'wuji_position_rad', 20)
                tt, at = sample.get('wuji_submit_ns'), sample.get('wuji_received_ns')
                if (sample.get('mode') != 'glove-to-hand2' or target is None or actual is None or
                    target[args.lag_axis] is None or actual[args.lag_axis] is None or
                    not valid_time(tt, now) or not valid_time(at, now)):
                    self.lag.clear()
                    self.last_valid_lag_ns = None
                    continue
                self.lag.add(tt, target[args.lag_axis], at, actual[args.lag_axis])
                if now - self.last_lag >= 200_000_000:
                    self.last_lag = now
                    lag = self.lag.estimate()
                    self.last_valid_lag_ns = now if lag is not None else None
                    if lag is not None:
                        self.publish_value('wuji_following_lag_estimate_s', [f'wuji_axis_{args.lag_axis}'],
                                           [lag], at, now, wall)

        def destroy_node(self):
            self.sock.close()
            if self.forward:
                self.forward.close()
            return super().destroy_node()

    rclpy.init()
    node = None
    try:
        node = Bridge()
        print(f"PLOTJUGGLER_BRIDGE_READY port={args.port} lag_axis={args.lag_axis}", flush=True)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', type=int, default=8766, help='本机UDP输入端口（默认8766）')
    parser.add_argument('--forward-port', type=int, help='可选原包转发至本机旧UI，例如8765')
    parser.add_argument('--lag-axis', type=int, default=6, help='Wuji时移估计轴索引0–19（默认6：食指S3）')
    parser.add_argument('--demo', action='store_true', help='发送合成演示包，不启动ROS或硬件')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or (args.forward_port is not None and
            (not 1 <= args.forward_port <= 65535 or args.forward_port == args.port)):
        parser.error('端口须为1–65535且转发端口不能等于输入端口')
    if not 0 <= args.lag_axis < 20:
        parser.error('--lag-axis须为0–19')
    demo(args.port) if args.demo else run(args)


if __name__ == '__main__':
    main()
