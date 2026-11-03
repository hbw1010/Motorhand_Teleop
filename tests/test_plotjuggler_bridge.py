"""桥协议、时间和数值估计回归：不导入ROS或设备SDK。"""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'plotjuggler_bridge', Path(__file__).resolve().parents[1] / 'scripts' / 'plotjuggler_bridge.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

NOW = 10_000_000_000


def packet(**kwargs):
    sample = dict(schema_version=1, session_id='a', sequence=1, source='hardware',
                  mode='glove-to-hand2', timestamp_ns=NOW-30_000_000,
                  published_ns=NOW-10_000_000, glove_received_ns=NOW-30_000_000,
                  wuji_received_ns=NOW-20_000_000, wuji_submit_ns=NOW-15_000_000,
                  glove_motor_received_ns=[NOW-30_000_000+i*1_000_000 for i in range(12)],
                  wuji_joint_names=[f'w{i}' for i in range(20)],
                  glove_position_names=[f'q{i}' for i in range(12)],
                  wuji_position_rad=[0.2]*20, wuji_target_rad=[0.3]*20,
                  glove_current_a=[None, 0.0, -1.0]+[None]*9)
    sample.update(kwargs)
    return sample


class BridgeTests(unittest.TestCase):
    def test_sessions_sequence_expiry_and_future(self):
        gate = bridge.SessionGate()
        self.assertTrue(gate.accept(packet(), NOW))
        self.assertFalse(gate.accept(packet(), NOW))
        self.assertFalse(gate.accept(packet(sequence=2, published_ns=NOW+1), NOW))
        self.assertFalse(gate.accept(packet(sequence=2, published_ns=NOW-1_000_000_000), NOW))
        self.assertFalse(gate.accept(packet(sequence=2, published_ns=NOW-20_000_000), NOW))
        self.assertFalse(gate.accept(packet(sequence=2, wuji_received_ns=NOW+1), NOW))
        self.assertFalse(gate.accept(packet(sequence=2, schema_version=2), NOW))
        self.assertTrue(gate.accept(packet(session_id='b', published_ns=NOW-1, sequence=1), NOW))
        self.assertFalse(gate.accept(packet(session_id='a', sequence=100, published_ns=NOW), NOW))
        self.assertFalse(gate.accept(packet(session_id='c', published_ns=NOW-2), NOW))

    def test_values_names_units_and_independent_times(self):
        result = {topic: (axes, values, stamp) for topic, axes, values, stamp in bridge.fields(packet(), NOW)}
        self.assertEqual(result['wuji_target_rad'][2], NOW-15_000_000)
        self.assertEqual(result['wuji_position_rad'][2], NOW-20_000_000)
        self.assertEqual(result['glove_current_a_motor_2'][2], NOW-29_000_000)
        self.assertEqual(result['glove_current_a_motor_2'][1], [0.0])
        self.assertEqual(result['glove_current_a_motor_3'][1], [-1.0])
        self.assertEqual(result['glove_current_a_motor_3'][0], ['motor_3'])
        self.assertNotIn('glove_current_a_motor_1', result)
        missing = dict((x[0], x) for x in bridge.fields(packet(glove_motor_received_ns=None), NOW))
        self.assertNotIn('glove_current_a_motor_2', missing)
        self.assertNotIn('wuji_target_rad', dict((x[0], x) for x in bridge.fields(packet(wuji_submit_ns=None), NOW)))
        self.assertNotIn('wuji_target_rad', dict((x[0], x) for x in bridge.fields(packet(mode='hand2-to-glove'), NOW)))
        self.assertNotIn('wuji_position_rad', dict((x[0], x) for x in bridge.fields(packet(wuji_received_ns=NOW-bridge.NS), NOW)))
        self.assertIsNone(bridge.numbers(packet(wuji_position_rad=[True]*20), 'wuji_position_rad', 20))
        self.assertIsNone(bridge.numbers(packet(wuji_position_rad=[float('nan')]*20), 'wuji_position_rad', 20))
        self.assertEqual(bridge.wall_stamp(NOW-20, NOW, 1000), 980)
        self.assertIsNone(bridge.wall_stamp(NOW+1, NOW, 1000))

    def test_diagnostics_require_real_times(self):
        result = dict(bridge.diagnostics(packet(jacobian_age_ns=100_000_000), NOW))
        self.assertAlmostEqual(result['glove_sample_to_wuji_submit_s'], .015)
        self.assertAlmostEqual(result['wuji_frame_age_s'], .02)
        self.assertAlmostEqual(result['glove_frame_age_s'], .03)
        self.assertAlmostEqual(result['telemetry_age_at_bridge_s'], .01)
        self.assertAlmostEqual(result['jacobian_reported_age_s'], .1)
        result = dict(bridge.diagnostics(packet(wuji_submit_ns=None, published_ns=None), NOW))
        self.assertNotIn('glove_sample_to_wuji_submit_s', result)
        self.assertNotIn('telemetry_age_at_bridge_s', result)
        self.assertNotIn('jacobian_reported_age_s', result)
        result = dict(bridge.diagnostics(packet(wuji_submit_ns=NOW-40_000_000), NOW))
        self.assertNotIn('glove_sample_to_wuji_submit_s', result)

    def test_known_delay_low_excitation_and_gaps(self):
        lag = bridge.LagEstimator()
        def signal(t):
            return .25*math.sin(6.1*t) + .15*math.sin(13.7*t) + .09*math.sin(23.3*t)
        for i in range(150):
            t = i * .02
            stamp = NOW + i * 20_000_000
            lag.add(stamp, signal(t), stamp, signal(t-.15))
        estimate = lag.estimate()
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate, .15, delta=.02)
        lag.add(NOW+4*bridge.NS, 0.0, NOW+4*bridge.NS, 0.0)
        self.assertIsNone(lag.estimate())
        lag.clear()
        for i in range(150):
            stamp = NOW+i*20_000_000
            lag.add(stamp, 0.1, stamp, 0.1)
        self.assertIsNone(lag.estimate())

    def test_json_rejects_bad_data(self):
        self.assertIsNone(bridge.decode(b'{"x":NaN}'))
        self.assertIsNone(bridge.decode(b'[]'))
        self.assertIsNone(bridge.decode(b'\xff'))
        self.assertIsNone(bridge.decode(b'x' * (bridge.MAX_PACKET+1)))


if __name__ == '__main__':
    unittest.main()
