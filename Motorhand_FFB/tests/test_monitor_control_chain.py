from motorhand_teleop.teleop_monitor.plots import build_pages
from motorhand_teleop.teleop_monitor.quantities import enrich,TorqueConversion


def test_default_page_shows_paired_angles_and_actual_used_contact():
    pages=build_pages(1,'driver_impedance')
    assert next(iter(pages))=='① Wuji受阻'
    first=pages[next(iter(pages))][0]
    assert {t.field for t in first.traces}=={'wuji_feedback_target_rad','wuji_feedback_position_rad'}
    assert all(t.time_field=='wuji_feedback_received_ns' and t.strict_time for t in first.traces)
    assert {t.field:t.time_field for t in first.states}=={
        'contact_active_mask':'contact_evaluated_ns','impedance_contact':'impedance_planned_ns'}


def test_chain_latency_uses_paired_feedback_and_sdk_submit_not_latest_state():
    p=enrich({'wuji_received_ns':900_000_000,'wuji_feedback_received_ns':100_000_000,
              'feedback_request_id':7,'glove_submit_ns':125_000_000},TorqueConversion.unavailable())
    assert p['feedback_chain_delay_ms']==[25.]
    p.pop('wuji_feedback_received_ns')
    assert enrich(p,TorqueConversion.unavailable())['feedback_chain_delay_ms']==[None]
