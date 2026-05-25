from __future__ import annotations

from tbrgs.traffic_model import speed_from_flow, travel_time_seconds


def test_speed_from_flow_is_capped_at_limit() -> None:
    speed = speed_from_flow(flow_veh_per_hour=100.0, speed_limit_kmh=60.0)
    assert speed <= 60.0


def test_speed_from_flow_positive_at_peak_domain() -> None:
    speed = speed_from_flow(flow_veh_per_hour=1500.0, speed_limit_kmh=60.0)
    assert speed > 0.0


def test_speed_from_flow_under_capacity_is_higher_than_over_capacity() -> None:
    under = speed_from_flow(flow_veh_per_hour=1200.0, assume_under_capacity=True)
    over = speed_from_flow(flow_veh_per_hour=1200.0, assume_under_capacity=False)
    assert under >= over


def test_travel_time_includes_intersection_delay() -> None:
    time_s = travel_time_seconds(distance_km=1.0, flow_veh_per_hour=500.0, intersection_delay_seconds=30.0)
    assert time_s > 30.0


def test_travel_time_increases_with_distance() -> None:
    t1 = travel_time_seconds(distance_km=0.5, flow_veh_per_hour=800.0)
    t2 = travel_time_seconds(distance_km=2.0, flow_veh_per_hour=800.0)
    assert t2 > t1
