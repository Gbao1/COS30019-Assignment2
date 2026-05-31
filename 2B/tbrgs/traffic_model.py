from __future__ import annotations

import math


def speed_from_flow(
    flow_veh_per_hour: float,
    speed_limit_kmh: float = 60.0,
    assume_under_capacity: bool = True,
) -> float:
    # flow = -1.4648375*v^2 + 93.75*v
    a = 1.4648375
    b = -93.75
    c = float(flow_veh_per_hour)
    disc = b * b - 4.0 * a * c
    if disc < 0:
        # Out-of-domain flows are clamped to very low speed.
        return 5.0

    root = math.sqrt(disc)
    v1 = (-b - root) / (2.0 * a)
    v2 = (-b + root) / (2.0 * a)

    candidates = [v for v in (v1, v2) if v > 0]
    if not candidates:
        return 5.0

    if assume_under_capacity:
        speed = max(candidates)
    else:
        speed = min(candidates)
    return min(speed, speed_limit_kmh)


def travel_time_seconds(
    distance_km: float,
    flow_veh_per_hour: float,
    speed_limit_kmh: float = 60.0,
    intersection_delay_seconds: float = 30.0,
    assume_under_capacity: bool = True,
) -> float:
    speed = speed_from_flow(
        flow_veh_per_hour=flow_veh_per_hour,
        speed_limit_kmh=speed_limit_kmh,
        assume_under_capacity=assume_under_capacity,
    )
    link_seconds = (distance_km / max(speed, 1e-6)) * 3600.0
    return link_seconds + intersection_delay_seconds
