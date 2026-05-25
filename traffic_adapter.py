import copy
import math
from .models import Problem, NodeId

class TrafficAdapter:
    """
    Acts as an isolation bridge between A2A pathfinders and ML data maps.
    Converts static distances into dynamic travel times without modifying original data structs.
    """
    def __init__(self, base_problem: Problem):
        self.base_problem = base_problem

    def _convert_flow_to_time(self, flow_per_hour: float, distance_km: float) -> float:
        A = -1.4648375
        B = 93.75
        
        #rule from assignment formula document
        if flow_per_hour <= 351.0:
            speed = 60.0
        else:
            discriminant = (B ** 2) + (4 * A * flow_per_hour)
            if discriminant < 0:
                speed = 32.0  # Fallback under extreme congestion values
            else:
                speed = (-B - math.sqrt(discriminant)) / (2 * A)

        # Formula: time = (distance / speed) in seconds + 30 sec intersection penalty
        travel_time_seconds = (distance_km / speed) * 3600 + 30.0
        return travel_time_seconds

    def generate_timed_problem(self, predictions_map: dict) -> Problem:
        """
        Returns a fresh deepcopy of the Problem model containing 
        travel times as edge costs instead of kilometres.
        """
        timed_problem = copy.deepcopy(self.base_problem)
        new_edges = {}

        for src, destinations in self.base_problem.edges.items():
            new_edges[src] = []
            for dst, distance_km in destinations:
                # Retrieve the forecasted traffic link weight (defaulting to zero congestion)
                predicted_flow = predictions_map.get((src, dst), 0.0)
                travel_time_cost = self._convert_flow_to_time(predicted_flow, distance_km)
                new_edges[src].append((dst, travel_time_cost))
                
        timed_problem.edges = new_edges
        return timed_problem