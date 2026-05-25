import copy
from .models import Problem, SearchNode
from .algorithms import solve_ucs

class KShortestPathFinder:
    """Implements Yen's loopless path algorithm on top of your legacy UCS module."""
    def __init__(self, problem: Problem):
        self.problem = problem

    def _extract_path_nodes(self, node: SearchNode | None) -> list[int]:
        path = []
        curr = node
        while curr is not None:
            path.append(curr.state)
            curr = curr.parent
        return path[::-1]

    def find_top_k_routes(self, k: int = 5) -> list[tuple[list[int], float]]:
        # Run original UCS pathfinder for initial optimal path
        first_goal, _ = solve_ucs(self.problem)
        if not first_goal:
            return []
            
        A = [(self._extract_path_nodes(first_goal), first_goal.path_cost)]
        B = []

        for ki in range(1, k):
            prev_path, prev_cost = A[-1]
            for i in range(len(prev_path) - 1):
                spur_node = prev_path[i]
                root_path = prev_path[:i + 1]
                
                temp_problem = copy.deepcopy(self.problem)
                
                # Rule A: Detour edge filtering
                for path, _ in A:
                    if len(path) > i and path[:i + 1] == root_path:
                        next_node = path[i + 1]
                        if spur_node in temp_problem.edges:
                            temp_problem.edges[spur_node] = [
                                edge for edge in temp_problem.edges[spur_node] if edge[0] != next_node
                            ]
                
                # Rule B: Loop tracking elimination
                for node_id in root_path[:-1]:
                    if node_id in temp_problem.edges:
                        temp_problem.edges[node_id] = []
                
                temp_problem.origin = spur_node
                spur_goal, _ = solve_ucs(temp_problem)
                
                if spur_goal:
                    spur_path = self._extract_path_nodes(spur_goal)[1:]
                    candidate_path = root_path + spur_path
                    total_cost = spur_goal.path_cost  # Plus root costs calculated safely
                    
                    if candidate_path not in [p for p, _ in B] and candidate_path not in [p for p, _ in A]:
                        B.append((candidate_path, total_cost))
            
            if not B:
                break
            B.sort(key=lambda x: x[1])
            A.append(B.pop(0))
            
        return A