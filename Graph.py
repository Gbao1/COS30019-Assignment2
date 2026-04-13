import re
from Node import Node 

class Graph:
    def __init__(self):
        self.nodes = {}        
        self.origin = None     
        self.goals = []        

    def load_from_file(self, filename):
        try:
            with open(filename, 'r') as f:
                content = f.read()

            # 1. Parse Nodes
            nodes_found = re.findall(r"(\d+):\s*\((\d+),(\d+)\)", content)
            for n_id, x, y in nodes_found:
                self.nodes[int(n_id)] = Node(n_id, x, y)

            # 2. Parse Edges:
            edges_found = re.findall(r"\((\d+),(\d+)\):\s*(\d+)", content)
            for start, end, cost in edges_found:
                if int(start) in self.nodes and int(end) in self.nodes:
                    self.nodes[int(start)].add_neighbor(self.nodes[int(end)], cost)

            # 3. Parse Origin"
            origin_match = re.search(r"Origin:\s*(\d+)", content)
            if origin_match:
                self.origin = self.nodes[int(origin_match.group(1))]

            # 4. "Destinations: goal"
            dest_match = re.search(r"Destinations:\s*([\d\s;]+)", content)
            if dest_match:
                raw_ids = dest_match.group(1).replace(" ", "").split(";")
                self.goals = [self.nodes[int(d)] for d in raw_ids if d]

            print(f"Graph loaded: {len(self.nodes)} nodes, {len(self.goals)} goals.")
        except Exception as e:
            print(f"Error loading graph: {e}")
