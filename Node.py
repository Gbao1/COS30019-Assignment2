import math

class Node:
    def __init__(self, node_id, x, y):
        self.id = int(node_id)
        self.x = int(x)
        self.y = int(y)
        self.neighbors = [] 

    def add_neighbor(self, neighbor_node, cost):
        self.neighbors.append((neighbor_node, int(cost)))
        self.neighbors.sort(key=lambda x: x[0].id)

    def __repr__(self):#representation of node 
        return f"Node {self.id}"

    def __lt__(self, other):
        return self.id < other.id
