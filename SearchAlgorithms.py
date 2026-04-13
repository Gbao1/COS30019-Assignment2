from collections import deque
from Node import Node 
from Graph import Graph
from SearchNodes import SearchNode
import heapq
import math

class SearchAlgorithms:
    def __init__(self, graph):
        self.graph = graph

    def heuristic(self, node):
        # Euclidean distance to the nearest goal. uses pythag to calculate straigt line to goal
        return min(math.sqrt((node.x - g.x)**2 + (node.y - g.y)**2) #min returns smallest value out of all goal straight lines
                   for g in self.graph.goals)

    def bfs(self): #Breath-first search
       
        root = SearchNode(self.graph.origin, None, 0)
        frontier = deque([root])
        visited = {self.graph.origin.id} #to avoid infinate loops 
        nodes_created = 1
    
        while frontier:
        # Expand the oldest node (chronological order)
            current_node = frontier.popleft() #poplast added 
        
        # Check if we reached any of the destinations
            if any(current_node.state.id == goal.id for goal in self.graph.goals):
                return current_node.state.id, nodes_created, current_node.get_path()
        
        # Expand neighbors)
            for neighbor, cost in current_node.state.neighbors:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    frontier.append(SearchNode(neighbor, current_node, current_node.cost + cost))
                    nodes_created += 1
            
        return None

    def dfs(self): #Depth-First Search
        root = SearchNode(self.graph.origin, None, 0)
        frontier = [root] 
        visited = {self.graph.origin.id}
        nodes_created = 1
        
        while frontier:
            current_node = frontier.pop() #LIFO stack
            
            if any(current_node.state.id == goal.id for goal in self.graph.goals):
                return current_node.state.id, nodes_created, current_node.get_path()
            
            # Reversed for stack
            for neighbor, cost in reversed(current_node.state.neighbors):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    new_node = SearchNode(neighbor, current_node, current_node.cost + cost)
                    frontier.append(new_node)
                    nodes_created += 1
        return None

    def a_star(self):
        #full cost zero and is cost to cuurent stop + cost to goal)
        _origin = self.heuristic(self.graph.origin)
        root = SearchNode(self.graph.origin, None, 0, _origin)
        frontier = [root]
        heapq.heapify(frontier) #Had to search for this one online took a while to understand how to use
        
        visited_costs = {self.graph.origin.id: 0}
        nodes_created = 1
        
        while frontier:
            current_node = heapq.heappop(frontier)
            
            if any(current_node.state.id == goal.id for goal in self.graph.goals):
                return current_node.state.id, nodes_created, current_node.get_path()
            
            for neighbor, cost in current_node.state.neighbors:
                new_g = current_node.cost + cost #new_g is total cost to get to this point
                
                if neighbor.id not in visited_costs or new_g < visited_costs[neighbor.id]:
                    visited_costs[neighbor.id] = new_g
                    new_f = new_g + self.heuristic(neighbor)
                    new_node = SearchNode(neighbor, current_node, new_g, new_f)
                    heapq.heappush(frontier, new_node)
                    nodes_created += 1
        return None
    
    def gbfs(self):
        if not self.graph.origin: return None
        
        # Priority is just the immediate cost value
        root = SearchNode(self.graph.origin, None, 0, self.heuristic(self.graph.origin))
        frontier = [root]
        heapq.heapify(frontier)
        
        visited = {self.graph.origin.id}
        nodes_created = 1
        
        while frontier:
            current_node = heapq.heappop(frontier)
            
            if any(current_node.state.id == goal.id for goal in self.graph.goals):
                return current_node.state.id, nodes_created, current_node.get_path()
            
            for neighbor, cost in current_node.state.neighbors:
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    # Use only heuristic for cost
                    h_val = self.heuristic(neighbor)
                    new_node = SearchNode(neighbor, current_node, current_node.cost + cost, h_val)
                    heapq.heappush(frontier, new_node)
                    nodes_created += 1
        return None

    def ucs(self):
        if not self.graph.origin: return None
        
        # Priority is just the path cost
        root = SearchNode(self.graph.origin, None, 0, 0)
        frontier = [root]
        heapq.heapify(frontier)
        
        visited_costs = {self.graph.origin.id: 0}
        nodes_created = 1
        
        while frontier:
            current_node = heapq.heappop(frontier)
            
            if any(current_node.state.id == goal.id for goal in self.graph.goals):
                return current_node.state.id, nodes_created, current_node.get_path()
            
            for neighbor, cost in current_node.state.neighbors:
                new_g = current_node.cost + cost
                
                if neighbor.id not in visited_costs or new_g < visited_costs[neighbor.id]:
                    visited_costs[neighbor.id] = new_g
                    # f_cost is just g_cost here
                    new_node = SearchNode(neighbor, current_node, new_g, new_g)
                    heapq.heappush(frontier, new_node)
                    nodes_created += 1
        return None

    def hill_climbing(self):
        current_node = self.graph.origin
        path = [str(current_node.id)]
        nodes_created = 1
        
        while True:
            if any(current_node.id == g.id for g in self.graph.goals):
                return current_node.id, nodes_created, " -> ".join(path)
            
            # Find the neighbour with the lowest distance to goal
            best_neighbour = None
            best_h = self.heuristic(current_node)
            
            for neighbor, cost in current_node.neighbors:
                nodes_created += 1
                h_val = self.heuristic(neighbor)
                if h_val < best_h:
                    best_h = h_val
                    best_neighbour = neighbor
            
            if best_neighbour is None:
                return None # Stuck
                
            current_node = best_neighbour
            path.append(str(current_node.id))