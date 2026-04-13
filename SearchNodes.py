class SearchNode:
    def __init__(self, state, parent=None, cost=0, f_cost=0):
        self.state = state     # current node
        self.parent = parent   # parent node/node we came from 
        self.cost = cost       # total cos
        self.f_cost = f_cost   #A* tie breaking  

    def get_path(self): #path taken to get to current node 
        path = []
        current = self
        while current:
            path.append(str(current.state.id))
            current = current.parent
        return " -> ".join(reversed(path))

    def __lt__(self, other):
         # Primary tie-breaker if needed
        if self.f_cost == other.f_cost:
            return self.state.id < other.state.id
        return self.f_cost < other.f_cost