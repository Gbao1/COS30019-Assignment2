from Graph import Graph
import sys
from SearchAlgorithms import SearchAlgorithms
def main():

    #if len(sys.argv) < 3:
     #   print("Usage: python search.py <PathFinder-test> <BFS>")
      #  return
    filename = "PathFinder-test.txt" #sys.argv[1]

    my_graph = Graph()
    my_graph.load_from_file(filename)
    
    if not my_graph.origin:
        print("No origin found. Check file.")
        return

          
    searcher = SearchAlgorithms(my_graph)
    algorithms = ["BFS", "DFS", "AS", "GBFS", "UCS", "HCLB"]
    print("-" * 15)
    #algorithm selection Currently looped

    for method in algorithms:
        if method == "BFS":
            result = searcher.bfs()
        elif method == "DFS":
            result = searcher.dfs()
        elif method == "AS":
            result = searcher.a_star()
        elif method == "GBFS":
            result = searcher.gbfs()
        elif method == "UCS":
            result = searcher.ucs()
        elif method == "HCLB":
            result = searcher.hill_climbing()

        if result:
            goal_id, nodes_count, path_string = result
            print(f"{filename} {method}")
            print(f"Goal: {goal_id} -- Nodes Checked:{nodes_count} ")
            print(path_string)
           
        else:
            print(f"{filename} {method}")
            print("No path found.")
        print("-" * 15)


if __name__ == "__main__":
    main()

