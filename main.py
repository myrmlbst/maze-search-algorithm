import random
import heapq
import math

def generate_maze(size):
    # Create a maze filled with walls
    maze = [[1 for _ in range(size)] for _ in range(size)]

    # Create a path in the maze
    for i in range(1, size, 2):
        for j in range(1, size, 2):
            maze[i][j] = 0

            # Randomly create paths
            if i < size - 2 and (j == 1 or (j > 1 and bool(random.getrandbits(1)))):
                maze[i + 1][j] = 0
            elif j < size - 2:
                maze[i][j + 1] = 0

    return maze

def maze_to_search_tree(maze):
    search_tree = {}
    rows, cols = len(maze), len(maze[0])

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 0:  # Only consider open cells
                neighbors = []

                # Check neighboring cells
                if i > 0 and maze[i - 1][j] == 0:
                    neighbors.append((i - 1, j))
                if i < rows - 1 and maze[i + 1][j] == 0:
                    neighbors.append((i + 1, j))
                if j > 0 and maze[i][j - 1] == 0:
                    neighbors.append((i, j - 1))
                if j < cols - 1 and maze[i][j + 1] == 0:
                    neighbors.append((i, j + 1))

                search_tree[(i, j)] = neighbors

    return search_tree

def bfs(graph, start, end):
    visited = set()
    queue = [[start]]
    nodes_searched = 0

    if start == end:
        return "Start node is the same as the end node."

    while queue:
        path = queue.pop(0)
        node = path[-1]
        nodes_searched += 1

        if node not in visited:
            neighbors = graph[node]
            for neighbor in neighbors:
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

                if neighbor == end:
                    return new_path, nodes_searched

            visited.add(node)

    return None, nodes_searched

# Generate a random maze with a grid size of 28
maze_size = 28
maze = generate_maze(maze_size)

# Transform the maze into a search tree
search_tree = maze_to_search_tree(maze)

# Print the maze
print("Generated Maze:")
for row in maze:
    print(' '.join(str(cell) for cell in row))

# Print the search tree
print("\nGenerated Search Tree (BFS):")
for node, neighbors in search_tree.items():
    print(f"{node} -> {neighbors}")

# Example usage to find a path in the generated maze
start = (1, 1)
end = (maze_size -1, maze_size -1)

# Find the solution path and count nodes searched using BFS
solution_path, nodes_searched = bfs(search_tree, start, end)

# Print the solution path and nodes searched
if solution_path:
    print("\nSolution Path found:", solution_path)
    print("Path Length:", len(solution_path) - 1)
    print("Total Nodes Searched:", nodes_searched)
else:
    print("\nNo solution path found.")

print("\n\n\nFor A* Algorithm:")

def maze_to_search_tree(maze):
    search_tree = {}
    rows, cols = len(maze), len(maze[0])

    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 0:  # Only consider open cells
                neighbors = []

                # Check neighboring cells
                if i > 0 and maze[i - 1][j] == 0:
                    neighbors.append((i - 1, j))
                if i < rows - 1 and maze[i + 1][j] == 0:
                    neighbors.append((i + 1, j))
                if j > 0 and maze[i][j - 1] == 0:
                    neighbors.append((i, j - 1))
                if j < cols - 1 and maze[i][j + 1] == 0:
                    neighbors.append((i, j + 1))

                search_tree[(i, j)] = neighbors

    return search_tree

def a_star(graph, start, goal):
    open_list = [(0, start)]
    g = {start: 0}
    came_from = {}

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.insert(0, current)
                current = came_from[current]
            return path + [goal]

        for neighbor in graph[current]:
            temp_g = g[current] + 1  # Assuming uniform cost for simplicity
            if neighbor not in g or temp_g < g[neighbor]:
                g[neighbor] = temp_g
                priority = temp_g + euclidean(neighbor, goal)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current

    return None

def euclidean(node, goal):
    # Euclidean distance
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

#def manhattan(node, goal):
    # Manhattan distance
   # return sum(abs(node-goal))

# Transform the maze into a search tree
search_tree = maze_to_search_tree(maze)

# Example usage to find a path in the generated maze using A*
start = (1, 1)
end = (maze_size - 1, maze_size - 1)

print("\nGenerated Search Tree:")
for node, neighbors in search_tree.items():
    print(f"{node} -> {neighbors}")

# Find the solution path using A*
solution_path = a_star(search_tree, start, end)

# Print the solution path
if solution_path:
    print("\nSolution Path found:", solution_path)
    print("Path Length:", len(solution_path) - 1)
else:
    print("\nNo solution path found.")

# Find the solution path and count nodes searched using A*
print("Result of Euclidean equation =", euclidean(start, end))
#m