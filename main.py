import random
import heapq
import math

def generate_maze(size):
    # create a maze filled with walls
    maze = [[1 for _ in range(size)] for _ in range(size)]

    # create a path in the maze
    for i in range(1, size, 2):
        for j in range(1, size, 2):
            maze[i][j] = 0

            # randomly create paths
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
            if maze[i][j] == 0:  # only consider open cells
                neighbors = []

                # check neighboring cells
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


# generate a random maze with a grid size of 28
maze_size = 28
maze = generate_maze(maze_size)

# transform the maze into a search tree
search_tree = maze_to_search_tree(maze)

# print the maze
print("Generated Maze:")
for row in maze:
    print(' '.join(str(cell) for cell in row))

# print the search tree
print("\nGenerated Search Tree (BFS):")
for node, neighbors in search_tree.items():
    print(f"{node} -> {neighbors}")

# example usage to find a path in the generated maze
start = (1, 1)
end = (maze_size -1, maze_size -1)

# find the solution path and count nodes searched using BFS
solution_path, nodes_searched = bfs(search_tree, start, end)

print("\nResults using BFS:")
# print the solution path and nodes searched
if solution_path:
    print("Solution Path found:", solution_path)
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
            if maze[i][j] == 0:  # only consider open cells
                neighbors = []

                # check neighboring cells
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
            # reconstruct the path
            path = []
            while current in came_from:
                path.insert(0, current)
                current = came_from[current]
            return path + [goal]

        for neighbor in graph[current]:
            temp_g = g[current] + 1  # assuming uniform cost for simplicity
            if neighbor not in g or temp_g < g[neighbor]:
                g[neighbor] = temp_g
                priority = temp_g + euclidean(neighbor, goal)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current

    return None


def euclidean(node, goal):
    # euclidean distance
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)


def manhattan(node, goal):
    # manhattan distance
    # create coordinates of starting node and goal node
    startNodeX, startNodeY = node
    goalX, goalY = goal
    return abs(startNodeX-goalX) + abs(startNodeY-goalY)
    # in this example, start always = (0,0) and goal = (27,27)

# transform the maze into a search tree
search_tree = maze_to_search_tree(maze)

# example usage to find a path in the generated maze using A*
start = (1, 1)
end = (maze_size - 1, maze_size - 1)

print("\nGenerated Search Tree (A*):")
for node, neighbors in search_tree.items():
    print(f"{node} -> {neighbors}")

# find the solution path using A*
solution_path = a_star(search_tree, start, end)

# print the solution path
print("\nResults using A*:")

if solution_path:
    print("Solution Path found:", solution_path)
    print("Path Length:", len(solution_path) - 1)
    nodes_searched = a_star(search_tree, start, end)
    print("Total Nodes Searched:", len(nodes_searched))
else:
    print("\nNo solution path found.")

# find the solution path and count nodes searched using A*
print("Result of Euclidean equation =", euclidean(start, end))
print("Result of Manhattan equation =", manhattan(start, end))
