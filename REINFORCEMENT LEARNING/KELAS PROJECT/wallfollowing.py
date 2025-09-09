from pyamaze import maze, agent, COLOR

def rotate_clockwise():
    global orientation
    keys = list(orientation.keys())
    values = list(orientation.values())
    rotated_values = [values[-1]] + values[:-1]
    orientation = dict(zip(keys, rotated_values))

def rotate_counterclockwise():
    global orientation
    keys = list(orientation.keys())
    values = list(orientation.values())
    rotated_values = values[1:] + [values[0]]
    orientation = dict(zip(keys, rotated_values))

def move_forward(cell):
    if orientation['forward'] == 'E':
        return (cell[0], cell[1] + 1), 'E'
    if orientation['forward'] == 'W':
        return (cell[0], cell[1] - 1), 'W'
    if orientation['forward'] == 'N':
        return (cell[0] - 1, cell[1]), 'N'
    if orientation['forward'] == 'S':
        return (cell[0] + 1, cell[1]), 'S'

def wall_following(m):
    global orientation
    orientation = {'forward': 'N', 'left': 'W', 'back': 'S', 'right': 'E'}
    current_cell = (m.rows, m.cols)
    path = ''
    print(path)
    while True:
        if current_cell == (1, 1):
            break
        if m.maze_map[current_cell][orientation['left']] == 0:
            if m.maze_map[current_cell][orientation['forward']] == 0:
                rotate_clockwise()
            else:
                current_cell, d = move_forward(current_cell)
                path += d
        else:
            rotate_counterclockwise()
            current_cell, d = move_forward(current_cell)
            path += d
    simplified_path = path
    while 'EW' in simplified_path or 'WE' in simplified_path or 'NS' in simplified_path or 'SN' in simplified_path:
        simplified_path = simplified_path.replace('EW', '')
        simplified_path = simplified_path.replace('WE', '')
        simplified_path = simplified_path.replace('NS', '')
        simplified_path = simplified_path.replace('SN', '')
    return path, simplified_path

if __name__ == '__main__':
    my_maze = maze(20, 20)
    my_maze.CreateMaze(theme=COLOR.red)

    my_agent = agent(my_maze, shape='arrow', color=COLOR.dark, footprints=True)
    path, simplified_path = wall_following(my_maze)
    my_maze.tracePath({my_agent: simplified_path})

    # print(path)
    my_maze.run()
