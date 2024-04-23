import sys
import collections
import numpy as np
import heapq
import time
import math
import numpy as np
global posWalls, posGoals

class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    nodes_expanded = 0  # Biến đếm số nút đã mở ra
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        nodes_expanded += 1  # Tăng biến đếm số nút đã mở ra lên 1
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp, nodes_expanded

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)    # Lấy ra các vị trí bắt đầu của Boxes
    beginPlayer = PosOfPlayer(gameState)    # Lấy ra vị trí bắt đầu của player

    # startState là tuple dùng để lưu state khởi đầu
    # Bao gồm vị trí bắt đầu của player và boxes
    startState = (beginPlayer, beginBox)

    # frontier là 1 queue dùng để store states
    # Lưu các list - trong đó mỗi list sẽ đại diện cho 1 tập các states
    # đã được explore từ vị trí khởi đầu
    frontier = collections.deque([[startState]])
    exploredSet = set()      # Store các state đã được explore
    actions = collections.deque([[0]]) # actions là 1 queue dùng để store actions
    temp = []
    nodes_expanded = 0  # Biến đếm số nút đã mở ra

    # Duyệt qua các trạng thái trong frontier
    while frontier:
        # Lấy ra trạng thái đầu tiên
        node = frontier.popleft()
        # Lấy ra hành động tương ứng
        node_action = actions.popleft()
        nodes_expanded += 1  # Tăng biến đếm số nút đã mở ra lên 1

        # Nếu trạng thái hiện tại là trạng thái kết thúc
        if isEndState(node[-1][-1]):
            # Thêm hành động vào temp và thoát khỏi vòng lặp
            temp += node_action[1:]
            break

        # Nếu trạng thái chưa được duyệt qua
        if node[-1] not in exploredSet:
            # Thêm trạng thái vào exploredSet
            exploredSet.add(node[-1])
            # Duyệt qua các hành động hợp lệ
            for action in legalActions(node[-1][0], node[-1][1]):
                # Cập nhật trạng thái mới
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                # Kiểm tra xem trạng thái mới có hợp lệ hay không
                if isFailed(newPosBox):
                    continue
                 # Thêm trạng thái mới vào frontier và cập nhật hành động
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp, nodes_expanded

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])


def get_cost(actions):
    #change cost function into calculate each step as minus one point
    return actions.count('l') + actions.count('r') + actions.count('u') + actions.count('d')


def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState)         # Lấy ra các vị trí bắt đầu của Boxes
    beginPlayer = PosOfPlayer(gameState)    # Lấy ra vị trí bắt đầu của player

    # startState là tuple dùng để lưu state khởi đầu
    # Bao gồm vị trí bắt đầu của player và boxes
    startState = (beginPlayer, beginBox)
    frontier = PriorityQueue() #Khai báo hàng đợi kiểu PriorityQueue
    frontier.push([startState], 0)
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], 0)
    temp = []       # list lưu các actions để đến được goal (Kết thúc game)
    nodes_expanded = 0  # Biến đếm số nút đã mở ra 

    while frontier: #trong khi vẫn tồn tại hành động trong hàng đợi
        node = frontier.pop()       #lấy ra trạng thái node từ hàng đợi
        node_action = actions.pop() #lấy ra một hành động hợp lệ từ không gian hành động
        nodes_expanded += 1  # Tăng biến đếm số nút đã mở ra lên 1
        #nếu trò chơi đã kết thúc, lưu toàn bộ quá trình vào danh sách tạm và thoát khỏi vòng lặp
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break

        # Nếu trạng thái chưa được duyệt qua
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])   #Thêm trạng thái vào exploredSet

            #Duyệt qua các hành động hợp lệ
            for action in legalActions(node[-1][0], node[-1][1]):
                #Cập nhật trạng thái mới
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)

                #nếu thuật toán kết thúc thì quay lại vòng lặp
                if isFailed(newPosBox):
                    continue

                #or else
                #đưa trạng thái node mới vào hàng đợi
                frontier.push(node + [(newPosPlayer, newPosBox)], get_cost(action))
                #đưa hành động vào hàng đợi ưu tiên và lấy chi phí là số ưu tiên
                actions.push(node_action + [action[-1]], get_cost(action))
    return temp, nodes_expanded

def heuristic(posPlayer, posBox):
    # print(posPlayer, posBox)
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0    # Khởi tạo biến lưu trữ tổng khoảng cách

    completes = set(posGoals) & set(posBox) # Xác định các hộp đã hoàn thành (hộp đã ở vị trí mục tiêu)
    sortposBox = list(set(posBox).difference(completes)) # Tách các hộp còn lại và vị trí mục tiêu tương ứng
    sortposGoals = list(set(posGoals).difference(completes))

    # Tính toán khoảng cách Manhattan cho mỗi cặp hộp-mục tiêu còn lại
    for i in range(len(sortposBox)):
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance # Trả về tổng khoảng cách Manhattan như ước tính heuristic



def aStarSearch(gameState):
    beginBox = PosOfBoxes(gameState) #beginBox chứa các cặp vị trí tương ứng của các hộp có trong màn chơi
    beginPlayer = PosOfPlayer(gameState) #beginPlayer chứa cặp vị trí tương ứng của người chơi trong màn

    startState = (beginPlayer, beginBox) #startState là tuple chứa vị trí khởi điểm của người chơi và các hộp
    frontier = PriorityQueue() #Khai báo hàng đợi kiểu PriorityQueue
    frontier.push([startState], heuristic(beginPlayer, beginBox)) #Thêm vào hàng đợi vị trí khởi đầu với Priority là giá trị tính toán của hàm ước lượng khoảng cách heuristic
    exploredSet = set() #Khai báo exploredSet để tránh lập lại những nước đi cũ
    actions = PriorityQueue() #Khai báo actions kiểu PriorityQueue
    actions.push([0], heuristic(beginPlayer, beginBox)) #Thêm vào action tương ứng với vị trí người chơi hiện tại là 0 vào actions với Priority tương ứng với vị trí như trên hàng đợi
    temp = [] #temp sẽ lưu trữ những bước đi đúng để trả về cuối cùng
    nodes_expanded = 0  # Biến đếm số nút đã mở ra

    while frontier: #Khi mà hàng đợi vẫn còn chứa node
        node = frontier.pop() #Sử dụng hàm pop để lấy node với giá trị Priority thấp nhất ra
        node_action = actions.pop() #Lấy action của node tương ứng ra
        nodes_expanded += 1  # Tăng biến đếm số nút đã mở ra lên 1
        if isEndState(node[-1][-1]): #Nếu như đây là EndState thì sẽ 
            temp += node_action[1:]  #Lấy tất cả những hành động của node kể từ sau vị trí đầu tiên ra           
            break #Và ngưng vòng lập
        if node[-1] not in exploredSet: #Nếu như node được thêm vào cuối cùng chưa được explored thì
            exploredSet.add(node[-1]) #Thêm node đó vào exploredSet 
            Cost = cost(node_action[1:]) #Giá trị cost được thêm vào sẽ được tính trong hàm cost với những hành động tương ứng kể từ sau bước đi đầu tiên
            for action in legalActions(node[-1][0],node[-1][-1]): #Với mỗi action có thể đi tương ứng được trả về từ hàm legalActions với tham số là vị trí của người chơi và hộp ở bước đi mới nhất
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) #Update vị trí mới nhất của người chơi và hộp
                if isFailed(node[-1][1]): #Nếu như vị trí của hộp có khả năng fail thì bỏ qua bước đi đó
                    continue
                
                Heuristic = heuristic(newPosPlayer, newPosBox) #Tính giá trị ước lượng Heuristic với vị trí hộp và người chơi mới
                frontier.push(node + [(newPosPlayer, newPosBox)], Heuristic + Cost) #Thêm vào hàng đợi vị trí mới của hộp và người chơi với Priority bằng giá trị ước lượng Heuristic cộng với giá trị thực tế đạt được trước khi đi bước đi đó Cost
                actions.push(node_action + [action[-1]], Heuristic + Cost) #Thêm vào hành động tương ứng với bước đi đạt được của vị trí mới với Priority bằng với Priority đã tính phía trên
    return temp, nodes_expanded  # Trả về đường đi để đưa node về vị trí của nó và số nút đã mở ra



"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    nodes_expanded = 0  # Biến đếm số nút đã mở ra

    if method == 'dfs':
        result, nodes_expanded = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result, nodes_expanded = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result, nodes_expanded = uniformCostSearch(gameState)
    elif method == 'astar':
        result, nodes_expanded = aStarSearch(gameState)        
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print('Nodes expanded in %s: %d' % (method, nodes_expanded))  # In ra số nút đã mở ra
    print(result)
    return result