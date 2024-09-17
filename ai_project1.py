import pygame
import random
import argparse
import math
import heapq



class PathfindingVisualizer:
    def __init__(self,gridsize):
        # Pygame 초기화
        pygame.init()

        # 기본 설정
        self.BUTTON_SIZE = (gridsize * 20, gridsize * 2)
        self.WINDOW_SIZE = (gridsize * 20 + 200, gridsize * 20 + self.BUTTON_SIZE[1])
        self.MAP_SIZE = (gridsize * 20 , gridsize * 20)
        self.GRID_SIZE = gridsize
        self.CELL_SIZE = self.MAP_SIZE[0] // self.GRID_SIZE, self.MAP_SIZE[1] // self.GRID_SIZE
        self.BACKGROUND_COLOR = (255, 255, 255)
        self.GRID_COLOR = (200, 200, 200)
        self.OBSTACLE_RATIO = 0.2
        self.font = pygame.font.Font(None, 24)  # 출발지, 도착지 글꼴
        self.start_center_y = None
        self.start_center_x = None
        self.goal_center_y = None
        self.goal_center_x = None
        self.dragging = False
        self.dragged_cell = None
        self.nowpath = None

        # Grid world 생성
        self.grid_world = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]  # 0은 빈 공간, 1은 장애물

        # Pygame 창 설정
        self.screen = pygame.display.set_mode(self.WINDOW_SIZE)
        pygame.display.set_caption("A* Algorithm")

        # 맵 화면 생성
        self.map_screen = pygame.Surface(self.MAP_SIZE)

        # 버튼 화면 생성
        self.button_screen = pygame.Surface(self.BUTTON_SIZE)

        # 라디오 버튼 리스트 초기화
        radio_button1 = Checkbox(self.MAP_SIZE[0], 50, "manhattan", 1)
        radio_button2 = Checkbox(self.MAP_SIZE[0], 100, "euclidean", 2)
        self.radio_buttons = [radio_button1, radio_button2]


    # 마우스 클릭 시 위치 읽기
    def run(self):
        # 이벤트 루프
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_button_down(event)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_button_up(event)

            self.update_display()

        pygame.quit()


    # 위치 보고 실행 
    def handle_mouse_button_down(self, event):
        if event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            
            # 라디오 버튼 클릭 시
            for radio_button in self.radio_buttons:
                radio_button.handle_click(mouse_pos)
            
            # 시작 위치 옮길 시
            if self.is_inside_point(mouse_pos, self.start_center_x, self.start_center_y):
                self.dragging = True
                self.dragged_cell = 's'
                
            # 도착 위치 옮길 시
            elif self.is_inside_point(mouse_pos, self.goal_center_x, self.goal_center_y):
                self.dragging = True
                self.dragged_cell = 'g'
                
            # 장애물 클릭해서 설정 시
            elif 0 <= mouse_pos[1] < self.MAP_SIZE[1]:
                self.place_obstacle_with_mouse(mouse_pos)
                
                
            # 버튼 클릭 시
            else:
                self.handle_button_click(mouse_pos)
    
    
    # 마우스로 시작, 도착 위치 드래그
    # s,g 인지 확인
    def handle_mouse_motion(self, event):
        if self.dragging:
            # 마우스 이벤트가 발생한 위치가 속한 셀을 식별
            grid_x = event.pos[0] // self.CELL_SIZE[0]
            grid_y = event.pos[1] // self.CELL_SIZE[1]

            # 해당 셀의 중심 좌표를 계산
            center_x = grid_x * self.CELL_SIZE[0] + (self.CELL_SIZE[0] // 2)
            center_y = grid_y * self.CELL_SIZE[1] + (self.CELL_SIZE[1] // 2)

            if self.dragged_cell == 's':
                # 시작 위치의 중심을 해당 좌표로 설정
                self.start_center_x, self.start_center_y = center_x, center_y
            elif self.dragged_cell == 'g':
                # 도착 위치의 중심을 해당 좌표로 설정
                self.goal_center_x, self.goal_center_y = center_x, center_y
                
                
    # 드래그 끝나면 이동을 정지 
    def handle_mouse_button_up(self, event):
        self.dragging = False
        self.dragged_cell = None
        
    # s,g 위치를 눌렀는 지 확인
    def is_inside_point(self, pos, center_x, center_y):
        cell_x = center_x - (self.CELL_SIZE[0] // 2) if center_x is not None else 0
        cell_y = center_y - (self.CELL_SIZE[1] // 2) if center_y is not None else 0
        return (pos[0] >= cell_x and pos[0] <= cell_x + self.CELL_SIZE[0]) and \
               (pos[1] >= cell_y and pos[1] <= cell_y + self.CELL_SIZE[1])
               
    # 장애물 바꾸기
    def place_obstacle_with_mouse(self, pos):
        x, y = pos
        grid_x = x // (self.MAP_SIZE[0] // self.GRID_SIZE)
        grid_y = y // (self.MAP_SIZE[1] // self.GRID_SIZE)
        if 0 <= grid_x < self.GRID_SIZE and 0 <= grid_y < self.GRID_SIZE:
            if self.grid_world[grid_y][grid_x] == 0:
                self.grid_world[grid_y][grid_x] = 1
            else:
                self.grid_world[grid_y][grid_x] = 0



    # 버튼 1,2,3 실행
    def handle_button_click(self, pos):
        button_width = self.MAP_SIZE[0] // 3

        if 0 <= pos[0] < button_width and pos[1] >= self.MAP_SIZE[1]:
            self.button1_action()

        elif button_width <= pos[0] < button_width * 2 and pos[1] >= self.MAP_SIZE[1]:
            self.button2_action()
            
        elif button_width * 2 <= pos[0] < button_width * 3 and pos[1] >= self.MAP_SIZE[1]:
            self.button3_action()
            

    def draw_lines(self, path):
        if path is not None:
            pygame.draw.lines(self.map_screen, (255,255,0), False , path, 3)

    def update_display(self, path = None):
        if path is not None :
            self.nowpath = path
        # 맵 화면 업데이트
        self.map_screen.fill(self.BACKGROUND_COLOR)
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(x * self.CELL_SIZE[0], y * self.CELL_SIZE[1], self.CELL_SIZE[0], self.CELL_SIZE[1])
                if self.grid_world[y][x] == 1:
                    pygame.draw.rect(self.map_screen, (100, 100, 100), rect)
                else:
                    pygame.draw.rect(self.map_screen, self.GRID_COLOR, rect, 1)

        if self.start_center_x is not None and self.start_center_y is not None:
            text_surface_start = self.font.render("s", True, (0, 0, 123))
            start_text_rect = text_surface_start.get_rect(center=(self.start_center_x, self.start_center_y))
            self.map_screen.blit(text_surface_start, start_text_rect)

        if self.goal_center_x is not None and self.goal_center_y is not None:
            text_surface_goal = self.font.render("g", True, (0, 0, 123))
            goal_text_rect = text_surface_goal.get_rect(center=(self.goal_center_x, self.goal_center_y))
            self.map_screen.blit(text_surface_goal, goal_text_rect)
        
        
        self.draw_lines(self.nowpath)
            
        # 버튼 화면 업데이트
        self.button_screen.fill((200, 200, 200))
        font = pygame.font.Font(None, 36)
        text_surface1 = font.render("Button 1", True, (0, 0, 0))
        text_surface2 = font.render("Button 2", True, (0, 0, 0))
        text_surface3 = font.render("Button 3", True, (0, 0, 0))
        self.button_screen.blit(text_surface1, (50, 20))
        self.button_screen.blit(text_surface2, (250, 20))
        self.button_screen.blit(text_surface3, (450, 20))

        # 메인 화면에 맵 화면과 버튼 화면 합성
        self.screen.fill(self.BACKGROUND_COLOR)
        self.screen.blit(self.map_screen, (0, 0))
        self.screen.blit(self.button_screen, (0, self.MAP_SIZE[1]))  # 버튼의 위치 조정
        
        # 메인 화면에 라디오 버튼도 합성
        for radio_button in self.radio_buttons:
            radio_button.draw(self.screen)
        
        
        pygame.display.flip()

    # 2번 버튼 눌러서 s,g 위치 랜덤 생성
    def place_goal_randomly(self):
        start_x = random.randint(0, self.GRID_SIZE - 1)
        start_y = random.randint(0, self.GRID_SIZE - 1)
        while True:
            goal_x = random.randint(0, self.GRID_SIZE - 1)
            goal_y = random.randint(0, self.GRID_SIZE - 1)
            if goal_x != start_x or goal_y != start_y:
                break

        self.grid_world[start_y][start_x] = 0
        self.start_center_x = start_x * self.CELL_SIZE[0] + (self.CELL_SIZE[0] // 2)
        self.start_center_y = start_y * self.CELL_SIZE[1] + (self.CELL_SIZE[1] // 2)

        self.grid_world[goal_y][goal_x] = 0
        self.goal_center_x = goal_x * self.CELL_SIZE[0] + (self.CELL_SIZE[0] // 2)
        self.goal_center_y = goal_y * self.CELL_SIZE[1] + (self.CELL_SIZE[1] // 2)
        
        pygame.display.flip()


    # 맵 장애물 랜덤 생성
    def place_obstacles_randomly(self):
        num_obstacles = int(self.GRID_SIZE * self.GRID_SIZE * self.OBSTACLE_RATIO)
        for _ in range(num_obstacles):
            x = random.randint(0, self.GRID_SIZE - 1)
            y = random.randint(0, self.GRID_SIZE - 1)
            self.grid_world[y][x] = 1
            
            
    # 맵 초기화
    def clear_map(self):
        self.grid_world = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.start_center_y = None
        self.start_center_x = None
        self.goal_center_x = None
        self.goal_center_y = None

    
    # 실행
    def start_aster(self):
        
        if self.start_center_x is not None and self.start_center_y is not None:
            start = (self.start_center_x , self.start_center_y)
        else:
            return
    
        if self.goal_center_x is not None and self.goal_center_y is not None:
            goal = (self.goal_center_x, self.goal_center_y)
        else:
            return
        
        # 맨해튼 거리 기반 A* 알고리즘 실행
        if self.radio_buttons[0].selected:
            path, closeSet = Node.astar(self.grid_world, start, goal, self.GRID_SIZE, Node.manhattan_h)
            print("정상작동")
        
        # 유클리드 거리 기반 A* 알고리즘 실행
        elif self.radio_buttons[1].selected:
            path, closeSet = Node.astar(self.grid_world, start, goal, self.GRID_SIZE, Node.euclidean_h)
            
        pygame.display.flip()
        
        # 경로가 없는 경우
        if path is None:
            print("길찾기 실패")
        else:
            print("탐색한 노드 수 :", len(closeSet))
        
        
        self.update_display(path)
            


    # 버튼 실행하는 위치
    def button1_action(self):
        self.start_aster()
        

    def button2_action(self):
        self.clear_map()
        self.place_obstacles_randomly()
        self.place_goal_randomly()
        self.nowpath = None

    def button3_action(self):
        self.clear_map()
        self.nowpath = None
        


class Node:
    def __init__(self, parent = None, position = None):
        
        self.parent = parent # 이전 노드
        self.position = position # 현재 위치
        self.g = 0 # 실제 비용
        self.h = 0 # 휴리스틱 비용
        self.f = 0 # h+g
        
        
    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    # 근처 셀의 값을 계산
    def manhattan_h(node, goal, D=1):
        dx = abs(node.position[0] - goal.position[0])
        dy = abs(node.position[1] - goal.position[1])
        return D * (dx + dy)
    
    def euclidean_h(node, goal):
        dx = abs(node.position[0] - goal.position[0])
        dy = abs(node.position[1] - node.position[1])
        return (dx**2 + dy**2)**0.5
    
    # 경로 탐색
    def astar(grid_world, start, end, gridsize, heuristic_func):
        startNode = Node(None, start)
        goalNode = Node(None, end)
        
        openList = []
        closeSet = set()
        
        heapq.heappush(openList, startNode)
        
        while openList:
            currentNode = heapq.heappop(openList)
            closeSet.add(currentNode.position)
            
            if currentNode == goalNode:
                path = []
                while currentNode is not None:
                    path.append(currentNode.position)
                    closeSet.add(currentNode.position)
                    currentNode = currentNode.parent
                return path[::-1], closeSet
            
            # 인근 셀 거리 확인
            for newPosition in [(0, -20), (0, 20), (-20, 0), (20, 0), (-20, -20), (-20, 20), (20, -20), (20, 20)]:
                nodePosition = (currentNode.position[0] + newPosition[0], currentNode.position[1] + newPosition[1])
                newIndex = nodePosition[0] // 20, nodePosition[1] // 20
                
                # 범위 내의 셀인지 확인
                if nodePosition[0] > (gridsize * 20) or nodePosition[0] < 0 or nodePosition[1] > (gridsize * 20) or nodePosition[1] < 0:
                    continue 
                
                # 장애물이 있는지 확인    
                if grid_world[newIndex[1]][newIndex[0]] == 1:
                    continue
                
                # 중복된 값인지 확인    
                if nodePosition in closeSet:
                    continue
                    
                new_node = Node(currentNode, nodePosition)
                
                # g = 실제 비용
                new_node.g = currentNode.g + 1
                # h = 휴리스틱 비용
                new_node.h = heuristic_func(new_node, goalNode)
                # f = g + h 로 최단 거리를 찾음
                new_node.f = new_node.g + new_node.h
                
                # g값이 더 작은 경우를 선택
                if any(child for child in openList if new_node == child and new_node.g > child.g):
                    continue
                    
                heapq.heappush(openList, new_node)
        
        return None





# 라디오 버튼 클래스
class Checkbox:
    def __init__(self, x, y, text, button_id):
        self.x = x
        self.y = y
        self.text = text
        self.font = pygame.font.Font(None, 20)
        self.selected = False
        self.button_id = button_id
        
        
        
    def draw(self, screen):
        self.draw_radio_circle(screen)
        text_surface = self.font.render(self.text, True, (0, 0, 0))
        screen.blit(text_surface, (self.x + 20, self.y))

    def draw_radio_circle(self, screen):
        pygame.draw.circle(screen, (0, 0, 0), (self.x + 10, self.y + 10), 10, 2)
        
        if self.selected:
            pygame.draw.circle(screen, (0, 0, 0), (self.x + 10, self.y + 10), 7)
            
            
    def handle_click(self, pos):
        
        # 클릭된 위치와 원의 중심 좌표 간의 거리 계산
        distance = math.sqrt((pos[0] - (self.x + 10)) ** 2 + (pos[1] - (self.y + 10)) ** 2)
        # 원의 반지름보다 작거나 같으면 클릭된 것으로 간주
        if distance <= 10:
            self.selected = True
            for button in visualizer.radio_buttons:
                if button != self:
                    button.selected = False
                
                    
            visualizer.update_display()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pathfinding Visualizer')
    parser.add_argument('--grid_size', type=int, default=30, help='Size of the grid (default: 30)')
    args = parser.parse_args()

    visualizer = PathfindingVisualizer(args.grid_size)
    visualizer.run()