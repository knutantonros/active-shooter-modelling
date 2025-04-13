from typing import Tuple, List, Dict, Any
import numpy as np
from .grid import GridCell, Obstacle, Wall

class Furniture(Obstacle):
    """Furniture obstacle that can provide cover."""
    
    def __init__(self, x: int, y: int, furniture_type: str = "desk"):
        super().__init__(x, y)
        self.name = f"{furniture_type}"
        self.furniture_type = furniture_type
        self.cover_value = self._get_cover_value(furniture_type)
    
    def _get_cover_value(self, furniture_type: str) -> float:
        """Get the cover value for this furniture type (0-1)."""
        cover_values = {
            "desk": 0.7,
            "table": 0.6,
            "chair": 0.3,
            "bookshelf": 0.8,
            "cabinet": 0.9,
            "couch": 0.5,
            "vending_machine": 0.9,
            "counter": 0.8
        }
        return cover_values.get(furniture_type, 0.5)
    
    def __repr__(self) -> str:
        return f"{self.furniture_type}({self.pos[0]}, {self.pos[1]})"


class Door(GridCell):
    """Door that connects rooms."""
    
    def __init__(self, x: int, y: int, is_open: bool = True, is_locked: bool = False):
        super().__init__(x, y)
        self.name = "door"
        self.is_open = is_open
        self.is_locked = is_locked
        self.can_pass = is_open and not is_locked
    
    def open(self):
        """Open the door if it's not locked."""
        if not self.is_locked:
            self.is_open = True
            self.can_pass = True
    
    def close(self):
        """Close the door."""
        self.is_open = False
        self.can_pass = False
    
    def lock(self):
        """Lock the door."""
        self.is_locked = True
        self.is_open = False
        self.can_pass = False
    
    def unlock(self):
        """Unlock the door."""
        self.is_locked = False
    
    def __repr__(self) -> str:
        status = "open" if self.is_open else "closed"
        if self.is_locked:
            status += " (locked)"
        return f"Door({self.pos[0]}, {self.pos[1]}, {status})"


class Window(GridCell):
    """Window that can be broken to create a passage."""
    
    def __init__(self, x: int, y: int, is_broken: bool = False):
        super().__init__(x, y)
        self.name = "window"
        self.is_broken = is_broken
        self.can_pass = is_broken
    
    def break_window(self):
        """Break the window to allow passage."""
        self.is_broken = True
        self.can_pass = True
    
    def __repr__(self) -> str:
        status = "broken" if self.is_broken else "intact"
        return f"Window({self.pos[0]}, {self.pos[1]}, {status})"


def place_furniture(grid: List[List[GridCell]], room_coords: Tuple[Tuple[int, int], Tuple[int, int]], 
                   room_type: str, rng: np.random.Generator) -> List[Furniture]:
    """Place furniture in a room based on room type.
    
    Args:
        grid: The 2D grid to modify
        room_coords: ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
        room_type: Type of room (classroom, office, etc.)
        rng: Random number generator
        
    Returns:
        List[Furniture]: List of placed furniture
    """
    (tl_x, tl_y), (br_x, br_y) = room_coords
    width = br_y - tl_y
    height = br_x - tl_x
    furniture_list = []
    
    if room_type == "classroom":
        # Add a teacher's desk at the front
        desk_x = tl_x + 1
        desk_y = tl_y + width // 2
        if grid[desk_x][desk_y] is None or isinstance(grid[desk_x][desk_y], GridCell):
            desk = Furniture(desk_x, desk_y, "desk")
            grid[desk_x][desk_y] = desk
            furniture_list.append(desk)
        
        # Add student desks in rows
        desk_spacing = 2
        for i in range(tl_x + 3, br_x - 1, desk_spacing):
            for j in range(tl_y + 2, br_y - 1, desk_spacing):
                if grid[i][j] is None or isinstance(grid[i][j], GridCell):
                    desk = Furniture(i, j, "desk")
                    grid[i][j] = desk
                    furniture_list.append(desk)
    
    elif room_type == "office":
        # Add desks and chairs in an office layout
        desk_spacing = 3
        for i in range(tl_x + 2, br_x - 1, desk_spacing):
            for j in range(tl_y + 2, br_y - 1, desk_spacing):
                if grid[i][j] is None or isinstance(grid[i][j], GridCell):
                    desk = Furniture(i, j, "desk")
                    grid[i][j] = desk
                    furniture_list.append(desk)
                
                # Add a chair next to each desk
                chair_x, chair_y = i, j + 1
                if 0 <= chair_x < len(grid) and 0 <= chair_y < len(grid[0]):
                    if grid[chair_x][chair_y] is None or isinstance(grid[chair_x][chair_y], GridCell):
                        chair = Furniture(chair_x, chair_y, "chair")
                        grid[chair_x][chair_y] = chair
                        furniture_list.append(chair)
    
    elif room_type == "cafeteria" or room_type == "food_court":
        # Add tables in a cafeteria/food court layout
        table_spacing = 4
        for i in range(tl_x + 2, br_x - 1, table_spacing):
            for j in range(tl_y + 2, br_y - 1, table_spacing):
                if grid[i][j] is None or isinstance(grid[i][j], GridCell):
                    table = Furniture(i, j, "table")
                    grid[i][j] = table
                    furniture_list.append(table)
                
                # Add chairs around each table
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    chair_x, chair_y = i + dx, j + dy
                    if 0 <= chair_x < len(grid) and 0 <= chair_y < len(grid[0]):
                        if grid[chair_x][chair_y] is None or isinstance(grid[chair_x][chair_y], GridCell):
                            chair = Furniture(chair_x, chair_y, "chair")
                            grid[chair_x][chair_y] = chair
                            furniture_list.append(chair)
    
    elif room_type == "store" or room_type == "shop":
        # Add shelves and counters in a store layout
        shelf_spacing = 3
        for i in range(tl_x + 2, br_x - 2, shelf_spacing):
            for j in range(tl_y + 2, br_y - 2, 1):
                if j % (width - 4) < (width - 6):  # Leave space for aisles
                    if grid[i][j] is None or isinstance(grid[i][j], GridCell):
                        shelf = Furniture(i, j, "bookshelf")
                        grid[i][j] = shelf
                        furniture_list.append(shelf)
        
        # Add a counter near the entrance
        counter_x = br_x - 2
        for j in range(tl_y + 2, tl_y + width // 3):
            if grid[counter_x][j] is None or isinstance(grid[counter_x][j], GridCell):
                counter = Furniture(counter_x, j, "counter")
                grid[counter_x][j] = counter
                furniture_list.append(counter)
    
    # Add random additional furniture
    num_additional = rng.integers(1, 5)
    for _ in range(num_additional):
        x = rng.integers(tl_x + 1, br_x)
        y = rng.integers(tl_y + 1, br_y)
        if grid[x][y] is None or isinstance(grid[x][y], GridCell):
            furniture_type = rng.choice(["chair", "cabinet", "bookshelf", "couch", "vending_machine"])
            furniture = Furniture(x, y, furniture_type)
            grid[x][y] = furniture
            furniture_list.append(furniture)
    
    return furniture_list


def place_doors_and_windows(grid: List[List[GridCell]], building, 
                           add_windows: bool = True, rng: np.random.Generator = None) -> Tuple[List[Door], List[Window]]:
    """Place doors and windows in the building.
    
    Args:
        grid: The 2D grid to modify
        building: Building object with room information
        add_windows: Whether to add windows to exterior walls
        rng: Random number generator
        
    Returns:
        Tuple[List[Door], List[Window]]: Lists of placed doors and windows
    """
    if rng is None:
        rng = np.random.default_rng()
    
    doors = []
    windows = []
    
    # Add doors to each room (from the room's door list)
    for room in building.rooms:
        for door_pos in room.doors:
            x, y = door_pos
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                door = Door(x, y)
                grid[x][y] = door
                doors.append(door)
    
    # Add windows on exterior walls
    if add_windows:
        width = len(grid[0])
        height = len(grid)
        
        # Check each exterior wall cell
        for i in range(height):
            for j in range(width):
                # Only check exterior walls
                if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                    # Only add windows to Wall cells and not at corners
                    if (isinstance(grid[i][j], Wall) and
                        not (i == 0 and j == 0) and
                        not (i == 0 and j == width - 1) and
                        not (i == height - 1 and j == 0) and
                        not (i == height - 1 and j == width - 1)):
                        
                        # Randomly decide whether to add a window
                        if rng.random() < 0.2:  # 20% chance for each eligible wall cell
                            window = Window(i, j)
                            grid[i][j] = window
                            windows.append(window)
    
    return doors, windows


def add_random_obstacles(grid: List[List[GridCell]], count: int, rng: np.random.Generator = None) -> List[Obstacle]:
    """Add random obstacles to the grid.
    
    Args:
        grid: The 2D grid to modify
        count: Number of obstacles to add
        rng: Random number generator
        
    Returns:
        List[Obstacle]: List of placed obstacles
    """
    if rng is None:
        rng = np.random.default_rng()
    
    height = len(grid)
    width = len(grid[0])
    obstacles = []
    
    for _ in range(count):
        # Find a random empty cell
        for attempt in range(100):  # Limit attempts to prevent infinite loop
            x = rng.integers(1, height - 1)
            y = rng.integers(1, width - 1)
            
            if grid[x][y] is None or isinstance(grid[x][y], GridCell):
                obstacle = Obstacle(x, y)
                grid[x][y] = obstacle
                obstacles.append(obstacle)
                break
    
    return obstacles