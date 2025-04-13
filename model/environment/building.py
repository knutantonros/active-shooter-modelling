import numpy as np
from typing import Tuple, List, Optional, Dict
from .grid import GridCell, Wall, Obstacle, Exit

class Room:
    """A room in the building layout."""
    
    def __init__(self, id: int, top_left: Tuple[int, int], bottom_right: Tuple[int, int], room_type: str = "classroom"):
        self.id = id
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.room_type = room_type
        self.width = bottom_right[1] - top_left[1]
        self.height = bottom_right[0] - top_left[0]
        self.area = self.width * self.height
        self.doors = []  # List of door positions
    
    def add_door(self, door_pos: Tuple[int, int]):
        """Add a door to the room."""
        self.doors.append(door_pos)
    
    def is_inside(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is inside the room."""
        x, y = pos
        return (self.top_left[0] <= x <= self.bottom_right[0] and 
                self.top_left[1] <= y <= self.bottom_right[1])
    
    def get_center(self) -> Tuple[int, int]:
        """Get the center coordinates of the room."""
        return (
            self.top_left[0] + self.height // 2,
            self.top_left[1] + self.width // 2
        )
    
    def __repr__(self) -> str:
        return f"Room(id={self.id}, type={self.room_type}, tl={self.top_left}, br={self.bottom_right})"


class Building:
    """A building with rooms, hallways, and other features."""
    
    def __init__(self, width: int, height: int, name: str = "unnamed"):
        self.width = width
        self.height = height
        self.name = name
        self.rooms = []
        self.hallways = []
        self.exits = []
    
    def add_room(self, room: Room):
        """Add a room to the building."""
        self.rooms.append(room)
    
    def add_hallway(self, top_left: Tuple[int, int], bottom_right: Tuple[int, int]):
        """Add a hallway to the building."""
        hallway = Room(len(self.rooms) + len(self.hallways), top_left, bottom_right, "hallway")
        self.hallways.append(hallway)
    
    def add_exit(self, pos: Tuple[int, int]):
        """Add an exit to the building."""
        self.exits.append(pos)
    
    def get_room_at(self, pos: Tuple[int, int]) -> Optional[Room]:
        """Get the room at a specific position."""
        for room in self.rooms:
            if room.is_inside(pos):
                return room
                
        for hallway in self.hallways:
            if hallway.is_inside(pos):
                return hallway
        
        return None
    
    def __repr__(self) -> str:
        return f"Building(name={self.name}, width={self.width}, height={self.height}, rooms={len(self.rooms)}, hallways={len(self.hallways)}, exits={len(self.exits)})"


def generate_school_layout(width: int, height: int, grid: List[List[GridCell]], seed: int = 42) -> Building:
    """Generate a school layout.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        grid: 2D grid to modify with walls, exits, etc.
        seed: Random seed for generation
        
    Returns:
        Building: Building object with room information
    """
    # Initialize random generator
    rng = np.random.default_rng(seed=seed)
    
    # Create a new building
    building = Building(width, height, "school")
    
    # Create hallways (cross shape)
    h_center = height // 2
    w_center = width // 2
    hallway_width = max(3, min(height, width) // 10)
    
    # Horizontal hallway
    building.add_hallway(
        (h_center - hallway_width // 2, 0),
        (h_center + hallway_width // 2, width - 1)
    )
    
    # Vertical hallway
    building.add_hallway(
        (0, w_center - hallway_width // 2),
        (height - 1, w_center + hallway_width // 2)
    )
    
    # Add walls around the layout
    for i in range(height):
        for j in range(width):
            # Skip hallway areas
            if ((h_center - hallway_width // 2 <= i <= h_center + hallway_width // 2) or
                (w_center - hallway_width // 2 <= j <= w_center + hallway_width // 2)):
                continue
                
            # Add wall if this is on the edge of the building
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                grid[i][j] = Wall(i, j)
    
    # Add exits at the ends of hallways
    building.add_exit((h_center, 0))  # Left exit
    building.add_exit((h_center, width - 1))  # Right exit
    building.add_exit((0, w_center))  # Top exit
    building.add_exit((height - 1, w_center))  # Bottom exit
    
    # Place exits in the grid
    for exit_pos in building.exits:
        x, y = exit_pos
        grid[x][y] = Exit(x, y)
    
    # Add classrooms on both sides of hallways
    room_id = 0
    
    # Above horizontal hallway
    row_height = (h_center - hallway_width // 2) // 2
    if row_height > 3:
        for i in range(2):
            for j in range(3):
                room_width = (width - 2) // 3
                tl_x = i * row_height + (1 if i > 0 else 0)
                tl_y = j * room_width + 1
                br_x = tl_x + row_height - 1
                br_y = tl_y + room_width - 1
                
                # Don't create rooms in the vertical hallway
                if not (tl_y <= w_center <= br_y and tl_x <= h_center <= br_x):
                    room = Room(room_id, (tl_x, tl_y), (br_x, br_y), "classroom")
                    building.add_room(room)
                    
                    # Add walls around the room
                    for x in range(tl_x, br_x + 1):
                        for y in range(tl_y, br_y + 1):
                            if x == tl_x or x == br_x or y == tl_y or y == br_y:
                                grid[x][y] = Wall(x, y)
                    
                    # Add a door connecting to the hallway
                    door_x = br_x + 1 if i == 0 else tl_x - 1
                    door_y = (tl_y + br_y) // 2
                    room.add_door((door_x, door_y))
                    
                    # Remove the wall at the door position
                    if 0 <= door_x < height and 0 <= door_y < width:
                        grid[door_x][door_y] = GridCell(door_x, door_y)
                    
                    room_id += 1
    
    return building


def generate_office_layout(width: int, height: int, grid: List[List[GridCell]], seed: int = 42) -> Building:
    """Generate an office layout."""
    # Office layout would be different than school, with cubicles, conference rooms, etc.
    # For now, we'll use a simplified approach similar to the school
    building = Building(width, height, "office")
    
    # Initialize random generator
    rng = np.random.default_rng(seed=seed)
    
    # Create a central hallway
    h_center = height // 2
    hallway_width = max(3, height // 10)
    
    # Horizontal hallway
    building.add_hallway(
        (h_center - hallway_width // 2, 0),
        (h_center + hallway_width // 2, width - 1)
    )
    
    # Add walls around the layout
    for i in range(height):
        for j in range(width):
            # Skip hallway area
            if h_center - hallway_width // 2 <= i <= h_center + hallway_width // 2:
                continue
                
            # Add wall if this is on the edge of the building
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                grid[i][j] = Wall(i, j)
    
    # Add exits
    building.add_exit((h_center, 0))  # Left exit
    building.add_exit((h_center, width - 1))  # Right exit
    
    # Place exits in the grid
    for exit_pos in building.exits:
        x, y = exit_pos
        grid[x][y] = Exit(x, y)
    
    # Add offices and conference rooms
    # (Simplified implementation)
    
    return building


def generate_mall_layout(width: int, height: int, grid: List[List[GridCell]], seed: int = 42) -> Building:
    """Generate a mall layout."""
    # Mall layout would have stores, food court, central walkways, etc.
    # For now, we'll use a simplified approach
    building = Building(width, height, "mall")
    
    # Initialize random generator
    rng = np.random.default_rng(seed=seed)
    
    # Create a cross-shaped walkway
    h_center = height // 2
    w_center = width // 2
    walkway_width = max(5, min(height, width) // 8)
    
    # Horizontal walkway
    building.add_hallway(
        (h_center - walkway_width // 2, 0),
        (h_center + walkway_width // 2, width - 1)
    )
    
    # Vertical walkway
    building.add_hallway(
        (0, w_center - walkway_width // 2),
        (height - 1, w_center + walkway_width // 2)
    )
    
    # Add walls around the layout
    for i in range(height):
        for j in range(width):
            # Skip walkway areas
            if ((h_center - walkway_width // 2 <= i <= h_center + walkway_width // 2) or
                (w_center - walkway_width // 2 <= j <= w_center + walkway_width // 2)):
                continue
                
            # Add wall if this is on the edge of the building
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                grid[i][j] = Wall(i, j)
    
    # Add exits at the ends of walkways
    building.add_exit((h_center, 0))  # Left exit
    building.add_exit((h_center, width - 1))  # Right exit
    building.add_exit((0, w_center))  # Top exit
    building.add_exit((height - 1, w_center))  # Bottom exit
    
    # Place exits in the grid
    for exit_pos in building.exits:
        x, y = exit_pos
        grid[x][y] = Exit(x, y)
    
    # Add store areas
    # (Simplified implementation)
    
    return building
