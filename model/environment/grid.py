from typing import Tuple, List

class GridCell:
    """Base class for all cells in the grid."""
    
    def __init__(self, x: int, y: int):
        self.pos = (x, y)
        self.name = "empty"
        self.can_pass = True  # Whether agents can pass through this cell
    
    def __repr__(self) -> str:
        return f"{self.name}({self.pos[0]}, {self.pos[1]})"


class Wall(GridCell):
    """Wall cell that agents cannot pass through."""
    
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.name = "wall"
        self.can_pass = False


class Obstacle(GridCell):
    """Obstacle cell that agents cannot pass through."""
    
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.name = "obstacle"
        self.can_pass = False


class Exit(GridCell):
    """Exit cell that agents can pass through to escape."""
    
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
        self.name = "exit"
        self.can_pass = True
        self.escaped_agent_ids = []  # IDs of agents that have escaped through this exit
    
    def add_escaped_agent(self, agent_id: int):
        """Record an agent escaping through this exit."""
        self.escaped_agent_ids.append(agent_id)
    
    def get_escaped_count(self) -> int:
        """Get the number of agents that have escaped through this exit."""
        return len(self.escaped_agent_ids)