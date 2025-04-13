import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import openai
import time
import logging
from abc import ABC, abstractmethod

class Agent(ABC):
    """Base class for all agents in the simulation."""
    
    def __init__(
        self, 
        id: int,
        pos: Tuple[int, int],
        width: int,
        height: int,
        seed: int = 0,
        model: str = "gpt-3.5-turbo",
        api_key: str = None
    ):
        self.id = id
        self.pos = pos
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        
        # Agent state
        self.status = "active"  # active, hiding, casualty, escaped, neutralized
        self.is_hidden = False
        self.pos_history = [pos]
        self.last_action = None
        self.target = None
        
        # LLM settings
        self.model = model
        self.api_key = api_key
        self.temperature = 0.2
        self.max_tokens = 150
        
        # Response history for consistency
        self.decision_history = []
    
    @abstractmethod
    def decide(self, society) -> Tuple[str, Any]:
        """Decide what action to take based on the current state.
        
        Args:
            society: The simulation environment
            
        Returns:
            Tuple[str, Any]: (action, target)
                action: The type of action to take (move, hide, etc.)
                target: Target information (e.g., position to move to)
        """
        pass
    
    def get_prompt(self, society, prompt_template: str) -> str:
        """Generate a prompt for the LLM based on the agent's context.
        
        Args:
            society: The simulation environment
            prompt_template: Template for the specific agent type
            
        Returns:
            str: The formatted prompt text
        """
        # This is a basic implementation that would be extended in subclasses
        return prompt_template.format(
            agent_id=self.id,
            position=self.pos,
            status=self.status,
            round=society.round
        )
    
    def get_visible_agents(self, society, visibility_range: int = 5) -> List[Dict]:
        """Get information about agents visible to this agent.
        
        Args:
            society: The simulation environment
            visibility_range: How far the agent can see
            
        Returns:
            List[Dict]: Information about visible agents
        """
        visible_agents = []
        my_x, my_y = self.pos
        
        for agent in society.all_agents:
            if agent.id == self.id:
                continue  # Skip self
                
            agent_x, agent_y = agent.pos
            distance = np.sqrt((my_x - agent_x)**2 + (my_y - agent_y)**2)
            
            if distance <= visibility_range:
                # Check if there's a clear line of sight
                if self._has_line_of_sight(society, agent.pos):
                    visible_agents.append({
                        "id": agent.id,
                        "type": agent.__class__.__name__,
                        "pos": agent.pos,
                        "status": agent.status,
                        "distance": distance
                    })
        
        return visible_agents
    
    def _has_line_of_sight(self, society, target_pos: Tuple[int, int]) -> bool:
        """Check if there's a clear line of sight to the target position.
        
        Args:
            society: The simulation environment
            target_pos: Position to check visibility to
            
        Returns:
            bool: True if there's a clear line of sight
        """
        # Bresenham's line algorithm to check for obstacles
        x0, y0 = self.pos
        x1, y1 = target_pos
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while x0 != x1 or y0 != y1:
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
                
            # Skip the starting and ending positions
            if (x0, y0) != self.pos and (x0, y0) != target_pos:
                # Check if there's an obstacle or wall blocking the view
                cell = society.grid[x0][y0]
                if hasattr(cell, 'name'):
                    if cell.name in ["wall", "obstacle"]:
                        return False
        
        return True
    
    def communicate(self, prompt: str) -> str:
        """Send a prompt to the LLM and get a response.
        
        Args:
            prompt: The prompt text
            
        Returns:
            str: The LLM's response
        """
        if not self.api_key:
            # Mock response for testing without API key
            return "move north"
        
        retries = 3
        backoff_factor = 2
        
        for attempt in range(retries):
            try:
                openai.api_key = self.api_key
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt < retries - 1:
                    sleep_time = backoff_factor ** attempt
                    logging.warning(f"API error: {e}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Failed to get LLM response after {retries} attempts: {e}")
                    return "move random"  # Fallback behavior
    
    def parse_llm_response(self, response: str) -> Tuple[str, Any]:
        """Parse the LLM's response into an action and target.
        
        Args:
            response: The LLM's response text
            
        Returns:
            Tuple[str, Any]: (action, target)
        """
        # This is a basic implementation - would be customized in subclasses
        # Extract action type from response (move, hide, etc.)
        words = response.lower().split()
        
        if "hide" in words:
            return "hide", None
            
        if "move" in words:
            # Try to extract direction
            directions = {
                "north": (-1, 0),
                "south": (1, 0),
                "east": (0, 1),
                "west": (0, -1),
                "up": (-1, 0),
                "down": (1, 0),
                "right": (0, 1),
                "left": (0, -1),
                "northeast": (-1, 1),
                "northwest": (-1, -1),
                "southeast": (1, 1),
                "southwest": (1, -1)
            }
            
            for direction, offset in directions.items():
                if direction in words:
                    x, y = self.pos
                    return "move", (x + offset[0], y + offset[1])
            
            # If no direction specified, move randomly
            return "move", self._get_random_adjacent_position()
        
        # Default to random movement if no recognizable action
        return "move", self._get_random_adjacent_position()
    
    def _get_random_adjacent_position(self) -> Tuple[int, int]:
        """Get a random adjacent position.
        
        Returns:
            Tuple[int, int]: New position coordinates
        """
        x, y = self.pos
        moves = [
            (x-1, y),    # North
            (x+1, y),    # South
            (x, y+1),    # East
            (x, y-1),    # West
            (x-1, y+1),  # Northeast
            (x-1, y-1),  # Northwest
            (x+1, y+1),  # Southeast
            (x+1, y-1),  # Southwest
        ]
        
        # Filter out invalid moves (outside grid)
        valid_moves = [(nx, ny) for nx, ny in moves if 0 <= nx < self.height and 0 <= ny < self.width]
        
        if valid_moves:
            return self.rng.choice(valid_moves)
        else:
            return self.pos  # Stay in place if no valid moves
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, pos={self.pos}, status={self.status})"
