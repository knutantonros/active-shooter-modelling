import numpy as np
from typing import Tuple, List, Dict, Any
from .agent import Agent

class Civilian(Agent):
    """Civilian agent in the active shooter simulation."""
    
    def __init__(
        self,
        id: int,
        pos: Tuple[int, int],
        width: int,
        height: int,
        personality_type: str = None,
        seed: int = 0,
        model: str = "gpt-3.5-turbo",
        api_key: str = None
    ):
        super().__init__(id, pos, width, height, seed, model, api_key)
        
        # Civilian-specific attributes
        self.personality_type = personality_type or self.rng.choice([
            "cautious", "brave", "panicked", "leader", "follower"
        ])
        self.awareness = self.rng.uniform(0.3, 1.0)  # how aware they are of the situation
        self.fear_level = 0.0  # increases when they see the shooter or hear gunshots
        self.knows_shooter_location = False
        self.has_seen_casualties = False
        
        # For tracking decision-making
        self.exit_knowledge = {}  # information about exits they've seen
        self.chosen_strategy = None  # run, hide, fight
    
    def decide(self, society) -> Tuple[str, Any]:
        """Decide what action to take based on current state.
        
        Args:
            society: The simulation environment
            
        Returns:
            Tuple[str, Any]: (action, target)
        """
        # Update knowledge about the environment
        self._update_environment_knowledge(society)
        
        # Generate LLM prompt
        prompt = self._generate_prompt(society)
        
        # Get response from LLM
        response = self.communicate(prompt)
        
        # Parse the response
        action, target = self.parse_llm_response(response)
        
        # Save for history
        self.last_action = action
        self.target = target
        self.decision_history.append({
            "round": society.round,
            "prompt": prompt,
            "response": response,
            "action": action,
            "target": target
        })
        
        return action, target
    
    def _update_environment_knowledge(self, society):
        """Update the agent's knowledge about the environment."""
        # Look for exits
        for exit in society.exits:
            if self._has_line_of_sight(society, exit.pos):
                distance = np.sqrt((self.pos[0] - exit.pos[0])**2 + (self.pos[1] - exit.pos[1])**2)
                self.exit_knowledge[exit.pos] = {
                    "distance": distance,
                    "last_seen_round": society.round
                }
        
        # Check for visible agents
        visible_agents = self.get_visible_agents(society)
        
        # Update knowledge based on what's visible
        for agent_info in visible_agents:
            # If they see the shooter
            if agent_info["type"] == "Shooter":
                self.knows_shooter_location = True
                self.fear_level = min(1.0, self.fear_level + 0.3)  # Increase fear
            
            # If they see casualties
            if agent_info["type"] == "Civilian" and agent_info["status"] == "casualty":
                self.has_seen_casualties = True
                self.fear_level = min(1.0, self.fear_level + 0.2)  # Increase fear
    
    def _generate_prompt(self, society) -> str:
        """Generate a prompt for the LLM based on the agent's context."""
        # Basic information about the agent
        prompt = f"""
You are a civilian (ID: {self.id}) in an active shooter scenario. Your personality type is {self.personality_type}.
The current round is {society.round}.
Your current position is {self.pos}.
Your current status is: {self.status}.

Your fear level is {"low" if self.fear_level < 0.3 else "medium" if self.fear_level < 0.7 else "high"}.

Environment information:
"""

        # Add information about exits
        if self.exit_knowledge:
            prompt += "Known exits:\n"
            for pos, info in self.exit_knowledge.items():
                prompt += f"- Exit at {pos}, distance: {info['distance']:.1f}, last seen at round {info['last_seen_round']}\n"
        else:
            prompt += "You don't know the location of any exits yet.\n"

        # Add information about visible agents
        visible_agents = self.get_visible_agents(society)
        if visible_agents:
            prompt += "\nVisible people:\n"
            for agent in visible_agents:
                prompt += f"- {agent['type']} at {agent['pos']}, status: {agent['status']}, distance: {agent['distance']:.1f}\n"
        
        # Add critical information
        if self.knows_shooter_location:
            prompt += "\nYou have seen the shooter!\n"
        
        if self.has_seen_casualties:
            prompt += "\nYou have seen casualties!\n"
        
        # Prompt for decision
        prompt += f"""
Based on the information above, what is your decision? Choose one:
1. Move toward a specific direction or location (e.g., move north, move to exit)
2. Hide in your current location
3. Follow another civilian

Explain your reasoning briefly (1-2 sentences) and then clearly state your action.
Your response should follow this format:
Reasoning: [your reasoning]
Action: [move/hide/follow] [direction/location if applicable]
"""
        
        return prompt
    
    def parse_llm_response(self, response: str) -> Tuple[str, Any]:
        """Parse the LLM's response into an action and target."""
        # Extract the action line from the response
        lines = response.strip().split('\n')
        action_line = ""
        for line in lines:
            if line.lower().startswith("action:"):
                action_line = line.lower().replace("action:", "").strip()
                break
        
        if not action_line:
            # Fallback if no action line is found
            return "move", self._get_random_adjacent_position()
        
        words = action_line.split()
        
        # Process hide action
        if "hide" in words:
            return "hide", None
        
        # Process follow action (treated as move for now)
        if "follow" in words:
            # Would need to implement following logic
            return "move", self._get_random_adjacent_position()
        
        # Process move action
        if "move" in words:
            # Check for exits
            if "exit" in words and self.exit_knowledge:
                # Find the closest exit
                closest_exit = min(self.exit_knowledge.keys(), 
                                  key=lambda pos: self.exit_knowledge[pos]["distance"])
                
                # Move toward the exit
                return "move", self._move_toward(closest_exit)
            
            # Check for directions
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
        
        # Default to random movement if no recognizable action
        return "move", self._get_random_adjacent_position()
    
    def _move_toward(self, target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate a new position that moves toward the target.
        
        Args:
            target_pos: Position to move toward
            
        Returns:
            Tuple[int, int]: New position coordinates
        """
        x, y = self.pos
        tx, ty = target_pos
        
        # Calculate direction
        dx = 0 if tx == x else 1 if tx > x else -1
        dy = 0 if ty == y else 1 if ty > y else -1
        
        # Create new position
        new_pos = (x + dx, y + dy)
        
        # Ensure it's within bounds
        nx, ny = new_pos
        if 0 <= nx < self.height and 0 <= ny < self.width:
            return new_pos
        else:
            return self.pos  # Stay in place if the move would be out of bounds