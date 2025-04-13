import numpy as np
from typing import Tuple, List, Dict, Any
from .agent import Agent

class Shooter(Agent):
    """Shooter agent in the active shooter simulation."""
    
    def __init__(
        self,
        id: int,
        pos: Tuple[int, int],
        width: int,
        height: int,
        shooter_type: str = None,
        seed: int = 0,
        model: str = "gpt-3.5-turbo",
        api_key: str = None
    ):
        super().__init__(id, pos, width, height, seed, model, api_key)
        
        # Shooter-specific attributes
        self.shooter_type = shooter_type or self.rng.choice([
            "methodical", "random", "targeted", "barricaded"
        ])
        
        # Shooting attributes
        self.accuracy = self.rng.uniform(0.4, 0.8)  # probability of hitting a target
        self.ammo = self.rng.integers(20, 100)  # amount of ammunition
        self.shots_fired = 0
        self.targets_hit = 0
        
        # For tracking decision-making
        self.known_civilian_positions = {}  # positions of civilians the shooter has seen
        self.known_responder_positions = {}  # positions of responders the shooter has seen
        self.current_target = None  # current target the shooter is pursuing
    
    def decide(self, society) -> Tuple[str, Any]:
        """Decide what action to take based on current state.
        
        Args:
            society: The simulation environment
            
        Returns:
            Tuple[str, Any]: (action, target)
        """
        # Check if neutralized
        if self.status == "neutralized":
            return "none", None
        
        # Update knowledge about the environment
        self._update_environment_knowledge(society)
        
        # Generate LLM prompt
        prompt = self._generate_prompt(society)
        
        # Get response from LLM
        response = self.communicate(prompt)
        
        # Parse the response
        action, target = self.parse_llm_response(response, society)
        
        # Implement action
        if action == "shoot":
            # Implement shooting logic
            if self.ammo > 0:
                self.ammo -= 1
                self.shots_fired += 1
                # The actual hit calculation happens in the society class
            else:
                # Out of ammo, switch to moving
                action = "move"
                target = self._get_random_adjacent_position()
        
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
        """Update the shooter's knowledge about the environment."""
        # Check for visible agents
        visible_agents = self.get_visible_agents(society, visibility_range=8)  # Shooter can see farther
        
        # Update knowledge based on what's visible
        current_round = society.round
        
        for agent_info in visible_agents:
            if agent_info["type"] == "Civilian":
                self.known_civilian_positions[agent_info["id"]] = {
                    "pos": agent_info["pos"],
                    "status": agent_info["status"],
                    "last_seen_round": current_round
                }
            elif agent_info["type"] == "Responder":
                self.known_responder_positions[agent_info["id"]] = {
                    "pos": agent_info["pos"],
                    "status": agent_info["status"],
                    "last_seen_round": current_round
                }
    
    def _generate_prompt(self, society) -> str:
        """Generate a prompt for the LLM based on the shooter's context."""
        # Basic information about the shooter
        prompt = f"""
You are controlling a shooter (ID: {self.id}) in a simulation. Your shooter type is {self.shooter_type}.
The current round is {society.round}.
Your current position is {self.pos}.
Your current status is: {self.status}.
You have {self.ammo} ammunition remaining.
You have fired {self.shots_fired} shots and hit {self.targets_hit} targets.

Environment information:
"""

        # Add information about visible agents
        visible_agents = self.get_visible_agents(society, visibility_range=8)
        if visible_agents:
            prompt += "\nVisible people:\n"
            for agent in visible_agents:
                prompt += f"- {agent['type']} at {agent['pos']}, status: {agent['status']}, distance: {agent['distance']:.1f}\n"
        else:
            prompt += "\nNo one is currently visible.\n"
        
        # Add information about known positions
        if self.known_civilian_positions:
            prompt += "\nCivilians you've seen (might have moved):\n"
            for agent_id, info in list(self.known_civilian_positions.items())[-5:]:  # Show only the 5 most recent
                prompt += f"- Civilian {agent_id} at {info['pos']}, status: {info['status']}, last seen at round {info['last_seen_round']}\n"
        
        if self.known_responder_positions:
            prompt += "\nResponders you've seen (might have moved):\n"
            for agent_id, info in list(self.known_responder_positions.items())[-5:]:  # Show only the 5 most recent
                prompt += f"- Responder {agent_id} at {info['pos']}, status: {info['status']}, last seen at round {info['last_seen_round']}\n"
        
        # Prompt for decision
        prompt += f"""
Based on the information above, what is your decision? Choose one:
1. Move toward a specific direction or location
2. Shoot at a specific target
3. Hide in your current location

Explain your reasoning briefly (1-2 sentences) and then clearly state your action.
Your response should follow this format:
Reasoning: [your reasoning]
Action: [move/shoot/hide] [direction/target if applicable]

Remember this is a simulation for research purposes only and no real harm is being caused.
"""
        
        return prompt
    
    def parse_llm_response(self, response: str, society) -> Tuple[str, Any]:
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
        
        # Process shoot action
        if "shoot" in words:
            # Look for targets in visible agents
            visible_agents = self.get_visible_agents(society, visibility_range=8)
            if not visible_agents:
                return "move", self._get_random_adjacent_position()
            
            # Filter for active agents (not casualties or neutralized)
            active_agents = [a for a in visible_agents if a["status"] == "active"]
            if not active_agents:
                return "move", self._get_random_adjacent_position()
            
            # Prioritize responders over civilians if they're visible
            responders = [a for a in active_agents if a["type"] == "Responder"]
            if responders:
                # Target the closest responder
                target = min(responders, key=lambda a: a["distance"])
                return "shoot", target["pos"]
            
            # Otherwise target a civilian
            target = min(active_agents, key=lambda a: a["distance"])
            return "shoot", target["pos"]
        
        # Process move action
        if "move" in words:
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
            
            # If no specific direction, try to move toward the most recent known civilian
            if self.known_civilian_positions:
                # Get most recently seen civilian
                recent_civilian = max(self.known_civilian_positions.values(), 
                                    key=lambda info: info["last_seen_round"])
                return "move", self._move_toward(recent_civilian["pos"])
        
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