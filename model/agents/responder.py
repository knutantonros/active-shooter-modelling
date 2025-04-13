import numpy as np
from typing import Tuple, List, Dict, Any
from .agent import Agent

class Responder(Agent):
    """First responder agent in the active shooter simulation."""
    
    def __init__(
        self,
        id: int,
        pos: Tuple[int, int],
        width: int,
        height: int,
        responder_type: str = None,
        seed: int = 0,
        model: str = "gpt-3.5-turbo",
        api_key: str = None
    ):
        super().__init__(id, pos, width, height, seed, model, api_key)
        
        # Responder-specific attributes
        self.responder_type = responder_type or self.rng.choice([
            "police", "swat", "medic", "security"
        ])
        
        # Responder capabilities
        self.neutralize_accuracy = 0.7 if self.responder_type in ["police", "swat"] else 0.3
        self.can_treat_casualties = self.responder_type == "medic"
        self.coordination = self.rng.uniform(0.6, 1.0)  # ability to coordinate with other responders
        
        # For tracking decision-making
        self.known_civilian_positions = {}  # positions of civilians the responder has seen
        self.known_shooter_positions = {}  # positions of shooters the responder has seen
        self.known_casualty_positions = {}  # positions of casualties the responder has seen
        self.reported_info = set()  # information that has been reported to other responders
        self.heard_gunshots = False  # whether the responder has heard gunshots
    
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
        action, target = self.parse_llm_response(response, society)
        
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
        """Update the responder's knowledge about the environment."""
        # Check for visible agents
        visible_agents = self.get_visible_agents(society, visibility_range=7)  # Responders can see far
        
        # Update knowledge based on what's visible
        current_round = society.round
        
        for agent_info in visible_agents:
            if agent_info["type"] == "Civilian":
                self.known_civilian_positions[agent_info["id"]] = {
                    "pos": agent_info["pos"],
                    "status": agent_info["status"],
                    "last_seen_round": current_round
                }
                
                # Track casualties specifically
                if agent_info["status"] == "casualty":
                    self.known_casualty_positions[agent_info["id"]] = {
                        "pos": agent_info["pos"],
                        "last_seen_round": current_round
                    }
                    
            elif agent_info["type"] == "Shooter":
                self.known_shooter_positions[agent_info["id"]] = {
                    "pos": agent_info["pos"],
                    "status": agent_info["status"],
                    "last_seen_round": current_round
                }
        
        # Check for other responders' knowledge
        # (In a real implementation, this would involve sharing information between responders)
        # For simplicity, we'll skip this for now
    
    def _generate_prompt(self, society) -> str:
        """Generate a prompt for the LLM based on the responder's context."""
        # Basic information about the responder
        prompt = f"""
You are a first responder (ID: {self.id}, Type: {self.responder_type}) in an active shooter scenario.
The current round is {society.round}.
Your current position is {self.pos}.
Your current status is: {self.status}.

Environment information:
"""

        # Add information about visible agents
        visible_agents = self.get_visible_agents(society, visibility_range=7)
        if visible_agents:
            prompt += "\nVisible people:\n"
            for agent in visible_agents:
                prompt += f"- {agent['type']} at {agent['pos']}, status: {agent['status']}, distance: {agent['distance']:.1f}\n"
        else:
            prompt += "\nNo one is currently visible.\n"
        
        # Add information about known positions
        if self.known_shooter_positions:
            prompt += "\nShooters you've seen (might have moved):\n"
            for agent_id, info in list(self.known_shooter_positions.items())[-5:]:  # Show only the 5 most recent
                prompt += f"- Shooter {agent_id} at {info['pos']}, status: {info['status']}, last seen at round {info['last_seen_round']}\n"
        
        if self.known_casualty_positions:
            prompt += "\nCasualties you've seen (might have been moved):\n"
            for agent_id, info in list(self.known_casualty_positions.items())[-5:]:  # Show only the 5 most recent
                prompt += f"- Casualty {agent_id} at {info['pos']}, last seen at round {info['last_seen_round']}\n"
        
        if self.known_civilian_positions:
            prompt += "\nCivilians you've seen (might have moved):\n"
            for agent_id, info in list(self.known_civilian_positions.items())[-5:]:  # Show only the 5 most recent
                prompt += f"- Civilian {agent_id} at {info['pos']}, status: {info['status']}, last seen at round {info['last_seen_round']}\n"
        
        # Prompt for decision
        prompt += f"""
Based on the information above, what is your decision? Choose one:
1. Move toward a specific direction or location
2. Neutralize a shooter
3. Help a casualty (if you're a medic)
4. Search for the shooter
5. Evacuate civilians

Explain your reasoning briefly (1-2 sentences) and then clearly state your action.
Your response should follow this format:
Reasoning: [your reasoning]
Action: [move/neutralize/help/search/evacuate] [direction/target if applicable]

Your primary goal is to neutralize the shooter(s) and prevent further casualties. Secondary goals are to help casualties and evacuate civilians.
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
        
        # Process neutralize action (targeting a shooter)
        if "neutralize" in words:
            # Look for visible shooters
            visible_agents = self.get_visible_agents(society, visibility_range=7)
            visible_shooters = [a for a in visible_agents if a["type"] == "Shooter" and a["status"] == "active"]
            
            if visible_shooters:
                # Target the closest shooter
                target = min(visible_shooters, key=lambda a: a["distance"])
                return "neutralize", target["pos"]
            elif self.known_shooter_positions:
                # Move toward the last known shooter position
                recent_shooter = max(self.known_shooter_positions.values(), 
                                   key=lambda info: info["last_seen_round"])
                return "move", self._move_toward(recent_shooter["pos"])
            else:
                # Search pattern if no shooter information
                return "move", self._get_random_adjacent_position()
        
        # Process help action (for medics treating casualties)
        if "help" in words and self.can_treat_casualties:
            # Look for visible casualties
            visible_agents = self.get_visible_agents(society, visibility_range=7)
            visible_casualties = [a for a in visible_agents if a["status"] == "casualty"]
            
            if visible_casualties:
                # Go to the closest casualty
                target = min(visible_casualties, key=lambda a: a["distance"])
                return "move", target["pos"]  # Move to casualty position
            elif self.known_casualty_positions:
                # Move toward the last known casualty position
                recent_casualty = max(self.known_casualty_positions.values(), 
                                    key=lambda info: info["last_seen_round"])
                return "move", self._move_toward(recent_casualty["pos"])
            else:
                # Search pattern if no casualty information
                return "move", self._get_random_adjacent_position()
        
        # Process search action (looking for the shooter)
        if "search" in words:
            # Implement search pattern
            # For now, just move randomly but could be more sophisticated
            return "move", self._get_random_adjacent_position()
        
        # Process evacuate action (helping civilians)
        if "evacuate" in words:
            # Look for visible civilians
            visible_agents = self.get_visible_agents(society, visibility_range=7)
            visible_civilians = [a for a in visible_agents if a["type"] == "Civilian" and a["status"] == "active"]
            
            if visible_civilians:
                # Go to the closest civilian
                target = min(visible_civilians, key=lambda a: a["distance"])
                return "move", target["pos"]  # Move to civilian position
            else:
                # Search pattern if no civilian information
                return "move", self._get_random_adjacent_position()
        
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
            
            # If no specific direction, but moving toward shooter or casualty was mentioned
            if "shooter" in words and self.known_shooter_positions:
                recent_shooter = max(self.known_shooter_positions.values(), 
                                   key=lambda info: info["last_seen_round"])
                return "move", self._move_toward(recent_shooter["pos"])
            
            if "casualty" in words and self.known_casualty_positions:
                recent_casualty = max(self.known_casualty_positions.values(), 
                                    key=lambda info: info["last_seen_round"])
                return "move", self._move_toward(recent_casualty["pos"])
            
            if "civilian" in words and self.known_civilian_positions:
                active_civilians = {id: info for id, info in self.known_civilian_positions.items() 
                                  if info["status"] == "active"}
                if active_civilians:
                    recent_civilian = max(active_civilians.values(), 
                                        key=lambda info: info["last_seen_round"])
                    return "move", self._move_toward(recent_civilian["pos"])
        
        # Default to searching pattern if no recognizable action
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