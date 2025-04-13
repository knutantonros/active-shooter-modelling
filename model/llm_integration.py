import openai
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

class LLMIntegration:
    """Class to handle LLM API integration and prompt management."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 150,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Initialize logging
        logging.getLogger(__name__).setLevel(logging.INFO)
        
        # Load prompt templates
        self._load_prompt_templates()
    
    def _load_prompt_templates(self):
        """Load prompt templates for different agent types and scenarios."""
        self.prompts = {
            "civilian": {
                "decide": """
You are a civilian (ID: {agent_id}) in an active shooter scenario. Your personality type is {personality_type}.
The current round is {round}.
Your current position is {position}.
Your current status is: {status}.

Your fear level is {fear_level}.

Environment information:
{environment_info}

Based on the information above, what is your decision? Choose one:
1. Move toward a specific direction or location (e.g., move north, move to exit)
2. Hide in your current location
3. Follow another civilian

Explain your reasoning briefly (1-2 sentences) and then clearly state your action.
Your response should follow this format:
Reasoning: [your reasoning]
Action: [move/hide/follow] [direction/location if applicable]
"""
            },
            "shooter": {
                "decide": """
You are controlling a shooter (ID: {agent_id}) in a simulation. Your shooter type is {shooter_type}.
The current round is {round}.
Your current position is {position}.
Your current status is: {status}.
You have {ammo} ammunition remaining.
You have fired {shots_fired} shots and hit {targets_hit} targets.

Environment information:
{environment_info}

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
            },
            "responder": {
                "decide": """
You are a first responder (ID: {agent_id}, Type: {responder_type}) in an active shooter scenario.
The current round is {round}.
Your current position is {position}.
Your current status is: {status}.

Environment information:
{environment_info}

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
            }
        }
    
    def get_prompt(self, agent_type: str, prompt_type: str, **kwargs) -> str:
        """Get a formatted prompt for a specific agent type and scenario.
        
        Args:
            agent_type: Type of agent (civilian, shooter, responder)
            prompt_type: Type of prompt (decide, etc.)
            **kwargs: Format variables to insert into the prompt
            
        Returns:
            str: Formatted prompt
        """
        if agent_type not in self.prompts or prompt_type not in self.prompts[agent_type]:
            raise ValueError(f"No prompt template found for {agent_type}/{prompt_type}")
        
        # Get the template
        template = self.prompts[agent_type][prompt_type]
        
        # Format with provided kwargs
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing format variable in prompt: {e}")
    
    def query_llm(self, prompt: str) -> str:
        """Send a prompt to the LLM and get a response.
        
        Args:
            prompt: The prompt text
            
        Returns:
            str: The LLM's response
        """
        if not self.api_key:
            # Mock response for testing without API key
            logging.warning("No API key provided. Using mock response.")
            return f"move north"
        
        openai.api_key = self.api_key
        
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    sleep_time = self.backoff_factor ** attempt
                    logging.warning(f"API error: {e}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logging.error(f"Failed to get LLM response after {self.max_retries} attempts: {e}")
                    return "move random"  # Fallback behavior
    
    def batch_query(self, prompts: List[str]) -> List[str]:
        """Send multiple prompts to the LLM and get responses.
        
        Args:
            prompts: List of prompt texts
            
        Returns:
            List[str]: List of LLM responses
        """
        # For now, just query them sequentially
        # In a production system, this could be optimized with batching
        return [self.query_llm(prompt) for prompt in prompts]
    
    def parse_decision(self, response: str, agent_type: str) -> Tuple[str, Any]:
        """Parse a decision response from the LLM.
        
        Args:
            response: The LLM's response text
            agent_type: Type of agent (civilian, shooter, responder)
            
        Returns:
            Tuple[str, Any]: (action, target)
        """
        # Extract the action line from the response
        lines = response.strip().split('\n')
        action_line = ""
        for line in lines:
            if line.lower().startswith("action:"):
                action_line = line.lower().replace("action:", "").strip()
                break
        
        if not action_line:
            # Fallback if no action line is found
            return "move", "random"
        
        # Split into words for parsing
        words = action_line.split()
        
        # Basic action parsing
        if "move" in words:
            # Extract direction or target
            for direction in ["north", "south", "east", "west", "up", "down", "left", "right"]:
                if direction in words:
                    return "move", direction
            
            # If no direction specified
            return "move", "random"
        
        elif "hide" in words:
            return "hide", None
        
        elif "shoot" in words or "neutralize" in words:
            # Extract target if specified (coordinate or description)
            # This is a simplified implementation
            return "shoot" if "shoot" in words else "neutralize", "nearest"
        
        elif "help" in words and agent_type == "responder":
            return "help", "nearest_casualty"
        
        elif "search" in words and agent_type == "responder":
            return "search", None
        
        elif "evacuate" in words and agent_type == "responder":
            return "evacuate", "nearest_civilian"
        
        elif "follow" in words and agent_type == "civilian":
            return "follow", "nearest_civilian"
        
        # Default fallback
        return "move", "random"