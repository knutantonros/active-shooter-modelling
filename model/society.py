import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
import time
import logging
from typing import List, Tuple, Dict, Any
from copy import deepcopy

# Import our agent classes
from .agents.agent import Agent
from .agents.civilian import Civilian
from .agents.shooter import Shooter
from .agents.responder import Responder

# Import environment classes
from .environment.grid import GridCell, Wall, Exit, Obstacle

class ActiveShooterSociety:
    """Main simulation environment for the active shooter scenario."""
    
    def __init__(
        self, 
        name: str,
        width: int = 50,
        height: int = 50,
        num_civilians: int = 100,
        num_shooters: int = 1,
        num_responders: int = 5,
        responder_arrival_time: int = 10,  # Time steps before responders arrive
        seed: int = 0,
        llm_model: str = "gpt-3.5-turbo",
        api_key: str = None,
        output_path: str = "output",
    ):
        # Basic simulation parameters
        self.name = name
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        
        # Agent parameters
        self.num_civilians = num_civilians
        self.num_shooters = num_shooters
        self.num_responders = num_responders
        self.responder_arrival_time = responder_arrival_time
        
        # LLM parameters
        self.llm_model = llm_model
        self.api_key = api_key
        
        # Environment setup
        self.grid = [[GridCell(i, j) for j in range(width)] for i in range(height)]
        self.exits = []
        self.obstacles = []
        self.walls = []
        
        # Agent lists
        self.civilians = []
        self.shooters = []
        self.responders = []
        self.all_agents = []  # Combined list for convenience
        
        # Simulation state
        self.round = 0
        self.is_active = True  # Simulation continues while True
        
        # Statistics
        self.num_casualties = 0
        self.num_escaped = 0
        self.shooter_neutralized = False
        
        # Logging and output
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        logging.basicConfig(
            filename=f"{output_path}/simulation.log",
            level=logging.INFO,
            filemode='w'
        )
        self.logging = logging
        
        # Initialize statistics log
        self._init_statistics_log()
        
    def _init_statistics_log(self):
        """Initialize the statistics log files."""
        # Agent log with positions, status, etc.
        df = pd.DataFrame(columns=[
            'round', 'id', 'type', 'pos', 'status', 'target', 'action'
        ])
        df.to_csv(f"{self.output_path}/agent_log.csv", index=False)
        
        # Overall statistics log
        df = pd.DataFrame(columns=[
            'round', 'num_casualties', 'num_escaped', 'shooter_neutralized'
        ])
        df.to_csv(f"{self.output_path}/stats_log.csv", index=False)
    
    def generate_building_layout(self, layout_type="school"):
        """Generate a building layout based on the specified type.
        
        Args:
            layout_type: Type of building layout to generate.
                Options: "school", "office", "mall", "custom"
        """
        if layout_type == "school":
            self._generate_school_layout()
        elif layout_type == "office":
            self._generate_office_layout()
        elif layout_type == "mall":
            self._generate_mall_layout()
        else:
            # Default simple layout
            self._generate_simple_layout()
            
        self.logging.info(f"Generated {layout_type} layout")
    
    def _generate_simple_layout(self):
        """Generate a simple layout with outer walls and a few exits."""
        # Add outer walls
        for i in range(self.height):
            for j in range(self.width):
                # Add walls on the perimeter
                if (i == 0 or i == self.height - 1 or j == 0 or j == self.width - 1):
                    self.grid[i][j] = Wall(i, j)
                    self.walls.append(self.grid[i][j])
        
        # Add exits (e.g., at the center of each wall)
        exit_positions = [
            (0, self.width // 2),                     # Top wall
            (self.height - 1, self.width // 2),       # Bottom wall
            (self.height // 2, 0),                    # Left wall
            (self.height // 2, self.width - 1)        # Right wall
        ]
        
        for pos in exit_positions:
            x, y = pos
            self.grid[x][y] = Exit(x, y)
            self.exits.append(self.grid[x][y])
            
        # Add some random obstacles inside
        num_obstacles = int(0.05 * self.width * self.height)  # 5% of cells
        for _ in range(num_obstacles):
            while True:
                x = self.rng.integers(1, self.height - 1)
                y = self.rng.integers(1, self.width - 1)
                if isinstance(self.grid[x][y], GridCell):
                    self.grid[x][y] = Obstacle(x, y)
                    self.obstacles.append(self.grid[x][y])
                    break
    
    def _generate_school_layout(self):
        """Generate a school-like layout with classrooms, hallways, etc."""
        # Implementation for school layout
        # This would create a more complex layout with classrooms, hallways, etc.
        # For simplicity, I'm leaving this as a placeholder
        self._generate_simple_layout()  # Fallback
    
    def populate_agents(self):
        """Create and place agents in the environment."""
        self._populate_civilians()
        self._populate_shooters()
        # Responders will be added after a delay
        
        self.all_agents = self.civilians + self.shooters
    
    def _populate_civilians(self):
        """Create and place civilian agents."""
        for i in range(self.num_civilians):
            # Find an empty spot
            while True:
                x = self.rng.integers(1, self.height - 1)
                y = self.rng.integers(1, self.width - 1)
                if isinstance(self.grid[x][y], GridCell):
                    civilian = Civilian(
                        id=i,
                        pos=(x, y),
                        width=self.width,
                        height=self.height,
                        seed=self.seed + i,
                        model=self.llm_model,
                        api_key=self.api_key
                    )
                    self.civilians.append(civilian)
                    self.grid[x][y] = civilian
                    break
        
        self.logging.info(f"Added {len(self.civilians)} civilians")
    
    def _populate_shooters(self):
        """Create and place shooter agents."""
        for i in range(self.num_shooters):
            # Find an empty spot
            while True:
                x = self.rng.integers(1, self.height - 1)
                y = self.rng.integers(1, self.width - 1)
                if isinstance(self.grid[x][y], GridCell):
                    shooter = Shooter(
                        id=i,
                        pos=(x, y),
                        width=self.width,
                        height=self.height,
                        seed=self.seed + self.num_civilians + i,
                        model=self.llm_model,
                        api_key=self.api_key
                    )
                    self.shooters.append(shooter)
                    self.grid[x][y] = shooter
                    break
        
        self.logging.info(f"Added {len(self.shooters)} shooters")
    
    def _add_responders(self):
        """Add first responder agents after the specified delay."""
        for i in range(self.num_responders):
            # Responders start at exits
            if self.exits:
                # Pick a random exit
                exit_cell = self.rng.choice(self.exits)
                x, y = exit_cell.pos
                
                # Find a nearby empty cell
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.height and 0 <= ny < self.width and 
                            isinstance(self.grid[nx][ny], GridCell)):
                            responder = Responder(
                                id=i,
                                pos=(nx, ny),
                                width=self.width,
                                height=self.height,
                                seed=self.seed + self.num_civilians + self.num_shooters + i,
                                model=self.llm_model,
                                api_key=self.api_key
                            )
                            self.responders.append(responder)
                            self.grid[nx][ny] = responder
                            self.all_agents.append(responder)
                            break
                    else:
                        continue
                    break
        
        self.logging.info(f"Added {len(self.responders)} responders")
    
    def step(self):
        """Execute one time step of the simulation."""
        # Check if responders should arrive
        if self.round == self.responder_arrival_time:
            self._add_responders()
        
        # Store the current grid state to avoid conflicts
        next_grid = deepcopy(self.grid)
        
        # Shuffle agent order for fairness
        self.rng.shuffle(self.all_agents)
        
        # Process agent actions
        for agent in self.all_agents:
            if agent.status != "active":
                continue
                
            # Get agent's decision using LLM
            action, target = agent.decide(self)
            
            # Process the action
            if action == "move":
                self._process_move(agent, target, next_grid)
            elif action == "shoot":
                self._process_shoot(agent, target)
            elif action == "hide":
                # Agent stays in place but hides
                agent.is_hidden = True
            elif action == "neutralize":
                self._process_neutralize(agent, target)
        
        # Update the grid with the new state
        self.grid = next_grid
        
        # Update simulation status
        self._update_simulation_status()
        
        # Log statistics
        self._log_statistics()
        
        # Increment round counter
        self.round += 1
        
        return self.is_active
    
    def _process_move(self, agent, target_pos, next_grid):
        """Process a move action for an agent."""
        x, y = target_pos
        
        # Check if the move is valid
        if 0 <= x < self.height and 0 <= y < self.width:
            # Check if target cell is free
            if isinstance(next_grid[x][y], GridCell):
                # Update agent position
                old_x, old_y = agent.pos
                next_grid[old_x][old_y] = GridCell(old_x, old_y)
                next_grid[x][y] = agent
                agent.pos = (x, y)
                
                # Check if agent reached an exit
                if isinstance(self.grid[x][y], Exit):
                    if isinstance(agent, Civilian):
                        agent.status = "escaped"
                        self.num_escaped += 1
    
    def _process_shoot(self, agent, target_pos):
        """Process a shoot action."""
        x, y = target_pos
        
        # Check if target is valid
        if 0 <= x < self.height and 0 <= y < self.width:
            target_cell = self.grid[x][y]
            
            # Check if target is an agent
            if isinstance(target_cell, Agent):
                # Determine hit success (simplified)
                hit_success = self.rng.random() < agent.accuracy
                
                if hit_success:
                    if isinstance(target_cell, Civilian):
                        target_cell.status = "casualty"
                        self.num_casualties += 1
                    elif isinstance(target_cell, Shooter):
                        target_cell.status = "neutralized"
                        self.shooter_neutralized = True
                    elif isinstance(target_cell, Responder):
                        target_cell.status = "casualty"
    
    def _process_neutralize(self, agent, target_pos):
        """Process a neutralize action (for responders)."""
        x, y = target_pos
        
        # Check if target is valid
        if 0 <= x < self.height and 0 <= y < self.width:
            target_cell = self.grid[x][y]
            
            # Check if target is a shooter
            if isinstance(target_cell, Shooter):
                # Determine neutralization success
                success = self.rng.random() < agent.neutralize_accuracy
                
                if success:
                    target_cell.status = "neutralized"
                    self.shooter_neutralized = True
    
    def _update_simulation_status(self):
        """Update the overall simulation status."""
        # Check if simulation should end
        if self.shooter_neutralized:
            # If all shooters are neutralized
            self.logging.info(f"Round {self.round}: All shooters neutralized")
        
        # If all civilians are either escaped or casualties
        active_civilians = sum(1 for c in self.civilians if c.status == "active")
        if active_civilians == 0:
            self.logging.info(f"Round {self.round}: All civilians have escaped or are casualties")
            self.is_active = False
        
        # If maximum round count reached (to prevent infinite loops)
        if self.round >= 100:  # Arbitrary limit
            self.logging.info(f"Round {self.round}: Maximum round count reached")
            self.is_active = False
    
    def _log_statistics(self):
        """Log statistics for the current round."""
        # Log agent positions and status
        agent_rows = []
        for agent in self.all_agents:
            agent_rows.append({
                'round': self.round,
                'id': agent.id,
                'type': agent.__class__.__name__,
                'pos': agent.pos,
                'status': agent.status,
                'target': agent.target if hasattr(agent, 'target') else None,
                'action': agent.last_action if hasattr(agent, 'last_action') else None,
            })
        
        # Append to agent log
        agent_df = pd.DataFrame(agent_rows)
        agent_df.to_csv(f"{self.output_path}/agent_log.csv", mode='a', header=False, index=False)
        
        # Log overall statistics
        stats_row = {
            'round': self.round,
            'num_casualties': self.num_casualties,
            'num_escaped': self.num_escaped,
            'shooter_neutralized': self.shooter_neutralized
        }
        
        # Append to stats log
        stats_df = pd.DataFrame([stats_row])
        stats_df.to_csv(f"{self.output_path}/stats_log.csv", mode='a', header=False, index=False)
    
    def render(self, show=True, save=True):
        """Visualize the current state of the simulation."""
        plt.figure(figsize=(10, 10))
        
        # Create a grid for visualization
        grid_data = np.zeros((self.height, self.width))
        
        # Define cell types
        # 0: Empty, 1: Wall, 2: Obstacle, 3: Exit
        # 4: Civilian (active), 5: Civilian (hiding), 6: Civilian (casualty)
        # 7: Shooter, 8: Responder
        
        for i in range(self.height):
            for j in range(self.width):
                cell = self.grid[i][j]
                if isinstance(cell, Wall):
                    grid_data[i, j] = 1
                elif isinstance(cell, Obstacle):
                    grid_data[i, j] = 2
                elif isinstance(cell, Exit):
                    grid_data[i, j] = 3
                elif isinstance(cell, Civilian):
                    if cell.status == "active":
                        grid_data[i, j] = 4
                    elif cell.status == "hiding":
                        grid_data[i, j] = 5
                    elif cell.status == "casualty":
                        grid_data[i, j] = 6
                    elif cell.status == "escaped":
                        grid_data[i, j] = 0  # Empty cell
                elif isinstance(cell, Shooter):
                    if cell.status == "active":
                        grid_data[i, j] = 7
                    else:
                        grid_data[i, j] = 0
                elif isinstance(cell, Responder):
                    grid_data[i, j] = 8
        
        # Define colors for different cell types
        colors = [
            'white',      # Empty
            'black',      # Wall
            'gray',       # Obstacle
            'green',      # Exit
            'blue',       # Civilian (active)
            'lightblue',  # Civilian (hiding)
            'red',        # Civilian (casualty)
            'darkred',    # Shooter
            'purple'      # Responder
        ]
        
        cmap = ListedColormap(colors)
        
        # Plot the grid
        plt.imshow(grid_data, cmap=cmap, interpolation='nearest')
        
        # Add a legend
        import matplotlib.patches as mpatches
        legend_elements = [
            mpatches.Patch(color='white', label='Empty'),
            mpatches.Patch(color='black', label='Wall'),
            mpatches.Patch(color='gray', label='Obstacle'),
            mpatches.Patch(color='green', label='Exit'),
            mpatches.Patch(color='blue', label='Civilian (active)'),
            mpatches.Patch(color='lightblue', label='Civilian (hiding)'),
            mpatches.Patch(color='red', label='Civilian (casualty)'),
            mpatches.Patch(color='darkred', label='Shooter'),
            mpatches.Patch(color='purple', label='Responder')
        ]
        
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid lines
        plt.grid(True, color='black', linewidth=0.5, alpha=0.3)
        
        # Add stats as text
        plt.figtext(0.01, 0.01, f"Round: {self.round}\nCasualties: {self.num_casualties}\nEscaped: {self.num_escaped}")
        
        if save:
            plt.savefig(f"{self.output_path}/round_{self.round:04d}.png", bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()