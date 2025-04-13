import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from typing import List, Dict, Any, Optional, Tuple
import io
from PIL import Image
import os

def render_grid(
    grid: List[List[Any]], 
    title: str = "Simulation",
    show: bool = True,
    save_path: Optional[str] = None,
    highlight_positions: List[Tuple[int, int]] = None,
    highlight_color: str = 'yellow',
    agent_colors: Dict[str, str] = None,
    include_legend: bool = True,
    figsize: Tuple[int, int] = (10, 10)
) -> Optional[Image.Image]:
    """Render the grid as an image.
    
    Args:
        grid: 2D grid to render
        title: Title for the plot
        show: Whether to display the plot
        save_path: Path to save the image (if None, won't save)
        highlight_positions: List of positions to highlight
        highlight_color: Color for highlighted positions
        agent_colors: Custom colors for agent types
        include_legend: Whether to include a legend
        figsize: Figure size (width, height) in inches
        
    Returns:
        Optional[Image.Image]: PIL Image if save_path is None and show is False, otherwise None
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a grid for visualization
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    grid_data = np.zeros((height, width))
    
    # Define default cell type mapping
    # 0: Empty, 1: Wall, 2: Obstacle, 3: Exit, 4: Door, 5: Window
    # 10: Civilian (active), 11: Civilian (hiding), 12: Civilian (casualty), 13: Civilian (escaped)
    # 20: Shooter (active), 21: Shooter (neutralized)
    # 30: Responder (active), 31: Responder (casualty)
    
    # Define default colors for different cell types
    default_colors = {
        0: 'white',       # Empty
        1: 'black',       # Wall
        2: 'gray',        # Obstacle
        3: 'green',       # Exit
        4: 'brown',       # Door
        5: 'lightblue',   # Window
        10: 'blue',       # Civilian (active)
        11: 'lightblue',  # Civilian (hiding)
        12: 'red',        # Civilian (casualty)
        13: 'green',      # Civilian (escaped)
        20: 'darkred',    # Shooter (active)
        21: 'red',        # Shooter (neutralized)
        30: 'purple',     # Responder (active)
        31: 'magenta'     # Responder (casualty)
    }
    
    # Merge with custom colors if provided
    if agent_colors:
        for key, value in agent_colors.items():
            if key in default_colors:
                default_colors[key] = value
    
    # Track used cell types for the legend
    used_cell_types = set()
    
    # Fill the grid data
    for i in range(height):
        for j in range(width):
            cell = grid[i][j]
            
            # Default to empty
            cell_type = 0
            
            # Determine cell type based on class and attributes
            if hasattr(cell, 'name'):
                if cell.name == "wall":
                    cell_type = 1
                    used_cell_types.add(1)
                elif cell.name == "obstacle":
                    cell_type = 2
                    used_cell_types.add(2)
                elif cell.name == "exit":
                    cell_type = 3
                    used_cell_types.add(3)
                elif cell.name == "door":
                    cell_type = 4
                    used_cell_types.add(4)
                elif cell.name == "window":
                    cell_type = 5
                    used_cell_types.add(5)
            
            # Check if it's an agent
            if hasattr(cell, '__class__') and cell.__class__.__name__ == "Civilian":
                if cell.status == "active":
                    cell_type = 10
                    used_cell_types.add(10)
                elif cell.status == "hiding":
                    cell_type = 11
                    used_cell_types.add(11)
                elif cell.status == "casualty":
                    cell_type = 12
                    used_cell_types.add(12)
                elif cell.status == "escaped":
                    cell_type = 13
                    used_cell_types.add(13)
            
            elif hasattr(cell, '__class__') and cell.__class__.__name__ == "Shooter":
                if cell.status == "active":
                    cell_type = 20
                    used_cell_types.add(20)
                else:
                    cell_type = 21
                    used_cell_types.add(21)
            
            elif hasattr(cell, '__class__') and cell.__class__.__name__ == "Responder":
                if cell.status == "active":
                    cell_type = 30
                    used_cell_types.add(30)
                else:
                    cell_type = 31
                    used_cell_types.add(31)
            
            grid_data[i, j] = cell_type
    
    # Highlight specific positions if requested
    if highlight_positions:
        for pos in highlight_positions:
            if 0 <= pos[0] < height and 0 <= pos[1] < width:
                # Add highlight by changing the cell color
                grid_data[pos[0], pos[1]] = 40  # Special value for highlight
                default_colors[40] = highlight_color
                used_cell_types.add(40)
    
    # Create a colormap with used colors
    all_colors = [default_colors.get(i, 'white') for i in range(max(used_cell_types) + 1)]
    cmap = ListedColormap(all_colors)
    
    # Plot the grid
    img = ax.imshow(grid_data, cmap=cmap, interpolation='nearest')
    
    # Add grid lines
    ax.grid(True, color='black', linewidth=0.5, alpha=0.3)
    
    # Set axis labels and title
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # Add a legend
    if include_legend:
        import matplotlib.patches as mpatches
        
        # Define legend labels for used cell types
        legend_labels = {
            0: 'Empty',
            1: 'Wall',
            2: 'Obstacle',
            3: 'Exit',
            4: 'Door',
            5: 'Window',
            10: 'Civilian (active)',
            11: 'Civilian (hiding)',
            12: 'Civilian (casualty)',
            13: 'Civilian (escaped)',
            20: 'Shooter (active)',
            21: 'Shooter (neutralized)',
            30: 'Responder (active)',
            31: 'Responder (casualty)',
            40: 'Highlighted Position'
        }
        
        # Create legend patches for used cell types
        legend_elements = [
            mpatches.Patch(color=default_colors[cell_type], label=legend_labels.get(cell_type, f'Type {cell_type}'))
            for cell_type in sorted(used_cell_types)
        ]
        
        # Add the legend
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    # Show the figure if requested
    if show:
        plt.show()
        return None
    else:
        # If not showing, return as PIL Image if not saving
        if not save_path:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return Image.open(buf)
        else:
            plt.close(fig)
            return None


def create_animation(frames_path: str, output_path: str, fps: int = 5):
    """Create an animation from a series of frames.
    
    Args:
        frames_path: Directory containing frame images
        output_path: Path to save the animation
        fps: Frames per second
    """
    import imageio
    
    # Find all frame files
    frame_files = [f for f in os.listdir(frames_path) if f.endswith('.png') and f.startswith('round_')]
    
    # Sort by frame number
    frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Read frames
    frames = []
    for file in frame_files:
        frames.append(imageio.imread(os.path.join(frames_path, file)))
    
    # Write animation
    imageio.mimsave(output_path, frames, fps=fps)


def render_statistics(
    stats: Dict[str, List[Any]],
    title: str = "Simulation Statistics",
    show: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Render statistics plots.
    
    Args:
        stats: Dictionary of statistics (keys are labels, values are lists of data points)
        title: Title for the plot
        show: Whether to display the plot
        save_path: Path to save the image (if None, won't save)
        figsize: Figure size (width, height) in inches
    """
    # Set up the figure
    fig, axes = plt.subplots(len(stats), 1, figsize=figsize)
    fig.suptitle(title)
    
    # If only one statistic, wrap axes in a list
    if len(stats) == 1:
        axes = [axes]
    
    # Plot each statistic
    for i, (label, values) in enumerate(stats.items()):
        axes[i].plot(values)
        axes[i].set_ylabel(label)
        axes[i].grid(True)
    
    # Set x-label for the bottom subplot
    axes[-1].set_xlabel("Round")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path)
    
    # Show the figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)


def render_heatmap(
    grid: List[List[float]],
    title: str = "Density Heatmap",
    show: bool = True,
    save_path: Optional[str] = None,
    colormap: str = 'hot',
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    """Render a heatmap visualization.
    
    Args:
        grid: 2D grid of values to render as a heatmap
        title: Title for the plot
        show: Whether to display the plot
        save_path: Path to save the image (if None, won't save)
        colormap: Matplotlib colormap name
        figsize: Figure size (width, height) in inches
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the heatmap
    img = ax.imshow(grid, cmap=colormap, interpolation='nearest')
    
    # Add a colorbar
    cbar = plt.colorbar(img, ax=ax)
    
    # Set axis labels and title
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # Save the figure if requested
    if save_path:
        plt.savefig(save_path)
    
    # Show the figure if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
