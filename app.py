import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import io
import base64
from typing import Dict, List, Tuple, Any

# Import the simulation model
from model.society import ActiveShooterSociety

def main():
    # Page configuration
    st.set_page_config(
        page_title="Active Shooter Simulation",
        page_icon="🚨",
        layout="wide",
    )
    
    # Application title
    st.title("Active Shooter Scenario Simulation")
    st.markdown("""
    This application simulates an active shooter scenario using agent-based modeling 
    enhanced by LLM-powered decision making.
    """)
    
    # Sidebar for simulation parameters
    with st.sidebar:
        st.header("Simulation Parameters")
        
        # Building parameters
        st.subheader("Environment")
        layout_type = st.selectbox(
            "Building Layout",
            ["school", "office", "mall", "simple"],
            index=0,
            help="Type of building layout to simulate"
        )
        
        width = st.slider("Width", min_value=20, max_value=100, value=50)
        height = st.slider("Height", min_value=20, max_value=100, value=50)
        
        # Agent parameters
        st.subheader("Agents")
        num_civilians = st.slider("Number of Civilians", min_value=10, max_value=500, value=100)
        num_shooters = st.slider("Number of Shooters", min_value=1, max_value=5, value=1)
        num_responders = st.slider("Number of Responders", min_value=1, max_value=20, value=5)
        responder_arrival_time = st.slider("Responder Arrival Time (rounds)", min_value=0, max_value=50, value=10)
        
        # Simulation parameters
        st.subheader("Simulation")
        seed = st.number_input("Random Seed", min_value=0, value=42)
        max_rounds = st.slider("Maximum Rounds", min_value=10, max_value=200, value=100)
        
        # LLM parameters
        st.subheader("LLM Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox(
            "LLM Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0
        )
        
        # Run button
        run_simulation = st.button("Run Simulation")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Simulation visualization area
        st.header("Simulation Visualization")
        visualization_placeholder = st.empty()
        
        # Controls for the visualization
        play_pause = st.button("Play/Pause")
        step_forward = st.button("Step Forward")
        reset = st.button("Reset")
        
        # Slider for simulation speed
        simulation_speed = st.slider("Simulation Speed", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        
        # Display current round
        round_display = st.empty()
    
    with col2:
        # Statistics and information
        st.header("Statistics")
        
        # Key metrics
        metrics_container = st.container()
        
        # Agent status breakdown
        agent_status_container = st.container()
        
        # Timeline of events
        st.subheader("Timeline of Events")
        timeline_container = st.empty()
    
    # Initialize session state
    if "society" not in st.session_state:
        st.session_state.society = None
    if "current_round" not in st.session_state:
        st.session_state.current_round = 0
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "simulation_complete" not in st.session_state:
        st.session_state.simulation_complete = False
    if "output_path" not in st.session_state:
        st.session_state.output_path = "output/active_shooter_simulation"
    
    # Run the simulation
    if run_simulation:
        # Create output directory
        output_path = f"output/active_shooter_sim_{int(time.time())}"
        os.makedirs(output_path, exist_ok=True)
        st.session_state.output_path = output_path
        
        # Initialize the simulation
        st.session_state.society = ActiveShooterSociety(
            name="ActiveShooterSim",
            width=width,
            height=height,
            num_civilians=num_civilians,
            num_shooters=num_shooters,
            num_responders=num_responders,
            responder_arrival_time=responder_arrival_time,
            seed=seed,
            llm_model=model,
            api_key=api_key,
            output_path=output_path
        )
        
        # Generate building layout
        st.session_state.society.generate_building_layout(layout_type)
        
        # Populate agents
        st.session_state.society.populate_agents()
        
        # Reset simulation state
        st.session_state.current_round = 0
        st.session_state.is_running = False
        st.session_state.simulation_complete = False
        
        # Render initial state
        st.session_state.society.render(show=False, save=True)
        
        # Display message
        st.success("Simulation initialized successfully. Use the controls to run the simulation.")
    
    # Handle play/pause button
    if play_pause:
        st.session_state.is_running = not st.session_state.is_running
    
    # Handle step forward button
    if step_forward and st.session_state.society and not st.session_state.simulation_complete:
        # Run one step of the simulation
        is_active = st.session_state.society.step()
        st.session_state.current_round += 1
        
        # Check if simulation is complete
        if not is_active or st.session_state.current_round >= max_rounds:
            st.session_state.simulation_complete = True
            st.session_state.is_running = False
            st.info("Simulation complete.")
    
    # Handle reset button
    if reset and st.session_state.society:
        # Reset by reinitializing the simulation
        run_simulation = True
    
    # Auto-play if running
    if st.session_state.is_running and st.session_state.society and not st.session_state.simulation_complete:
        # Run one step of the simulation
        is_active = st.session_state.society.step()
        st.session_state.current_round += 1
        
        # Adjust for simulation speed
        time.sleep(1.0 / simulation_speed)
        
        # Check if simulation is complete
        if not is_active or st.session_state.current_round >= max_rounds:
            st.session_state.simulation_complete = True
            st.session_state.is_running = False
            st.info("Simulation complete.")
        
        # Rerun to update the UI
        st.experimental_rerun()
    
    # Display the current visualization
    if st.session_state.society:
        # Display current round
        round_display.markdown(f"**Current Round:** {st.session_state.current_round}")
        
        # Load and display the visualization
        try:
            img_path = f"{st.session_state.output_path}/round_{st.session_state.current_round:04d}.png"
            img = Image.open(img_path)
            visualization_placeholder.image(img, use_column_width=True)
        except FileNotFoundError:
            visualization_placeholder.info("Visualization not available for this round.")
        
        # Update statistics
        if os.path.exists(f"{st.session_state.output_path}/stats_log.csv"):
            stats_df = pd.read_csv(f"{st.session_state.output_path}/stats_log.csv")
            if not stats_df.empty:
                # Get the latest stats
                latest_stats = stats_df.iloc[-1]
                
                # Update metrics
                with metrics_container:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Casualties", latest_stats["num_casualties"])
                    col2.metric("Escaped", latest_stats["num_escaped"])
                    col3.metric("Shooter Neutralized", "Yes" if latest_stats["shooter_neutralized"] else "No")
                
                # Update agent status breakdown
                with agent_status_container:
                    if os.path.exists(f"{st.session_state.output_path}/agent_log.csv"):
                        agent_df = pd.read_csv(f"{st.session_state.output_path}/agent_log.csv")
                        current_round_agents = agent_df[agent_df["round"] == st.session_state.current_round]
                        
                        # Count agents by type and status
                        agent_counts = current_round_agents.groupby(["type", "status"]).size().unstack().fillna(0)
                        
                        # Plot status breakdown
                        if not agent_counts.empty:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            agent_counts.plot(kind="bar", stacked=True, ax=ax)
                            ax.set_xlabel("Agent Type")
                            ax.set_ylabel("Count")
                            ax.set_title("Agent Status by Type")
                            ax.legend(title="Status")
                            st.pyplot(fig)
                
                # Update timeline of events
                timeline_df = stats_df[["round", "num_casualties", "num_escaped", "shooter_neutralized"]]
                timeline_df = timeline_df.rename(columns={
                    "round": "Round",
                    "num_casualties": "Casualties",
                    "num_escaped": "Escaped",
                    "shooter_neutralized": "Shooter Neutralized"
                })
                timeline_container.dataframe(timeline_df, height=300)
    
    # Information about the simulation
    st.markdown("""
    ### About This Simulation
    
    This simulation uses agent-based modeling to simulate an active shooter scenario. Each agent 
    (civilians, shooters, and responders) makes decisions based on their own knowledge and situation, 
    powered by Large Language Models (LLMs).
    
    #### Agent Types:
    - **Civilians**: Try to escape, hide, or in rare cases, fight
    - **Shooters**: Move through the environment targeting civilians or responders
    - **Responders**: Arrive after a delay to neutralize shooters and help civilians
    
    #### Research Purpose:
    This simulation is intended for research and training purposes only, to help understand 
    emergency response patterns and improve safety protocols.
    """)

if __name__ == "__main__":
    main()