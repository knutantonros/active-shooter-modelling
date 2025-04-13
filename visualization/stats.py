import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Set
import os
import json
from collections import Counter, defaultdict

class SimulationStats:
    """Class for collecting and analyzing simulation statistics."""
    
    def __init__(self, output_path: str = "output"):
        self.output_path = output_path
        self.round_data = []
        self.agent_data = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
    
    def log_round_stats(self, round_num: int, stats: Dict[str, Any]) -> None:
        """Log statistics for a simulation round.
        
        Args:
            round_num: Round number
            stats: Dictionary of statistics for the round
        """
        # Add round number to stats
        stats["round"] = round_num
        
        # Store in round data
        self.round_data.append(stats)
        
        # Write to CSV
        df = pd.DataFrame([stats])
        if round_num == 0:
            df.to_csv(f"{self.output_path}/round_stats.csv", index=False)
        else:
            df.to_csv(f"{self.output_path}/round_stats.csv", mode='a', header=False, index=False)
    
    def log_agent_stats(self, round_num: int, agent_stats: List[Dict[str, Any]]) -> None:
        """Log statistics for agents in a simulation round.
        
        Args:
            round_num: Round number
            agent_stats: List of dictionaries with agent statistics
        """
        # Add round number to each agent's stats
        for stats in agent_stats:
            stats["round"] = round_num
            self.agent_data.append(stats)
        
        # Write to CSV
        df = pd.DataFrame(agent_stats)
        if round_num == 0 and agent_stats:
            df.to_csv(f"{self.output_path}/agent_stats.csv", index=False)
        elif agent_stats:
            df.to_csv(f"{self.output_path}/agent_stats.csv", mode='a', header=False, index=False)
    
    def generate_reports(self) -> None:
        """Generate summary reports and visualizations."""
        if not self.round_data:
            print("No data available to generate reports.")
            return
        
        # Load data from CSV (in case there's more data than in memory)
        try:
            round_df = pd.read_csv(f"{self.output_path}/round_stats.csv")
            agent_df = pd.read_csv(f"{self.output_path}/agent_stats.csv")
        except FileNotFoundError:
            round_df = pd.DataFrame(self.round_data)
            agent_df = pd.DataFrame(self.agent_data)
        
        # Generate summary statistics
        self._generate_summary_stats(round_df, agent_df)
        
        # Generate visualizations
        self._generate_visualizations(round_df, agent_df)
    
    def _generate_summary_stats(self, round_df: pd.DataFrame, agent_df: pd.DataFrame) -> None:
        """Generate summary statistics.
        
        Args:
            round_df: DataFrame with round statistics
            agent_df: DataFrame with agent statistics
        """
        # Basic summary stats
        summary = {
            "total_rounds": round_df["round"].max() + 1,
            "total_agents": agent_df["id"].nunique() if "id" in agent_df.columns else 0,
            "final_casualties": round_df["num_casualties"].iloc[-1] if "num_casualties" in round_df.columns else 0,
            "final_escaped": round_df["num_escaped"].iloc[-1] if "num_escaped" in round_df.columns else 0,
            "shooter_neutralized": round_df["shooter_neutralized"].iloc[-1] if "shooter_neutralized" in round_df.columns else False,
        }
        
        # Agent type breakdown
        if "type" in agent_df.columns:
            agent_types = agent_df[agent_df["round"] == agent_df["round"].max()]["type"].value_counts().to_dict()
            summary["agent_type_counts"] = agent_types
        
        # Agent status breakdown
        if "status" in agent_df.columns:
            final_statuses = agent_df[agent_df["round"] == agent_df["round"].max()]["status"].value_counts().to_dict()
            summary["final_status_counts"] = final_statuses
        
        # Time to neutralize shooter
        if "shooter_neutralized" in round_df.columns:
            neutralized_rounds = round_df[round_df["shooter_neutralized"] == True]["round"]
            if not neutralized_rounds.empty:
                summary["time_to_neutralize"] = neutralized_rounds.min()
        
        # Save summary to JSON
        with open(f"{self.output_path}/summary_stats.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Print summary
        print("\nSimulation Summary:")
        print(f"Total Rounds: {summary['total_rounds']}")
        print(f"Total Agents: {summary['total_agents']}")
        print(f"Final Casualties: {summary['final_casualties']}")
        print(f"Final Escaped: {summary['final_escaped']}")
        print(f"Shooter Neutralized: {summary['shooter_neutralized']}")
        
        if "time_to_neutralize" in summary:
            print(f"Time to Neutralize Shooter: {summary['time_to_neutralize']} rounds")
    
    def _generate_visualizations(self, round_df: pd.DataFrame, agent_df: pd.DataFrame) -> None:
        """Generate visualizations.
        
        Args:
            round_df: DataFrame with round statistics
            agent_df: DataFrame with agent statistics
        """
        # Plot casualties and escapes over time
        if "num_casualties" in round_df.columns and "num_escaped" in round_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(round_df["round"], round_df["num_casualties"], label="Casualties")
            plt.plot(round_df["round"], round_df["num_escaped"], label="Escaped")
            plt.xlabel("Round")
            plt.ylabel("Count")
            plt.title("Casualties and Escapes Over Time")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{self.output_path}/casualties_escapes.png")
            plt.close()
        
        # Plot agent status distribution over time
        if "status" in agent_df.columns and "type" in agent_df.columns:
            # Create a pivot table of agent status counts by round
            status_counts = agent_df.pivot_table(
                index="round", 
                columns=["type", "status"], 
                values="id", 
                aggfunc="count", 
                fill_value=0
            )
            
            # Plot for civilians
            if ("Civilian", "active") in status_counts.columns:
                civilian_status = status_counts[("Civilian",)]
                plt.figure(figsize=(10, 6))
                for status in civilian_status.columns.levels[1]:
                    if (("Civilian", status) in status_counts.columns):
                        plt.plot(civilian_status.index, civilian_status[status], label=f"{status}")
                plt.xlabel("Round")
                plt.ylabel("Count")
                plt.title("Civilian Status Over Time")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{self.output_path}/civilian_status.png")
                plt.close()
        
        # Generate heatmap of agent positions
        if "pos" in agent_df.columns:
            # Extract positions and create a heatmap
            try:
                # Positions might be stored as strings like "(x, y)"
                positions = agent_df["pos"].apply(eval)
                # Extract x and y coordinates
                x_coords = positions.apply(lambda p: p[0])
                y_coords = positions.apply(lambda p: p[1])
                
                # Create a grid for the heatmap
                max_x = x_coords.max()
                max_y = y_coords.max()
                grid = np.zeros((max_x + 1, max_y + 1))
                
                # Count agents at each position
                for x, y in zip(x_coords, y_coords):
                    grid[x, y] += 1
                
                # Plot heatmap
                plt.figure(figsize=(10, 10))
                plt.imshow(grid, cmap='hot', interpolation='nearest')
                plt.colorbar(label='Agent Count')
                plt.title("Agent Position Heatmap")
                plt.xlabel("Column")
                plt.ylabel("Row")
                plt.savefig(f"{self.output_path}/position_heatmap.png")
                plt.close()
            except:
                print("Could not generate position heatmap due to format issues.")
    
    def generate_evacuation_analysis(self) -> Dict[str, Any]:
        """Analyze evacuation patterns and effectiveness.
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Load data
            round_df = pd.read_csv(f"{self.output_path}/round_stats.csv")
            agent_df = pd.read_csv(f"{self.output_path}/agent_stats.csv")
            
            # Total civilians
            total_civilians = agent_df[
                (agent_df["type"] == "Civilian") & (agent_df["round"] == 0)
            ].shape[0]
            
            # Evacuation rate (percentage of civilians that escaped)
            final_escaped = round_df["num_escaped"].iloc[-1] if "num_escaped" in round_df.columns else 0
            evacuation_rate = (final_escaped / total_civilians) * 100 if total_civilians > 0 else 0
            
            # Evacuation speed (average rounds to escape per civilian)
            # First, find the round at which each civilian escaped
            if "status" in agent_df.columns:
                escape_rounds = {}
                for _, group in agent_df[agent_df["type"] == "Civilian"].groupby("id"):
                    escaped_rows = group[group["status"] == "escaped"]
                    if not escaped_rows.empty:
                        escape_rounds[group["id"].iloc[0]] = escaped_rows["round"].min()
                
                avg_escape_time = np.mean(list(escape_rounds.values())) if escape_rounds else None
            else:
                avg_escape_time = None
            
            # Analysis by exit (if exit information is available)
            exit_analysis = {}
            if "target_exit" in agent_df.columns:
                for exit_pos, group in agent_df[
                    (agent_df["type"] == "Civilian") & (agent_df["status"] == "escaped")
                ].groupby("target_exit"):
                    exit_analysis[str(exit_pos)] = {
                        "count": group.shape[0],
                        "percentage": (group.shape[0] / final_escaped) * 100 if final_escaped > 0 else 0
                    }
            
            # Compile results
            analysis = {
                "total_civilians": total_civilians,
                "total_escaped": final_escaped,
                "evacuation_rate": evacuation_rate,
                "avg_escape_time": avg_escape_time,
                "exit_usage": exit_analysis
            }
            
            # Save to JSON
            with open(f"{self.output_path}/evacuation_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=4)
            
            return analysis
            
        except Exception as e:
            print(f"Error generating evacuation analysis: {e}")
            return {}
    
    def generate_shooter_analysis(self) -> Dict[str, Any]:
        """Analyze shooter behavior and effectiveness.
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Load data
            agent_df = pd.read_csv(f"{self.output_path}/agent_stats.csv")
            
            # Filter for shooter data
            shooter_df = agent_df[agent_df["type"] == "Shooter"]
            
            if shooter_df.empty:
                print("No shooter data available for analysis.")
                return {}
            
            # Movement patterns
            shooter_positions = {}
            for shooter_id, group in shooter_df.groupby("id"):
                # Get positions in order of rounds
                positions = []
                for _, row in group.sort_values("round").iterrows():
                    if isinstance(row["pos"], str):
                        positions.append(eval(row["pos"]))
                    else:
                        positions.append(row["pos"])
                
                shooter_positions[shooter_id] = positions
            
            # Calculate distances moved
            shooter_movements = {}
            for shooter_id, positions in shooter_positions.items():
                distances = []
                for i in range(1, len(positions)):
                    # Calculate Euclidean distance between consecutive positions
                    x1, y1 = positions[i-1]
                    x2, y2 = positions[i]
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    distances.append(dist)
                
                shooter_movements[shooter_id] = {
                    "total_distance": sum(distances),
                    "avg_distance_per_round": np.mean(distances) if distances else 0,
                    "stationary_rounds": distances.count(0) if distances else 0
                }
            
            # Shooting behavior
            shooter_actions = {}
            if "action" in shooter_df.columns:
                for shooter_id, group in shooter_df.groupby("id"):
                    shoot_actions = group[group["action"] == "shoot"]
                    shooter_actions[shooter_id] = {
                        "total_shots": shoot_actions.shape[0],
                        "shots_per_round": shoot_actions.shape[0] / len(group) if len(group) > 0 else 0
                    }
            
            # Compile results
            analysis = {
                "shooter_count": shooter_df["id"].nunique(),
                "shooter_movements": shooter_movements,
                "shooter_actions": shooter_actions
            }
            
            # Save to JSON
            with open(f"{self.output_path}/shooter_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=4)
            
            return analysis
            
        except Exception as e:
            print(f"Error generating shooter analysis: {e}")
            return {}
    
    def generate_responder_analysis(self) -> Dict[str, Any]:
        """Analyze responder effectiveness.
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Load data
            round_df = pd.read_csv(f"{self.output_path}/round_stats.csv")
            agent_df = pd.read_csv(f"{self.output_path}/agent_stats.csv")
            
            # Filter for responder data
            responder_df = agent_df[agent_df["type"] == "Responder"]
            
            if responder_df.empty:
                print("No responder data available for analysis.")
                return {}
            
            # Time to neutralize shooter
            neutralized_round = None
            if "shooter_neutralized" in round_df.columns:
                neutralized_rows = round_df[round_df["shooter_neutralized"] == True]
                if not neutralized_rows.empty:
                    neutralized_round = neutralized_rows["round"].min()
            
            # Responder arrival time
            first_responder_round = responder_df["round"].min()
            
            # Time from arrival to neutralization
            time_to_neutralize = None
            if neutralized_round is not None and first_responder_round is not None:
                time_to_neutralize = neutralized_round - first_responder_round
            
            # Responder casualties
            responder_casualties = 0
            if "status" in responder_df.columns:
                responder_casualties = responder_df[responder_df["status"] == "casualty"]["id"].nunique()
            
            # Movement patterns
            responder_positions = {}
            for responder_id, group in responder_df.groupby("id"):
                # Get positions in order of rounds
                positions = []
                for _, row in group.sort_values("round").iterrows():
                    if isinstance(row["pos"], str):
                        positions.append(eval(row["pos"]))
                    else:
                        positions.append(row["pos"])
                
                responder_positions[responder_id] = positions
            
            # Calculate search efficiency (ratio of area covered to total grid area)
            # This is a simplification - a more sophisticated analysis would consider visibility cones
            unique_positions = set()
            for positions in responder_positions.values():
                unique_positions.update(positions)
            
            max_x = max(p[0] for p in unique_positions) if unique_positions else 0
            max_y = max(p[1] for p in unique_positions) if unique_positions else 0
            total_area = (max_x + 1) * (max_y + 1)
            search_efficiency = len(unique_positions) / total_area if total_area > 0 else 0
            
            # Compile results
            analysis = {
                "responder_count": responder_df["id"].nunique(),
                "first_responder_round": first_responder_round,
                "neutralized_round": neutralized_round,
                "time_to_neutralize_after_arrival": time_to_neutralize,
                "responder_casualties": responder_casualties,
                "responder_casualties_percentage": (responder_casualties / responder_df["id"].nunique()) * 100,
                "search_efficiency": search_efficiency,
                "search_efficiency_percentage": search_efficiency * 100
            }
            
            # Save to JSON
            with open(f"{self.output_path}/responder_analysis.json", 'w') as f:
                json.dump(analysis, f, indent=4)
            
            return analysis
            
        except Exception as e:
            print(f"Error generating responder analysis: {e}")
            return {}
    
    def generate_comprehensive_report(self) -> None:
        """Generate a comprehensive report combining all analyses."""
        # Generate individual analyses
        self.generate_reports()
        evacuation_analysis = self.generate_evacuation_analysis()
        shooter_analysis = self.generate_shooter_analysis()
        responder_analysis = self.generate_responder_analysis()
        
        # Load summary statistics
        try:
            with open(f"{self.output_path}/summary_stats.json", 'r') as f:
                summary_stats = json.load(f)
        except FileNotFoundError:
            summary_stats = {}
        
        # Combine all analyses
        comprehensive_report = {
            "summary": summary_stats,
            "evacuation": evacuation_analysis,
            "shooter": shooter_analysis,
            "responder": responder_analysis
        }
        
        # Save to JSON
        with open(f"{self.output_path}/comprehensive_report.json", 'w') as f:
            json.dump(comprehensive_report, f, indent=4)
        
        # Generate a human-readable text report
        self._generate_text_report(comprehensive_report)
    
    def _generate_text_report(self, report: Dict[str, Any]) -> None:
        """Generate a human-readable text report.
        
        Args:
            report: Combined report dictionary
        """
        with open(f"{self.output_path}/simulation_report.txt", 'w') as f:
            f.write("=======================================\n")
            f.write("ACTIVE SHOOTER SIMULATION REPORT\n")
            f.write("=======================================\n\n")
            
            # Summary section
            f.write("SUMMARY\n")
            f.write("-------\n")
            if "summary" in report and report["summary"]:
                summary = report["summary"]
                f.write(f"Total Rounds: {summary.get('total_rounds', 'N/A')}\n")
                f.write(f"Total Agents: {summary.get('total_agents', 'N/A')}\n")
                f.write(f"Final Casualties: {summary.get('final_casualties', 'N/A')}\n")
                f.write(f"Final Escaped: {summary.get('final_escaped', 'N/A')}\n")
                f.write(f"Shooter Neutralized: {summary.get('shooter_neutralized', 'N/A')}\n")
                
                if "time_to_neutralize" in summary:
                    f.write(f"Time to Neutralize Shooter: {summary['time_to_neutralize']} rounds\n")
            else:
                f.write("No summary data available.\n")
            
            f.write("\n")
            
            # Evacuation section
            f.write("EVACUATION ANALYSIS\n")
            f.write("-------------------\n")
            if "evacuation" in report and report["evacuation"]:
                evac = report["evacuation"]
                f.write(f"Total Civilians: {evac.get('total_civilians', 'N/A')}\n")
                f.write(f"Total Escaped: {evac.get('total_escaped', 'N/A')}\n")
                f.write(f"Evacuation Rate: {evac.get('evacuation_rate', 'N/A'):.1f}%\n")
                f.write(f"Average Escape Time: {evac.get('avg_escape_time', 'N/A')} rounds\n")
                
                if "exit_usage" in evac and evac["exit_usage"]:
                    f.write("\nExit Usage:\n")
                    for exit_pos, data in evac["exit_usage"].items():
                        f.write(f"  {exit_pos}: {data['count']} civilians ({data['percentage']:.1f}%)\n")
            else:
                f.write("No evacuation data available.\n")
            
            f.write("\n")
            
            # Shooter section
            f.write("SHOOTER ANALYSIS\n")
            f.write("----------------\n")
            if "shooter" in report and report["shooter"]:
                shooter = report["shooter"]
                f.write(f"Shooter Count: {shooter.get('shooter_count', 'N/A')}\n")
                
                if "shooter_movements" in shooter and shooter["shooter_movements"]:
                    f.write("\nShooter Movements:\n")
                    for shooter_id, data in shooter["shooter_movements"].items():
                        f.write(f"  Shooter {shooter_id}:\n")
                        f.write(f"    Total Distance Moved: {data.get('total_distance', 'N/A'):.1f}\n")
                        f.write(f"    Average Distance per Round: {data.get('avg_distance_per_round', 'N/A'):.1f}\n")
                        f.write(f"    Stationary Rounds: {data.get('stationary_rounds', 'N/A')}\n")
                
                if "shooter_actions" in shooter and shooter["shooter_actions"]:
                    f.write("\nShooter Actions:\n")
                    for shooter_id, data in shooter["shooter_actions"].items():
                        f.write(f"  Shooter {shooter_id}:\n")
                        f.write(f"    Total Shots: {data.get('total_shots', 'N/A')}\n")
                        f.write(f"    Shots per Round: {data.get('shots_per_round', 'N/A'):.2f}\n")
            else:
                f.write("No shooter data available.\n")
            
            f.write("\n")
            
            # Responder section
            f.write("RESPONDER ANALYSIS\n")
            f.write("------------------\n")
            if "responder" in report and report["responder"]:
                resp = report["responder"]
                f.write(f"Responder Count: {resp.get('responder_count', 'N/A')}\n")
                f.write(f"First Responder Arrival: Round {resp.get('first_responder_round', 'N/A')}\n")
                f.write(f"Shooter Neutralized: {'No' if resp.get('neutralized_round') is None else f'Yes (Round {resp.get('neutralized_round')})'}\n")
                
                if resp.get("time_to_neutralize_after_arrival") is not None:
                    f.write(f"Time to Neutralize After Arrival: {resp['time_to_neutralize_after_arrival']} rounds\n")
                
                f.write(f"Responder Casualties: {resp.get('responder_casualties', 'N/A')} ({resp.get('responder_casualties_percentage', 'N/A'):.1f}%)\n")
                f.write(f"Search Efficiency: {resp.get('search_efficiency_percentage', 'N/A'):.1f}% of area covered\n")
            else:
                f.write("No responder data available.\n")
            
            f.write("\n")
            
            # Conclusion
            f.write("CONCLUSION\n")
            f.write("----------\n")
            # Generate a simple conclusion based on the data
            try:
                evac_rate = report.get("evacuation", {}).get("evacuation_rate", 0)
                shooter_neutralized = report.get("summary", {}).get("shooter_neutralized", False)
                casualties = report.get("summary", {}).get("final_casualties", 0)
                
                if shooter_neutralized and evac_rate > 80:
                    f.write("The response was highly effective. The shooter was neutralized and most civilians escaped safely.\n")
                elif shooter_neutralized and evac_rate > 50:
                    f.write("The response was moderately effective. The shooter was neutralized but many civilians were unable to escape.\n")
                elif shooter_neutralized:
                    f.write("The response was partially effective. The shooter was neutralized but most civilians were unable to escape.\n")
                elif evac_rate > 80:
                    f.write("The evacuation was successful but the shooter was not neutralized.\n")
                elif evac_rate > 50:
                    f.write("The evacuation was partially successful but the shooter was not neutralized.\n")
                else:
                    f.write("The response was ineffective. Most civilians were unable to escape and the shooter was not neutralized.\n")
                
                f.write(f"\nFinal statistics: {casualties} casualties, {evac_rate:.1f}% evacuation rate, shooter neutralized: {shooter_neutralized}.\n")
            except:
                f.write("Insufficient data to generate conclusion.\n")
            
            f.write("\n=======================================\n")