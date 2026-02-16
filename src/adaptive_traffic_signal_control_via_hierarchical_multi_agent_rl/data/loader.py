"""Data loading utilities for traffic simulation scenarios."""

import logging
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import traci
import sumolib

from ..utils.config import Config

logger = logging.getLogger(__name__)


class TrafficDataLoader:
    """Base class for traffic data loading."""

    def __init__(self, config: Config) -> None:
        """Initialize data loader.

        Args:
            config: Configuration object.
        """
        self.config = config
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def load_scenario(self, scenario_name: str) -> Dict[str, any]:
        """Load traffic scenario data.

        Args:
            scenario_name: Name of the scenario to load.

        Returns:
            Dictionary containing scenario data.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load_scenario")

    def get_available_scenarios(self) -> List[str]:
        """Get list of available scenarios.

        Returns:
            List of scenario names.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement get_available_scenarios")


class SUMODataLoader(TrafficDataLoader):
    """Data loader for SUMO traffic simulation scenarios."""

    def __init__(self, config: Config) -> None:
        """Initialize SUMO data loader.

        Args:
            config: Configuration object.
        """
        super().__init__(config)
        self.sumo_binary = config.data.sumo.binary_path
        self.config_dir = Path(config.data.sumo.config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)

        # Validate SUMO installation
        self._validate_sumo()

    def _validate_sumo(self) -> None:
        """Validate SUMO installation."""
        try:
            result = subprocess.run(
                [self.sumo_binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"SUMO validation successful: {result.stdout.strip()}")
            else:
                raise RuntimeError(f"SUMO validation failed: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"SUMO not found or not working: {e}")

    def load_scenario(self, scenario_name: str) -> Dict[str, any]:
        """Load SUMO scenario data.

        Args:
            scenario_name: Name of the SUMO scenario to load.

        Returns:
            Dictionary containing scenario data including network, routes, and metadata.
        """
        scenario_path = self.config_dir / scenario_name

        if not scenario_path.exists():
            logger.warning(f"Scenario {scenario_name} not found, generating...")
            self._generate_scenario(scenario_name)

        scenario_data = {
            "name": scenario_name,
            "network_file": str(scenario_path / f"{scenario_name}.net.xml"),
            "route_file": str(scenario_path / f"{scenario_name}.rou.xml"),
            "config_file": str(scenario_path / f"{scenario_name}.sumocfg"),
            "additional_file": str(scenario_path / f"{scenario_name}.add.xml"),
        }

        # Load network data
        try:
            network = sumolib.net.readNet(scenario_data["network_file"])
            scenario_data["network"] = network
            scenario_data["intersections"] = self._extract_intersections(network)
            scenario_data["edges"] = self._extract_edges(network)
            scenario_data["topology"] = self._build_network_topology(network)
        except Exception as e:
            logger.error(f"Failed to load network data for {scenario_name}: {e}")
            raise

        # Load route data
        try:
            scenario_data["routes"] = self._load_routes(scenario_data["route_file"])
        except Exception as e:
            logger.warning(f"Failed to load routes for {scenario_name}: {e}")
            scenario_data["routes"] = []

        logger.info(f"Successfully loaded scenario: {scenario_name}")
        return scenario_data

    def _generate_scenario(self, scenario_name: str) -> None:
        """Generate a SUMO scenario.

        Args:
            scenario_name: Name of the scenario to generate.
        """
        if scenario_name == "manhattan_grid":
            self._generate_manhattan_grid()
        elif scenario_name == "cologne":
            self._download_cologne_scenario()
        else:
            raise ValueError(f"Unknown scenario: {scenario_name}")

    def _generate_manhattan_grid(self) -> None:
        """Generate Manhattan grid scenario."""
        grid_size = self.config.environment.grid_size
        scenario_dir = self.config_dir / "manhattan_grid"
        scenario_dir.mkdir(exist_ok=True)

        # Generate network using netgenerate
        net_file = scenario_dir / "manhattan_grid.net.xml"
        cmd = [
            "netgenerate",
            "--grid",
            f"--grid.x-number={grid_size[0]}",
            f"--grid.y-number={grid_size[1]}",
            "--grid.x-length=200",
            "--grid.y-length=200",
            "--grid.attach-length=100",
            "--output-file", str(net_file),
            "--junction-type=traffic_light",
            "--tls.default-type=static",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"netgenerate failed: {result.stderr}")
            logger.info("Generated Manhattan grid network")
        except subprocess.TimeoutExpired:
            raise RuntimeError("netgenerate timed out")

        # Generate routes
        self._generate_random_routes(
            str(net_file),
            str(scenario_dir / "manhattan_grid.rou.xml"),
            num_vehicles=200
        )

        # Create configuration file
        self._create_sumo_config(
            scenario_dir / "manhattan_grid.sumocfg",
            net_file,
            scenario_dir / "manhattan_grid.rou.xml",
            scenario_dir / "manhattan_grid.add.xml"
        )

        # Create additional files (detectors, traffic lights)
        self._create_additional_files(
            str(net_file),
            str(scenario_dir / "manhattan_grid.add.xml")
        )

    def _generate_random_routes(
        self,
        net_file: str,
        route_file: str,
        num_vehicles: int = 200
    ) -> None:
        """Generate random routes for the network.

        Args:
            net_file: Path to network file.
            route_file: Path to output route file.
            num_vehicles: Number of vehicles to generate.
        """
        cmd = [
            "python",
            "/usr/share/sumo/tools/randomTrips.py",
            "-n", net_file,
            "-r", route_file,
            "-e", str(self.config.environment.simulation_time),
            "--vehicle-class", "passenger",
            "--vclass", "passenger",
            "--trip-attributes", 'departLane="best" departSpeed="max"',
            "--random"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.warning(f"randomTrips.py warning: {result.stderr}")
            logger.info(f"Generated {num_vehicles} random routes")
        except subprocess.TimeoutExpired:
            raise RuntimeError("randomTrips.py timed out")

    def _create_sumo_config(
        self,
        config_file: Path,
        net_file: Path,
        route_file: Path,
        add_file: Path
    ) -> None:
        """Create SUMO configuration file.

        Args:
            config_file: Path to output configuration file.
            net_file: Path to network file.
            route_file: Path to route file.
            add_file: Path to additional file.
        """
        config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_file.name}"/>
        <route-files value="{route_file.name}"/>
        <additional-files value="{add_file.name}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{self.config.environment.simulation_time}"/>
        <step-length value="{self.config.environment.step_length}"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <time-to-teleport value="300"/>
    </processing>
    <traci_server>
        <remote-port value="8873"/>
    </traci_server>
</configuration>"""

        with open(config_file, "w") as f:
            f.write(config_content)

        logger.info(f"Created SUMO configuration: {config_file}")

    def _create_additional_files(self, net_file: str, add_file: str) -> None:
        """Create additional files for detectors and traffic lights.

        Args:
            net_file: Path to network file.
            add_file: Path to output additional file.
        """
        network = sumolib.net.readNet(net_file)

        add_content = ['<?xml version="1.0" encoding="UTF-8"?>']
        add_content.append('<additionalFile>')

        # Add induction loop detectors at intersections
        for junction in network.getNodes():
            if junction.getType() == "traffic_light":
                incoming_edges = junction.getIncoming()
                for i, edge in enumerate(incoming_edges):
                    if edge.allows("passenger"):
                        lanes = edge.getLanes()
                        for j, lane in enumerate(lanes):
                            detector_id = f"det_{junction.getID()}_{i}_{j}"
                            add_content.append(
                                f'    <inductionLoop id="{detector_id}" '
                                f'lane="{lane.getID()}" '
                                f'pos="-50" freq="1" file="NUL"/>'
                            )

        add_content.append('</additionalFile>')

        with open(add_file, "w") as f:
            f.write("\n".join(add_content))

        logger.info(f"Created additional files: {add_file}")

    def _download_cologne_scenario(self) -> None:
        """Download and prepare Cologne traffic scenario."""
        scenario_dir = self.config_dir / "cologne"
        scenario_dir.mkdir(exist_ok=True)

        # This would typically download from the SUMO website
        # For now, we'll create a placeholder
        logger.warning("Cologne scenario download not implemented. "
                      "Please download manually from SUMO website.")

        # Create a minimal Cologne-like scenario
        self._generate_cologne_like_scenario(scenario_dir)

    def _generate_cologne_like_scenario(self, scenario_dir: Path) -> None:
        """Generate a Cologne-like scenario for demonstration.

        Args:
            scenario_dir: Directory to create scenario files.
        """
        # This is a simplified version for demonstration
        # In practice, you would use the actual Cologne data
        logger.info("Generating Cologne-like scenario...")

        net_file = scenario_dir / "cologne.net.xml"
        # Generate a more complex grid to simulate Cologne
        cmd = [
            "netgenerate",
            "--grid",
            "--grid.x-number=8",
            "--grid.y-number=6",
            "--grid.x-length=300",
            "--grid.y-length=250",
            "--output-file", str(net_file),
            "--junction-type=traffic_light",
            "--tls.default-type=actuated",
        ]

        subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Generate routes and config files
        self._generate_random_routes(
            str(net_file),
            str(scenario_dir / "cologne.rou.xml"),
            num_vehicles=500
        )

        self._create_sumo_config(
            scenario_dir / "cologne.sumocfg",
            net_file,
            scenario_dir / "cologne.rou.xml",
            scenario_dir / "cologne.add.xml"
        )

        self._create_additional_files(
            str(net_file),
            str(scenario_dir / "cologne.add.xml")
        )

    def _extract_intersections(self, network: sumolib.net.Net) -> List[Dict[str, any]]:
        """Extract intersection information from SUMO network.

        Args:
            network: SUMO network object.

        Returns:
            List of intersection dictionaries.
        """
        intersections = []

        for node in network.getNodes():
            if node.getType() == "traffic_light":
                intersection = {
                    "id": node.getID(),
                    "coord": node.getCoord(),
                    "incoming_edges": [edge.getID() for edge in node.getIncoming()],
                    "outgoing_edges": [edge.getID() for edge in node.getOutgoing()],
                    "tls_id": node.getTLSID() if hasattr(node, 'getTLSID') else None,
                }
                intersections.append(intersection)

        logger.info(f"Extracted {len(intersections)} intersections")
        return intersections

    def _extract_edges(self, network: sumolib.net.Net) -> List[Dict[str, any]]:
        """Extract edge information from SUMO network.

        Args:
            network: SUMO network object.

        Returns:
            List of edge dictionaries.
        """
        edges = []

        for edge in network.getEdges():
            if not edge.isSpecial():  # Skip internal edges
                edge_data = {
                    "id": edge.getID(),
                    "from_node": edge.getFromNode().getID(),
                    "to_node": edge.getToNode().getID(),
                    "length": edge.getLength(),
                    "speed_limit": edge.getSpeed(),
                    "num_lanes": edge.getLaneNumber(),
                    "priority": edge.getPriority(),
                }
                edges.append(edge_data)

        logger.info(f"Extracted {len(edges)} edges")
        return edges

    def _build_network_topology(self, network: sumolib.net.Net) -> nx.DiGraph:
        """Build network topology as NetworkX graph.

        Args:
            network: SUMO network object.

        Returns:
            NetworkX directed graph representing the network topology.
        """
        G = nx.DiGraph()

        # Add nodes (intersections)
        for node in network.getNodes():
            if node.getType() == "traffic_light":
                G.add_node(node.getID(), **{
                    "coord": node.getCoord(),
                    "type": "intersection"
                })

        # Add edges (roads)
        for edge in network.getEdges():
            if not edge.isSpecial():
                from_node = edge.getFromNode().getID()
                to_node = edge.getToNode().getID()

                if G.has_node(from_node) and G.has_node(to_node):
                    G.add_edge(from_node, to_node, **{
                        "edge_id": edge.getID(),
                        "length": edge.getLength(),
                        "speed_limit": edge.getSpeed(),
                        "num_lanes": edge.getLaneNumber(),
                    })

        logger.info(f"Built network topology: {G.number_of_nodes()} nodes, "
                   f"{G.number_of_edges()} edges")
        return G

    def _load_routes(self, route_file: str) -> List[Dict[str, any]]:
        """Load route information from SUMO route file.

        Args:
            route_file: Path to SUMO route file.

        Returns:
            List of route dictionaries.
        """
        routes = []

        try:
            tree = ET.parse(route_file)
            root = tree.getroot()

            for route_elem in root.findall(".//route"):
                route = {
                    "id": route_elem.get("id"),
                    "edges": route_elem.get("edges", "").split(),
                }
                routes.append(route)

            for trip_elem in root.findall(".//trip"):
                trip = {
                    "id": trip_elem.get("id"),
                    "from_edge": trip_elem.get("from"),
                    "to_edge": trip_elem.get("to"),
                    "depart": float(trip_elem.get("depart", 0)),
                }
                routes.append(trip)

        except ET.ParseError as e:
            logger.error(f"Failed to parse route file {route_file}: {e}")

        logger.info(f"Loaded {len(routes)} routes")
        return routes

    def get_available_scenarios(self) -> List[str]:
        """Get list of available SUMO scenarios.

        Returns:
            List of scenario names.
        """
        scenarios = ["manhattan_grid", "cologne"]

        # Check for existing scenario directories
        if self.config_dir.exists():
            for item in self.config_dir.iterdir():
                if item.is_dir() and (item / f"{item.name}.sumocfg").exists():
                    scenarios.append(item.name)

        return list(set(scenarios))

    def start_simulation(self, scenario_data: Dict[str, any]) -> str:
        """Start SUMO simulation.

        Args:
            scenario_data: Scenario data dictionary.

        Returns:
            SUMO connection label.

        Raises:
            RuntimeError: If simulation fails to start.
        """
        config_file = scenario_data["config_file"]
        label = f"sumo_{scenario_data['name']}_{np.random.randint(10000)}"

        try:
            traci.start([
                self.sumo_binary,
                "-c", config_file,
                "--no-warnings",
                "--no-step-log",
                "--time-to-teleport", "300",
                "--waiting-time-memory", "10000"
            ], label=label)

            logger.info(f"Started SUMO simulation: {label}")
            return label

        except Exception as e:
            logger.error(f"Failed to start SUMO simulation: {e}")
            raise RuntimeError(f"SUMO simulation failed: {e}")

    def stop_simulation(self, label: str) -> None:
        """Stop SUMO simulation.

        Args:
            label: SUMO connection label.
        """
        try:
            traci.switch(label)
            traci.close()
            logger.info(f"Stopped SUMO simulation: {label}")
        except Exception as e:
            logger.warning(f"Failed to stop SUMO simulation {label}: {e}")