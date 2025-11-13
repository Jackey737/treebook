from __future__ import annotations

import heapq
from typing import Dict, Hashable, Iterable, List, Tuple, Union

Node = Hashable
Weight = Union[int, float]
Edge = Tuple[Node, Weight]
Graph = Dict[Node, Iterable[Edge]]


def shortest_path(graph: Graph, source: Node, target: Node) -> Tuple[Weight, List[Node]]:
    """
    Compute the shortest path between two nodes in a weighted graph using Dijkstra's algorithm.

    Args:
        graph: A mapping of nodes to an iterable of (neighbor, weight) edges.
               Edge weights must be non-negative.
        source: The node to start from.
        target: The node to reach.

    Returns:
        A tuple of the total distance and the list of nodes representing the shortest path.

    Raises:
        KeyError: If the source node does not exist in the graph.
        ValueError: If an edge with a negative weight is encountered.
    """
    if source not in graph:
        raise KeyError(f"Source node {source!r} is not present in the graph.")

    distances: Dict[Node, Weight] = {source: 0}
    previous: Dict[Node, Node] = {}
    frontier: List[Tuple[Weight, Node]] = [(0, source)]

    while frontier:
        current_distance, current_node = heapq.heappop(frontier)

        # Skip if we have already found a better path.
        if current_distance != distances.get(current_node, float("inf")):
            continue

        if current_node == target:
            break

        for neighbor, weight in graph.get(current_node, []):
            if weight < 0:
                raise ValueError("Dijkstra's algorithm does not support negative weight edges.")

            new_distance = current_distance + weight
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(frontier, (new_distance, neighbor))

    if target not in distances:
        return float("inf"), []

    path: List[Node] = [target]
    while path[-1] != source:
        path.append(previous[path[-1]])
    path.reverse()

    return distances[target], path


__all__ = ["shortest_path"]

