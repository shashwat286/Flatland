import math
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.step_utils.states import TrainState
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from collections import namedtuple

# RailEnvNextAction was removed in Flatland 4.x — define it locally.
RailEnvNextAction = namedtuple('RailEnvNextAction', ['action', 'next_position', 'next_direction'])

# Flatland 4.x active states (agent is on the grid)
_ACTIVE_STATES = {TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION}

# ---------------------------------------------------------------------------
# Rail transition cache  (per worker process — safe with mp spawn)
# Key: (position, direction)  Value: transitions tuple
# Must be cleared on every env.reset() because the rail layout regenerates.
# ---------------------------------------------------------------------------
_trans_cache: dict = {}

def clear_trans_cache() -> None:
    """Clear the rail transition cache. Call after every env.reset()."""
    _trans_cache.clear()


def get_valid_move_actions_(agent_direction: Grid4TransitionsEnum,
                            agent_position: Tuple[int, int],
                            rail: GridTransitionMap) -> Set[RailEnvNextAction]:
    """Get the valid move actions (forward, left, right) for an agent."""
    valid_actions: Set[RailEnvNextAction] = set()
    _key = (agent_position, agent_direction)
    if _key not in _trans_cache:
        _trans_cache[_key] = rail.get_transitions((agent_position, agent_direction))
    possible_transitions = _trans_cache[_key]
    num_transitions = np.count_nonzero(possible_transitions)

    if rail.is_dead_end(agent_position):
        action = RailEnvActions.MOVE_FORWARD
        exit_direction = (agent_direction + 2) % 4
        if possible_transitions[exit_direction]:
            new_position = get_new_position(agent_position, exit_direction)
            valid_actions.add(RailEnvNextAction(action, new_position, exit_direction))
    elif num_transitions == 1:
        action = RailEnvActions.MOVE_FORWARD
        for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[new_direction]:
                new_position = get_new_position(agent_position, new_direction)
                valid_actions.add(RailEnvNextAction(action, new_position, new_direction))
    else:
        for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[new_direction]:
                if new_direction == agent_direction:
                    action = RailEnvActions.MOVE_FORWARD
                elif new_direction == (agent_direction + 1) % 4:
                    action = RailEnvActions.MOVE_RIGHT
                elif new_direction == (agent_direction - 1) % 4:
                    action = RailEnvActions.MOVE_LEFT
                else:
                    continue
                new_position = get_new_position(agent_position, new_direction)
                valid_actions.add(RailEnvNextAction(action, new_position, new_direction))
    return valid_actions


def get_shortest_paths(distance_map,
                       agent_pos=None,
                       agent_dir=None,
                       max_depth: Optional[int] = None,
                       agent_handle: Optional[int] = None) -> Dict[int, Optional[List[Waypoint]]]:
    """
    Computes the shortest path for each agent to its target.
    Uses Flatland 4.x TrainState API (flatland.envs.step_utils.states).
    """
    shortest_paths = dict()

    def _shortest_path_for_agent(agent, agent_pos, agent_dir):
        # Flatland 4.x uses agent.state (TrainState enum)
        state = agent.state

        if agent_pos is None:
            if state == TrainState.WAITING:
                # Agent not yet on grid — start from initial position
                position  = agent.initial_position
                direction = agent.initial_direction
            elif state in _ACTIVE_STATES:
                position  = agent.position
                direction = agent.direction
            elif state == TrainState.DONE:
                position  = agent.target
                direction = agent.direction
            else:
                # MALFUNCTION_OFF_MAP or unknown — skip
                shortest_paths[agent.handle] = None
                return
        else:
            position  = agent_pos
            direction = agent_dir

        shortest_paths[agent.handle] = []
        distance = math.inf
        depth = 0

        while position != agent.target and (max_depth is None or depth < max_depth):
            next_actions = get_valid_move_actions_(direction, position, distance_map.rail)
            best_next_action = None

            for next_action in next_actions:
                next_action_distance = distance_map.get()[
                    agent.handle,
                    next_action.next_position[0],
                    next_action.next_position[1],
                    next_action.next_direction
                ]
                if next_action_distance < distance:
                    best_next_action = next_action
                    distance = next_action_distance

            shortest_paths[agent.handle].append(Waypoint(position, direction))
            depth += 1

            if best_next_action is None:
                shortest_paths[agent.handle] = None
                return

            position  = best_next_action.next_position
            direction = best_next_action.next_direction

        if max_depth is None or depth < max_depth:
            shortest_paths[agent.handle].append(Waypoint(position, direction))

    if agent_handle is not None:
        _shortest_path_for_agent(distance_map.agents[agent_handle], agent_pos, agent_dir)
    else:
        for agent in distance_map.agents:
            _shortest_path_for_agent(agent, agent_pos, agent_dir)

    return shortest_paths
