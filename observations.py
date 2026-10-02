import numpy as np
import copy
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.distance_map import DistanceMap
from shortestpath import get_shortest_paths

class StateMaskingObs(ObservationBuilder):
    def __init__(self):
        super(StateMaskingObs, self).__init__()
        self.obs_size = 18

    def reset(self):
        return

    def _get_shortest_path(self, agent_handle):
        # Uses the distance map to find the best path for an agent skipping collisions
        distance_map = self.env.distance_map
        path = get_shortest_paths(distance_map, agent_handle=agent_handle)
        return path[agent_handle]

    def get(self, handle=0):
        # Initial status check
        if self.env.dones[handle]:
            return [0.0, -1.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, -1.0, 0.0, 0.0, 0.0, 0.0]

        agent = self.env.agents[handle]
        my_pos = agent.position if agent.position is not None else agent.initial_position
        my_direction = agent.direction if agent.direction is not None else agent.initial_direction
        
        # Possible Move-Actions: Left, Forward, Right (1, 2, 3)
        valid_act_pos_pair = self._get_next_valid_position(my_pos, my_direction)
        
        # State classification
        is_stopping = self.StateClassifier(my_pos, my_direction) in [1, 4]
        
        obs = []
        for action in [1, 2, 3]:
            next_step = valid_act_pos_pair[action]
            if next_step[1] is not None:
                # Calculate features along the path for this action
                has_sol, features = self._count_path_block(next_step[1], next_step[0], handle)
                features.insert(0, 1.0) # Mask: Valid action available
                obs.append(features)
            else:
                obs.append([0.0, -1.0, 0.0, 0.0, 0.0, 0.0]) # Mask: Action invalid
        
        obs = np.array(obs, dtype=float)
        # Normalize distances
        max_dist = np.max(obs[:, 1])
        for i in range(3):
            if obs[i, 1] > 0:
                obs[i, 1] = obs[i, 1] / max_dist if max_dist > 0 else 0
            else:
                obs[i, 1] = -1.0
                
        return obs.flatten().tolist()

    def StateClassifier(self, pos, direction):
        """
        0: Normal, 1: Stopping Point, 2: Junction, 3: Must Stop, 4: Stopping+Must Stop
        """
        avb_moves = self.env.rail.get_transitions((pos, direction))
        if sum(avb_moves) > 1: return 2
        
        # Look ahead for junction
        avb_idx = avb_moves.index(1)
        # Simple map based on direction
        dir_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        next_pos = (pos[0] + dir_offsets[avb_idx][0], pos[1] + dir_offsets[avb_idx][1])
        
        is_next_junction = False
        for d in range(4):
            if sum(self.env.rail.get_transitions((tuple(next_pos), d))) > 2:
                is_next_junction = True
                break
        
        # Check occupancy
        occupied = any(a.position == next_pos for a in self.env.agents)
        
        if is_next_junction:
            return 4 if occupied else 1
        return 3 if occupied else 0

    def _get_next_valid_position(self, pos, direction):
        # Maps 1,2,3 actions to absolute grid movements
        # Simplified for robustness
        res = {1: [None, None], 2: [None, None], 3: [None, None]}
        transitions = self.env.rail.get_transitions((pos, direction))
        # Relative directions: -1 (left), 0 (forward), 1 (right)
        rel_dirs = [-1, 0, 1]
        for i, rd in enumerate(rel_dirs):
            new_dir = (direction + rd) % 4
            if transitions[new_dir]:
                off = [(-1, 0), (0, 1), (1, 0), (0, -1)][new_dir]
                res[i+1] = [new_dir, (pos[0] + off[0], pos[1] + off[1])]
        return res

    def _count_path_block(self, start_pos, heading, handle):
        # Uses shortest path to look ahead
        path = self._get_shortest_path(handle)
        if not path:
            return False, [-1, 0, 0, 0, 0]
        
        path_len = len(path)
        junction_blocks = 0
        total_blocks = 0
        dist_to_junction = path_len
        
        other_positions = {a.position: a.direction for i, a in enumerate(self.env.agents) 
                          if i != handle and a.position is not None}
        
        found_junction = False
        for i, step in enumerate(path):
            pos, dr = step.position, step.direction
            if not found_junction and self.StateClassifier(pos, dr) == 2:
                dist_to_junction = i
                found_junction = True
            
            if pos in other_positions:
                total_blocks += 1
                if self.StateClassifier(pos, dr) == 2:
                    junction_blocks += 1
                    
        return True, [float(path_len), float(total_blocks), float(junction_blocks), 
                     float(dist_to_junction), 0.0]
