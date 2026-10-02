from flatland.evaluators.client import FlatlandRemoteClient
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import EnvAgent, RailAgentStatus
import numpy as np
import time
import heapq
import copy
from queue import Queue

EPS = 0.0001
INFINITY = 1000000007
SAFE_LAYER = 4
START_TIME_LIMIT = 60
REPLAN_LIMIT = 250
MAX_TIME_ONE = 25

def getStepsToExitCell(v):
    return int(1 / v + EPS)

class Agent:
    def __init__(self, agentId, env):
        self.start_i = -1
        self.start_j = -1
        self.fin_i = -1
        self.fin_j = -1
        self.current_pos = 0
        self.actions = []
        self.obligations = None
        self.agentId = agentId
        self.spawned = False
        self.malfunctioning = False

    def getAgent(self, env):
        if env.agents[self.agentId].position is None:
            self.start_i = env.agents[self.agentId].initial_position[0]
            self.start_j = env.agents[self.agentId].initial_position[1]
        else:
            self.start_i = env.agents[self.agentId].position[0]
            self.start_j = env.agents[self.agentId].position[1]

        self.fin_i = env.agents[self.agentId].target[0]
        self.fin_j = env.agents[self.agentId].target[1]
        self.dir = env.agents[self.agentId].direction
        self.stepsToExitCell = getStepsToExitCell(env.agents[self.agentId].speed_data["speed"])

class Agents:
    def __init__(self):
        self.allAgents = []
        self.size = 0

    def getAgents(self, env):
        self.size = env.get_num_agents()
        if not self.allAgents:
            for ind in range(self.size):
                self.allAgents.append(Agent(ind, env))
                self.allAgents[ind].getAgent(env)
        else:
            for ind in range(self.size):
                self.allAgents[ind].getAgent(env)

    def reset_agent(self, number, new_i, new_j):
        self.allAgents[number].actions = []
        self.allAgents[number].current_pos = 0
        self.allAgents[number].start_i = new_i
        self.allAgents[number].start_j = new_j

class Node:
    def __init__(self, i, j, dir):
        self.i = i
        self.j = j
        self.t = 0
        self.f = 0
        self.g = 0
        self.h = 0
        self.dir = dir
        self.spawned = False
        self.parent = None

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j and self.t == other.t and self.dir == other.dir and self.spawned == other.spawned

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.f == other.f:
            if self.g == other.g:
                if self.spawned == other.spawned:
                    if self.i == other.i:
                        if self.j == other.j:
                            return self.dir < other.dir
                        return self.j < other.j
                    return self.i < other.i
                return self.spawned < other.spawned
            return self.g > other.g
        return self.f < other.f

    def __hash__(self):
        return hash((self.i, self.j, self.t, self.dir, self.spawned))

class Node_h:
    def __init__(self, x, y, dir, time):
        self.i = x
        self.j = y
        self.dir = dir
        self.time = time

class Global_H:
    def __init__(self, env):
        self.database = dict()
        self.env = env
        for ind in range(env.get_num_agents()):
            self.start_agent(ind)

    def get_neighbors(self, curNode):
        available = self.env.rail.get_transitions(*[curNode.i, curNode.j], curNode.dir)
        answer = []
        if available[0]: answer.append(Node_h(curNode.i - 1, curNode.j, 0, curNode.time + 1))
        if available[1]: answer.append(Node_h(curNode.i, curNode.j + 1, 1, curNode.time + 1))
        if available[2]: answer.append(Node_h(curNode.i + 1, curNode.j, 2, curNode.time + 1))
        if available[3]: answer.append(Node_h(curNode.i, curNode.j - 1, 3, curNode.time + 1))
        return answer

    def get_dir(self, position):
        answer = []
        for dest in range(4):
            available = self.env.rail.get_transitions(*position, dest)
            if sum(available) > 0:
                answer.append(dest)
        return answer

    def start_agent(self, number):
        correct_dir = self.get_dir(self.env.agents[number].target)
        queue = Queue()
        for dest in correct_dir:
            start = Node_h(self.env.agents[number].target[0], self.env.agents[number].target[1], dest, 0)
            queue.put(start)
        while not queue.empty():
            current = queue.get()
            candidates = self.get_neighbors(current)
            for node in candidates:
                if (number, current.i, current.j, (node.dir + 2) % 4) not in self.database:
                    self.database[(number, current.i, current.j, (node.dir + 2) % 4)] = current.time
                    queue.put(node)

    def get_heuristic(self, agentId, x, y, dir):
        if (agentId, x, y, dir) in self.database:
            return self.database[(agentId, x, y, dir)]
        else:
            return INFINITY

class ISearch:
    def __init__(self, env, heuristic):
        self.lppath = [[] for _ in range(env.get_num_agents())]
        self.reservations = dict()
        self.maxTime = 5000
        self.additional_reserve = 6
        self.heuristic = heuristic

    def checkReservation(self, i, j, t):
        return (t, i, j) in self.reservations
    def get_occupator(self, i, j, t):
        return self.reservations.get((t, i, j))

    def startallAgents(self, env, control_agent, order, time_limit, current_step):
        path_exists = [False] * env.get_num_agents()
        start_time = time.time()
        for i in range(len(order)):
            agent = control_agent.allAgents[order[i]]
            if agent.spawned:
                path_exists[agent.agentId] = True
                continue
            path_exists[agent.agentId] = self.startSearch(agent, env, current_step)
            if time.time() - start_time > time_limit:
                break
        return path_exists

    def startSearch(self, agent, env, current_step):
        startNode = agent.obligations
        finNode = Node(agent.fin_i, agent.fin_j, agent.dir)
        openHeap = []
        openCopy = set()
        pathFound = False
        heapq.heappush(openHeap, startNode)
        start_search_time = time.time()
        while not pathFound and openHeap:
            curNode = heapq.heappop(openHeap)
            if curNode.t >= self.maxTime or curNode.h == INFINITY or time.time() - start_search_time >= MAX_TIME_ONE:
                break
            if curNode.i == finNode.i and curNode.j == finNode.j:
                finNode = curNode
                pathFound = True
                break
            successors = self.findSuccessors(curNode, agent, env, self.heuristic)
            for scNode in successors:
                if scNode not in openCopy:
                    scNode.parent = curNode
                    heapq.heappush(openHeap, scNode)
                    openCopy.add(scNode)
        if pathFound:
            self.makePrimaryPath(finNode, startNode, agent)
            self.makeFlatlandFriendlyPath(agent)
            return True
        return False

    def findSuccessors(self, curNode, agent, env, heuristic):
        position = [curNode.i, curNode.j]
        available = env.rail.get_transitions(*position, curNode.dir)
        inter_answer = []
        if available[0]: inter_answer.append(Node(curNode.i - 1, curNode.j, 0))
        if available[1]: inter_answer.append(Node(curNode.i, curNode.j + 1, 1))
        if available[2]: inter_answer.append(Node(curNode.i + 1, curNode.j, 2))
        if available[3]: inter_answer.append(Node(curNode.i, curNode.j - 1, 3))
        inter_answer.append(Node(curNode.i, curNode.j, curNode.dir))
        successors = []
        if not curNode.spawned:
            not_spawned = Node(curNode.i, curNode.j, curNode.dir)
            not_spawned.g = curNode.g + 1
            not_spawned.t = curNode.t + 1
            not_spawned.h = curNode.h
            not_spawned.f = curNode.f + 1
            not_spawned.spawned = False
            successors.append(not_spawned)
            spawned_on_this_turn = copy.deepcopy(not_spawned)
            spawned_on_this_turn.spawned = True
            spawned_on_this_turn.h -= 1
            spawned_on_this_turn.f -= 1
            if self.correct_point(spawned_on_this_turn, agent):
                successors.append(spawned_on_this_turn)
            return successors
        for scNode in inter_answer:
            scNode.h = heuristic.get_heuristic(agent.agentId, scNode.i, scNode.j, scNode.dir) * agent.stepsToExitCell
            scNode.spawned = True
            if scNode.i == curNode.i and scNode.j == curNode.j:
                scNode.t = curNode.t + 1
                scNode.g = curNode.g + 1
            else:
                scNode.t = curNode.t + agent.stepsToExitCell
                scNode.g = curNode.g + agent.stepsToExitCell
            scNode.f = scNode.g + scNode.h
            if not self.correct_point(scNode, agent): continue
            successors.append(scNode)
        return successors

    def correct_point(self, scNode, agent):
        for step in range(agent.stepsToExitCell):
            occ = self.get_occupator(scNode.i, scNode.j, scNode.t + step)
            if occ is not None and occ != agent.agentId: return False
        other = self.get_occupator(scNode.i, scNode.j, scNode.t + agent.stepsToExitCell)
        if other is not None and other < agent.agentId: return False
        otherp = self.get_occupator(scNode.i, scNode.j, scNode.t - 1)
        if otherp is not None and otherp > agent.agentId: return False
        return True

    def makePrimaryPath(self, curNode, startNode, agent):
        wait_action = False
        while curNode != startNode:
            self.lppath[agent.agentId].append(curNode)
            if not wait_action:
                for step in range(agent.stepsToExitCell):
                    self.reservations[(curNode.t + step, curNode.i, curNode.j)] = agent.agentId
            elif curNode.spawned:
                self.reservations[(curNode.t, curNode.i, curNode.j)] = agent.agentId
            if curNode.i == curNode.parent.i and curNode.j == curNode.parent.j:
                wait_action = True
            else:
                wait_action = False
            curNode = curNode.parent
        self.lppath[agent.agentId].append(curNode)
        self.lppath[agent.agentId] = self.lppath[agent.agentId][::-1]

    def makeFlatlandFriendlyPath(self, agent):
        for ind in range(1, len(self.lppath[agent.agentId])):
            curr = self.lppath[agent.agentId][ind]
            prev = self.lppath[agent.agentId][ind-1]
            if curr.i == prev.i and curr.j == prev.j:
                if not prev.spawned and curr.spawned: agent.actions.append(2)
                else: agent.actions.append(4)
            elif abs(curr.dir - prev.dir) % 2 == 0:
                for _ in range(agent.stepsToExitCell): agent.actions.append(2)
            elif (prev.dir + 1) % 4 == curr.dir:
                for _ in range(agent.stepsToExitCell): agent.actions.append(3)
            else:
                for _ in range(agent.stepsToExitCell): agent.actions.append(1)

    def delete_all(self, number):
        to_delete = [cell for cell, val in self.reservations.items() if val == number]
        for cell in to_delete: del self.reservations[cell]

    def need_enter(self, scNode, agent, first_agent):
        occ = self.get_occupator(scNode.i, scNode.j, scNode.t)
        if occ is not None and occ != agent.agentId and (occ < agent.agentId or occ == first_agent):
            return False
        occp = self.get_occupator(scNode.i, scNode.j, scNode.t - 1)
        if occp is not None and occp > agent.agentId: return False
        return True

    def replan_agent(self, agent, env, current_step, calculated, start_replanning_time, second_queue):
        self.delete_all(agent.agentId)
        self.lppath[agent.agentId] = []
        if not agent.spawned:
            for step in range(current_step, agent.obligations.t): agent.actions.append(4)
            second_queue.append(agent.agentId)
            return []
        passers_by = []
        for step in range(current_step, agent.obligations.t):
            occ = self.get_occupator(agent.start_i, agent.start_j, step)
            if occ is not None and occ != agent.agentId:
                passers_by.append(occ)
                self.delete_all(occ)
            self.reservations[(step, agent.start_i, agent.start_j)] = agent.agentId
            agent.actions.append(4)
        path_exists = self.startSearch(agent, env, current_step)
        return passers_by

class Solver:
    def __init__(self, env, heuristic):
        self.env = env
        self.control_agent = Agents()
        self.control_agent.getAgents(env)
        self.answer_build = False
        self.search = ISearch(env, heuristic)
        self.current_step = 0
        self.prev_action = [2] * self.env.get_num_agents()
        self.heuristic = heuristic

    def make_obligation(self, number):
        agent = self.control_agent.allAgents[number]
        start_i, start_j = (agent.start_i, agent.start_j)
        agent.obligations = Node(start_i, start_j, agent.dir)
        agent.obligations.h = agent.stepsToExitCell * self.heuristic.get_heuristic(agent.agentId, start_i, start_j, agent.dir) + (not agent.spawned)
        agent.obligations.spawned = agent.spawned
        agent.obligations.t = self.current_step + self.env.agents[number].malfunction_data['malfunction']

    def build_on_the_start(self):
        for ind in range(self.env.get_num_agents()): self.make_obligation(ind)
        self.search.startallAgents(self.env, self.control_agent, list(range(self.env.get_num_agents())), START_TIME_LIMIT, self.current_step)
        self.answer_build = True

def my_controller(env, path_finder):
    for ind in range(path_finder.control_agent.size):
        if path_finder.env.agents[ind].position is not None:
            path_finder.control_agent.allAgents[ind].spawned = True
    if not path_finder.answer_build:
        path_finder.build_on_the_start()
    # Simplified controller for expert guide
    actions = {}
    for ind in range(env.get_num_agents()):
        agent = path_finder.control_agent.allAgents[ind]
        if agent.current_pos < len(agent.actions):
            actions[ind] = agent.actions[agent.current_pos]
            agent.current_pos += 1
        else:
            actions[ind] = 4
    path_finder.current_step += 1
    return actions
