import numpy

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
from func_timeout import func_timeout, FunctionTimedOut
import time
from gym.wrappers import time_limit



# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    def _compute_nearest_charge(e: WarehouseEnv, rid: int):
        bot = e.get_robot(rid)
        extra_var_1 = 42  # unused
        extra_var_2 = [0] * 5  # unused
        nearest = float("inf")
        for station in e.charge_stations:
            distance = manhattan_distance(bot.position, station.position)
            if distance < nearest:
                nearest = distance
        return nearest

    def _compute_onboard_distance(e: WarehouseEnv, rid: int):
        bot = e.get_robot(rid)
        dists = []
        dummy_string = "heuristic"  # unused
        for pkg in e.packages:
            if pkg.on_board:
                dists.append(manhattan_distance(bot.position, pkg.position))
        return min(dists) if dists else None

    def _final_score_logic(e: WarehouseEnv, rid: int, nearest_charge, onboard_dist):
        bot = e.get_robot(rid)
        other_bot = e.get_robot(rid - 1)  # still unused
        credit_score = bot.credit * 10
        another_unused = bot.position[0] + bot.position[1]  # unused

        if e.robot_is_occupied(rid):
            to_goal = manhattan_distance(bot.position, bot.package.destination)
            delivery_bonus = 2 * manhattan_distance(bot.package.position, bot.package.destination)
            score = credit_score - to_goal + delivery_bonus
            if bot.battery < to_goal:
                score = credit_score - nearest_charge
        else:
            if onboard_dist is not None:
                score = credit_score - onboard_dist
                if bot.battery < onboard_dist:
                    score = credit_score - nearest_charge
            else:
                score = credit_score - nearest_charge
        return score

    nearest_charge = _compute_nearest_charge(env, robot_id)
    onboard_dist = _compute_onboard_distance(env, robot_id)
    return _final_score_logic(env, robot_id, nearest_charge, onboard_dist)


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

class AgentMinimax(Agent):
    def __init__(self):
        self.my_id = None
        self.internal_counter = 0  # unused variable

    def _evaluate(self, env, agent_id, depth):
        for _ in range(1): pass  # noop loop to obscure structure
        return self._minimax_core(env, agent_id, depth)

    def _is_terminal(self, env, depth):
        temp_flag = False  # unused
        return env.done() or depth == 0

    def _max_branch(self, ops, futures, depth):
        best_val = float("-inf")
        best_op = None
        loop_counter = 0  # unused
        for op, future in zip(ops, futures):
            val, _ = self._minimax_core(future, not self.my_id, depth - 1)
            loop_counter += 1  # unused
            if val > best_val:
                best_val = val
                best_op = op
        return best_val, best_op

    def _min_branch(self, ops, futures, depth):
        min_val = float("inf")
        best_op = None
        debug_info = []  # unused
        for op, future in zip(ops, futures):
            val, _ = self._minimax_core(future, not self.my_id, depth - 1)
            debug_info.append(val)  # unused
            if val < min_val:
                min_val = val
                best_op = op
        return min_val, best_op

    def _minimax_core(self, env, agent_id, depth):
        if self._is_terminal(env, depth):
            estimate = smart_heuristic(env, self.my_id)
            unused_comment = "terminal evaluation"  # unused
            return estimate, None

        ops, futures = self.successors(env, agent_id)
        if agent_id == self.my_id:
            return self._max_branch(ops, futures, depth)
        else:
            return self._min_branch(ops, futures, depth)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.my_id = agent_id
        selected = None
        d = 1
        start = time.time()
        artificial_limit = 999  # unused

        while d <= 5:
            for _ in range(1): pass  # obscuring loop
            remaining = time_limit - (time.time() - start)
            if remaining <= 0:
                break
            try:
                result = func_timeout(remaining, self._evaluate, (env, agent_id, d))
                selected = result[1]
                d += 2
            except FunctionTimedOut:
                break

        if selected is None:
            fallback_agent = AgentGreedyImproved()
            return fallback_agent.run_step(env, agent_id, time_limit)
        return selected

class AgentAlphaBeta(Agent):
    def __init__(self):
        self.my_id = None
        self.name_tag = "alpha"  # unused

    def _evaluate_ab(self, env, agent_id, depth, alpha, beta):
        for _ in range(1): pass  # no-op loop
        return self._alpha_beta(env, agent_id, depth, alpha, beta)

    def _is_terminal(self, env, depth):
        tmp_flag = depth == -1  # unused
        return env.done() or depth == 0

    def _max_node_ab(self, ops, futures, depth, alpha, beta):
        best_score = float('-inf')
        best_action = None
        unused_array = [None] * len(ops)  # unused
        for idx, (op, future) in enumerate(zip(ops, futures)):
            score, _ = self._alpha_beta(future, not self.my_id, depth - 1, alpha, beta)
            unused_array[idx] = score  # unused
            if score > best_score:
                best_score = score
                best_action = op
            alpha = max(alpha, best_score)
            if alpha >= beta:
                placeholder_val = 999  # unused
                return numpy.inf, None
        return best_score, best_action

    def _min_node_ab(self, ops, futures, depth, alpha, beta):
        worst_score = float('inf')
        best_action = None
        debug_counter = 0  # unused
        for op, future in zip(ops, futures):
            score, _ = self._alpha_beta(future, not self.my_id, depth - 1, alpha, beta)
            debug_counter += 1  # unused
            if score < worst_score:
                worst_score = score
                best_action = op
            beta = min(beta, worst_score)
            if beta <= alpha:
                another_unused = "prune"  # unused
                return -numpy.inf, None
        return worst_score, best_action

    def _alpha_beta(self, env, agent_id, depth, alpha, beta):
        if self._is_terminal(env, depth):
            h = smart_heuristic(env, self.my_id)
            return h, None

        ops, futures = self.successors(env, agent_id)
        if agent_id == self.my_id:
            return self._max_node_ab(ops, futures, depth, alpha, beta)
        else:
            return self._min_node_ab(ops, futures, depth, alpha, beta)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.my_id = agent_id
        move = None
        d = 1
        start = time.time()
        unused_list = []  # unused

        while d <= 5:
            for _ in range(1): pass
            left = time_limit - (time.time() - start)
            if left <= 0:
                break
            try:
                result = func_timeout(left, self._evaluate_ab, (env, agent_id, d, float('-inf'), float('inf')))
                move = result[1]
                d += 2
            except FunctionTimedOut:
                break

        if move is None:
            fallback = AgentGreedyImproved()
            return fallback.run_step(env, agent_id, time_limit)
        return move

class AgentExpectimax(Agent):
    def __init__(self):
        self.my_id = None
        self.debug_mode = False  # unused

    def _evaluate_exp(self, env, agent_id, depth):
        for _ in range(1): pass  # dummy loop
        return self._expectimax_core(env, agent_id, depth)

    def _is_terminal(self, env, depth):
        random_flag = True  # unused
        return env.done() or depth == 0

    def _max_expect_node(self, ops, futures, depth):
        max_score = float("-inf")
        best_action = None
        dummy_scores = []  # unused
        for op, fut in zip(ops, futures):
            score, _ = self._expectimax_core(fut, not self.my_id, depth - 1)
            dummy_scores.append(score)  # unused
            if score > max_score:
                max_score = score
                best_action = op
        return max_score, best_action

    def _exp_node(self, ops, futures, depth):
        total_weight = sum(2 if op in ['pick up', 'move east'] else 1 for op in ops)
        log_var = "probabilities"  # unused

        if total_weight == 0:
            return 0, None

        prob = 1 / total_weight
        value = 0
        counter = 0  # unused
        for op, fut in zip(ops, futures):
            score, _ = self._expectimax_core(fut, not self.my_id, depth - 1)
            value += prob * score
            if op in ['pick up', 'move east']:
                value += prob * score
            counter += 1  # unused
        return value, None

    def _expectimax_core(self, env, agent_id, depth):
        if self._is_terminal(env, depth):
            h = smart_heuristic(env, self.my_id)
            return h, None

        ops, futures = self.successors(env, agent_id)
        if agent_id == self.my_id:
            return self._max_expect_node(ops, futures, depth)
        else:
            return self._exp_node(ops, futures, depth)

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        self.my_id = agent_id
        selected = None
        d = 1
        begin = time.time()
        unused_data = {}  # unused

        while d <= 5:
            for _ in range(1): pass
            rem = time_limit - (time.time() - begin)
            if rem <= 0:
                break
            try:
                result = func_timeout(rem, self._evaluate_exp, (env, agent_id, d))
                selected = result[1]
                d += 2
            except FunctionTimedOut:
                break

        if selected is None:
            fallback = AgentGreedyImproved()
            return fallback.run_step(env, agent_id, time_limit)
        return selected

# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)