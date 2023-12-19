import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class Node:
    def __init__(self, parent, prior_p, state=None):
        self._parent = parent
        self._children = []
        self._n_visits = 0
        self._Q = 0
        self._prior_p = prior_p
        self._state = state

    def expand(self, action_priors):
        self._children = [Node(self, p) for p in action_priors]

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return self._children == []

    def update_recursive(self, value):
        self._n_visits += 1
        self._Q += ((self._n_visits - 1) * self._Q + value) / self._n_visits
        if not self.is_root():
            self._parent.update_recursive(-value)


class MCTS:
    def __init__(self, game, nnet, args):
        self._game = game
        self._nnet = nnet
        self._args = args

    def select_action(self, node, mask):
        assert not node.is_leaf()
        scores = [self.score_fn(child) * m for child, m in zip(node._children, mask)]
        return np.argmax(scores)

    def get_action_probs(self, node, temp=1):
        assert not node.is_leaf()
        counts = [child._n_visits ** (1.0 / temp) for child in node._children]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def score_fn(self, node):
        return node._Q + self._args.cpuct * node._prior_p * math.sqrt(node._parent._n_visits) / (1 + node._n_visits)

    def search_til_leaf(self, node):
        ended = self._game.getGameEnded(node._state, 1)
        if ended != 0:
            node._parent.update_recursive(-ended)
            return

        while not node.is_leaf():
            state = node._state
            action = self.select_action(node, self._game.getValidMoves(state, 1))
            n_state, n_player = self._game.get_next_state(node._state, 1, action)
            n_state = self._game.get_canonical_form(n_state, n_player)

            node = node._children[action]
            node._state = n_state
            self.search_til_leaf(node)

        if node.is_leaf():
            action_probs, value = self._nnet.predict(node._state)
            node.expand(action_probs)
            node._parent.update_recursive(-value)
            return
    
    def search(self, canonicalBoard):
        root = Node(None, 1.0)
        root._state = canonicalBoard
        self.search_til_leaf(root)
        return root


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
