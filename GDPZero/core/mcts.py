import numpy as np
import logging
import math

from core.helpers import DialogSession
from core.game import DialogGame
from core.players import DialogPlanner


logger = logging.getLogger(__name__)


class MCTS():
	def __init__(self, game:DialogGame, player:DialogPlanner, configs) -> None:
		self.game = game
		self.player = player
		self.configs = configs
		# U(s,a) = Q(s,a) + c * P(s,a) * (\sqrt{ \sum_{a'} N(s,a')}) / (1+N(s,a))
		self.Ns: dict = {}  # saves compute
		self.Nsa: dict = {} # 노드-엑션 state 방문횟수 
		self.Q: dict = {} # 노드-액션에 대한 value function 저장
		self.P: dict = {} # 노드-액션에 대한 사전 확률 저장
		# utility
		self.valid_moves: dict = {}    ################### vaild_moves 어디에서 정의하는지?
		self.terminals: dict = {}
		# debugging / more information
		self.Vs: dict = {}
		return

	def _to_string_rep(self, state:DialogSession):
		# for tree search, keep all dialog turns
		return state.to_string_rep(keep_sys_da=True, keep_user_da=True, max_turn_to_display=-1)
	 	### histoy에서 이전turn을 제외하고 몇개의 history로 보여주는지 => keep all dialog turns (history)

	def _init_node(self, state:DialogSession):
		hashable_state = self._to_string_rep(state) # state의 모든 dialog turn을 저장  
		allowed_actions = self.player.get_valid_moves(state) # 1 if the i-th dialog act is valid, 0 otherwise [a1,a2,a3,...] = [1,1,0,...]
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0] 
		# nonzero(): 1인 원소의 인덱스를 반환하므로, 유효한 action의 인덱스를 저장


		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]} # hashable_state의 action 마다 Q값 

		prior, v = self.player.predict(state)  
		# 현재 state를 기준으로 각 dialog act 가 나온 횟수 저장 -> prob / # user의 응답score평균 
		self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.P[hashable_state] = prior * allowed_actions  ## 유효한 행동이면 1 아니면 0  *prior(각 유요한 action이 나올 확률 )
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])  # 유효한 p 이면 renormalize 
		return v

	###################### 다음 시스템 발화의 최적의 action을 정하기 위한 search  ####################
	def search(self, state:DialogSession):
		hashable_state = self._to_string_rep(state) # keep all dialog turns (history) : state에 있는 모든 sys utt 의 대화 (rold, da, utt)포함
		 
		is_leaf_node = False
		v = 0.0
		if hashable_state not in self.terminals:
			# selected leaf node, expand
			self.terminals[hashable_state] = self.game.get_dialog_ended(state)  # terminal 에서 성공여부를 왜 여기에서 지정해주는지?
			# returns 0 if not ended, then (in general) 1 if system success, -1 if failure -> 기부하면 성공. 안하면 실패 
			# terminate if there is a <donate> action in persudee resp
			v = self._init_node(state) # 터미널 node가 아니면 초기화 
			is_leaf_node = True
		# if this leaf node is terminal, return the value
		if self.terminals[hashable_state] > 0:
			# terminal node
			logger.debug("ended")
			return self.terminals[hashable_state]
		# otherwise, return v
		if is_leaf_node:
			return v
		
		# existing, continue selection
		# go next state by picking best according to U(s,a)
		best_uct = -float('inf')
		best_action = -1
		for a in self.valid_moves[hashable_state]:
			Ns = self.Ns[hashable_state]
			if Ns == 0:
				Ns = 1e-8
			uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (1 + self.Nsa[hashable_state][a])
			if uct > best_uct:
				best_uct = uct
				best_action = a
		# transition
		next_state = self.game.get_next_state(state, best_action)
		
		# 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
		# 2. if leaf, we will expand it and return the value for backpropagation
		v = self.search(next_state)

		# update stats
		# add in new estimate and average
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1
		
		# now we are single player, hence just v instead of -v
		return v

	def get_action_prob(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.Ns:
			# selected leaf node, expand
			logging.warn("querying a state that has not been visited")
			self._init_node(state) # expand 후 노드 초기화 
		# get the counts for all moves
		# convert to prob
		prob = np.zeros(self.player.get_valid_moves(state).shape) # 각 state 0 
		for a in self.valid_moves[hashable_state]:
			prob[a] = self.Nsa[hashable_state][a]   
		prob /= prob.sum()
		return prob


class OpenLoopMCTS(MCTS):
	def __init__(self, game, player, configs) -> None:
		super().__init__(game, player, configs)
		self.realizations: dict = {}  # state -> list of real DialogSessions
		self.realizations_Vs: dict = {}  # state -> {realization: V(realization)}
		self.realizations_Ns: dict = {}  # state -> {realization: N(realization)}
		self.max_realizations = configs.max_realizations
		return

	def _to_string_rep(self, state:DialogSession):
		# for tree search, keep all dialog turns
		das = []
		for (speaker, da, _) in state:
			if speaker == state.SYS:
				das.append(da)
		return "__".join(das) 

	def _init_node(self, state:DialogSession):
		hashable_state = self._to_string_rep(state) ## for tree search, keep all dialog turns
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}
		self.realizations[hashable_state] = [state.copy()]

		prior, v = self.player.predict(state)
		self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.P[hashable_state] = prior * allowed_actions
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])
		return v

	def _sample_realization(self, hashable_state):
		rand_i = np.random.randint(len(self.realizations[hashable_state]))
		return self.realizations[hashable_state][rand_i]

	def _add_new_realizations(self, state):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.realizations:
			self.realizations[hashable_state] = []
		if state in self.realizations[hashable_state]:
			return
		
		self.realizations[hashable_state].append(state.copy())
		if len(self.realizations[hashable_state]) > self.max_realizations:
			# should never happen
			logger.warning(f"len(self.realizations[hashable_state])={len(self.realizations[hashable_state])}")
			self.realizations[hashable_state].pop(0)
		return

	def _get_next_state(self, state, best_action):
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[best_action]
		if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
			# use the cached realization -> cache 로 모든 history를 저장하고 같은 의도와 의미 대화인 경우 버림 -> 메모리 효과적으로 사용 
			return self._sample_realization(prefetch_state) # 비슷한 맥락의 실제 발화로 부터 랜덤하게 선택 
		
		# otherwise, generate a new realization
		next_state = self.game.get_next_state(state, best_action) # 새로운 발화문을 만들어서 next state를 만듦 
		return next_state
	
	def _update_realizations_Vs(self, state: DialogSession, v: float):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.realizations_Vs:
			self.realizations_Vs[hashable_state] = {}
			self.realizations_Ns[hashable_state] = {}
		sys_utt = state.get_turn_utt(
			turn=-1,
			role=state.SYS,
		)
		if sys_utt not in self.realizations_Vs[hashable_state]:
			self.realizations_Vs[hashable_state][sys_utt] = 0
			self.realizations_Ns[hashable_state][sys_utt] = 0
		# update
		self.realizations_Ns[hashable_state][sys_utt] += 1
		self.realizations_Vs[hashable_state][sys_utt] += (v - self.realizations_Vs[hashable_state][sys_utt]) / self.realizations_Ns[hashable_state][sys_utt]
		return
	
	###################### 다음 시스템 발화의 최적의 action을 정하기 위한 search  ####################
	def search(self, state:DialogSession): 
		hashable_state = self._to_string_rep(state) ## keep all dialog turns (history) : state에 있는 모든 sys utt 의 대화 (rold, da, utt)포함
		
		# check everytime since state is stochastic, does not map to hashable_state
		terminated_v = self.game.get_dialog_ended(state) 
		## returns 0 if not ended, then (in general) 1 if system success, -1 if failure -> 기부하면 성공. 안하면 실패 
		# terminate if there is a <donate> action in persudee resp

		# check if it is terminal node
		if terminated_v == 1.0:
			logger.debug("ended")
			return terminated_v
		
		# otherwise, if is nontermial leaf node, we initialize and return v
		if hashable_state not in self.P:     # P에 각 hashable state -> 각 유효한 action마다 확률을 저장?  ============> self.P가 무엇을 의미하는지 잘 모르겠다. 
			# selected leaf node, expand it
			# first visit V because v is only evaluated once for a hashable_state
			v = self._init_node(state) # leaf node인 경우 child node를 초기화 
			return v 
		else:
			# add only when it is new
			self._add_new_realizations(state) # ternimal state인 경우 새로운 대화 쌍을 추가해줌 
		
		# existing, continue selection
		# go next state by picking best according to U(s,a)
		best_uct = -float('inf')
		best_action = -1
		for a in self.valid_moves[hashable_state]: # 해당 hashable_state -> a : 유효한 move 하나 
			Ns = self.Ns[hashable_state] # 해당 hashable state의 전체 방문 
			if Ns == 0:
				Ns = 1e-8
			# a variant of PUCT
				# Q : 초기에는 arg default값인 0으로 모두 할당
				# cpuct: 탐색을 조절하는 파라미터값 
				# self.P :  각 유효한 action이 나올 확률?? 
				# NS :  해당 hashable state의 전체 방문 횟수
				# Nsa : 해당 hashable state에 존재하는 action 중 현재 action
			# U(s,a) = Q(s,a) + c * P(s,a) * (\sqrt{ \sum_{a'} N(s,a')}) / (1+N(s,a))
			uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (1 + self.Nsa[hashable_state][a])
			if uct > best_uct:
				best_uct = uct
				best_action = a
		# transition. For open loop, first sample from an existing realization
		state = self._sample_realization(hashable_state)     # hashable_state 에서 sample_realization -> 수많은 history set 중에 랜럼하게 선택  -> state 
		next_state = self._get_next_state(state, best_action)   #  best action을 기준으로 대화쌍 기존거 이용 or 새로운 맥락이면 새로운 발화
		
		# 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
		# 2. if leaf, we will expand it and return the value for backpropagation
		v = self.search(next_state)    #############################################

		# update stats
		# add in new estimate and average
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1

		# update v to realizations for NLG at inference
		self._update_realizations_Vs(next_state, v)    # v값 구하는 식
		# now we are single player, hence just v instead of -v
		return v ############################################################## V 가 정확하게 무엇인지 ? 
	
	def get_best_realization(self, state:DialogSession, action: int):
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[action]
		if prefetch_state not in self.realizations_Vs:
			raise Exception("querying a state that has no realizations sampled before")
		# get the counts for all moves
		# convert to prob
		curr_best_v = -float('inf')
		curr_best_realization = None
		for sys_utt, v in self.realizations_Vs[prefetch_state].items():
			if v > curr_best_v:
				curr_best_v = v
				curr_best_realization = sys_utt
		return curr_best_realization
	

class OpenLoopMCTSParallel(OpenLoopMCTS):
	def __init__(self, game, player, configs) -> None:
		super().__init__(game, player, configs)

	def _populate_next_realizations(self, state, next_action, num_to_add):
		next_states = self.game.get_next_state_batched(state, next_action, batch=num_to_add)
		for next_state in next_states:
			self._add_new_realizations(next_state)
		return

	def _get_next_state(self, state, best_action):
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[best_action]
		if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
			# use the cached realization
			return self._sample_realization(prefetch_state)

		self._populate_next_realizations(state, best_action, self.max_realizations)
		return self._sample_realization(prefetch_state)
	
	def _init_node(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}
		# should have been initialized during _get_next_state, except for the root node
		if hashable_state not in self.realizations:
			self.realizations[hashable_state] = [state.copy()]

		# TODO: batch predict value function
		prior, v = self.player.predict(state)
		self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.P[hashable_state] = prior * allowed_actions
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])
		return v