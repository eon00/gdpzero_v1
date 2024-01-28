import numpy as np
import logging
import argparse

from tqdm.auto import tqdm
from core.gen_models import (
	LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
from core.players import (
	PersuadeeModel, PersuaderModel, P4GSystemPlanner,
	PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.game import PersuasionGame
from core.mcts import MCTS, OpenLoopMCTS, OpenLoopMCTSParallel
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)


def play_gdpzero(backbone_model, args):
	args = dotdict({  # mcts에 들어가는 arguments 집합 
		"cpuct": 1.0,
		"num_MCTS_sims": args.num_mcts_sims, 
		"max_realizations": args.max_realizations,
		"Q_0": args.Q_0,
	})
	## 1. setting 
	## 1-1. persuasiongame모듈에서 sys_da, user_da, system name, user_name 불러오기  
	game_ontology = PersuasionGame.get_game_ontology()
	sys_da = game_ontology['system']['dialog_acts']
	user_da = game_ontology['user']['dialog_acts']
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR

	# 1-2. initialize root node / history
	# EXP_DIALOG : [(role,da,utt), ... ]  : sys-user 번갈아 대화한 것 대화문 ex) [(sys, da, utt), (usr, da,utt), ... ]
	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)  

	# 1-3. system, user, planner 별로 chatmodel 불러와서 정의 
	system = PersuaderChatModel(
		sys_da,
		backbone_model, 
		conv_examples=[exp_1],  ## [[(role,da,utt), ... ]]
		inference_args={
			"temperature": 0.7,
			"do_sample": True,  # for MCTS open loop
			"return_full_text": False,
		}
	)
	user = PersuadeeChatModel(
		user_da,
		inference_args={
			"max_new_tokens": 128,
			"temperature": 1.1,
			"repetition_penalty": 1.0,
			"do_sample": True,  # for MCTS open loop
			"return_full_text": False,
		},
		backbone_model=backbone_model, 
		conv_examples=[exp_1]
	)
	planner = P4GChatSystemPlanner( 
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[exp_1] 
	)

	game = PersuasionGame(system, user) # persuasiongame <- sys_da, user_da, system name, user_name 정의되어 있음 
	state = game.init_dialog() #  init_dialog() : return DialogSession(class)

	# init
	state.add_single(game.SYS, 'greeting', "Hello. How are you?") # history 길이를 보고 (sys , da , utt)를 history에 append / 일단 초기 상태는 인사부터 
	print("You are now the Persuadee. Type 'q' to quit, and 'r' to restart.")
	print("Persuader: Hello. How are you?")

	your_utt = input("You: ")
	while your_utt.strip() != "q":
		if your_utt.strip() == "r":
			state = game.init_dialog()
			state.add_single(game.SYS, 'greeting', "Hello. How are you?")
			game.display(state)
			your_utt = input("You: ")
			continue
		
		# used for da prediction
		tmp_state = state.copy()
		tmp_state.add_single(game.USR, 'neutral', your_utt.strip())  ###################### (sys , da , utt)를 history에 append
		user_da = user.predict_da(tmp_state) ## 현재 user 발화를 추가한 state 에서 message -> 응답 생성하도록 : user의 반응 중 많이 나온 action 것 선택 

		logging.info(f"user_da: {user_da}")
		state.add_single(game.USR, user_da, your_utt.strip()) # 위에 tem_state를 만들어 예측한 user의 응답을 현재 발화의 da 취급 / user ,da, stt 를 state history로 추가 

		# planning
		if isinstance(backbone_model, OpenAIModel):
			backbone_model._cached_generate.cache_clear()
		dialog_planner = OpenLoopMCTS(game, planner, args) # openloop mcts 객체 생성
		for i in tqdm(range(args.num_MCTS_sims)): # args.num_MCTS_sims mcts 시뮬레이션 횟수만큼 search를 반복하여 Q값 등을 업데이트 
			dialog_planner.search(state)  # Q값 state 이런거 계속 업데이트

		mcts_policy = dialog_planner.get_action_prob(state)
		mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)] # 확률값을 최대로 만드는 인덱스???
		logger.info(f"mcts_policy: {mcts_policy}")
		logger.info(f"mcts_policy_next_da: {mcts_policy_next_da}")
		logger.info(dialog_planner.Q)

		sys_utt = dialog_planner.get_best_realization(state, np.argmax(mcts_policy)) #######################3 get_best_realization
		logging.info(f"sys_da: [{mcts_policy_next_da}]")
		print(f"Persuader: {sys_utt}")
		
		state.add_single(game.SYS, mcts_policy_next_da, sys_utt) # 
		your_utt = input("You: ")
	return


def play_raw_prompt(backbone_model):
	system_name = PersuasionGame.SYS
	user_name = PersuasionGame.USR
	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	game_ontology = PersuasionGame.get_game_ontology()
	sys_da = game_ontology['system']['dialog_acts']
	user_da = game_ontology['user']['dialog_acts']

	system = PersuaderChatModel(
		sys_da,
		backbone_model, 
		conv_examples=[exp_1]
	)
	user = PersuadeeChatModel(
		user_da,
		inference_args={
			"max_new_tokens": 128,
			"temperature": 1.1,
			"repetition_penalty": 1.0,
			"do_sample": True,
			"return_full_text": False,
		},
		backbone_model=backbone_model, 
		conv_examples=[exp_1]
	)
	planner = P4GChatSystemPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[exp_1]
	)
	game = PersuasionGame(system, user)
	state = game.init_dialog()

	# init
	state.add_single(game.SYS, 'greeting', "Hello. How are you?")
	print("You are now the Persuadee. Type 'q' to quit, and 'r' to restart.")
	print("Persuader: Hello. How are you?")

	your_utt = input("You: ")
	while your_utt.strip() != "q":
		if your_utt.strip() == "r":
			state = game.init_dialog()
			state.add_single(game.SYS, 'greeting', "Hello. How are you?")
			game.display(state)
			your_utt = input("You: ")
			continue
		# used for da prediction
		state.add_single(game.USR, 'neutral', your_utt.strip())

		# planning
		prior, v = planner.predict(state)
		greedy_policy = system.dialog_acts[np.argmax(prior)]
		next_best_state = game.get_next_state(state, np.argmax(prior))
		greedy_pred_resp = next_best_state.history[-2][2]
		
		logging.info(f"sys_da: [{greedy_policy}]")
		print(f"Persuader: {greedy_pred_resp}")
		
		state.add_single(game.SYS, greedy_policy, greedy_pred_resp)
		your_utt = input("You: ")
	return


def main(args):
	# argument에서 받은 llm에 따라 backbonemodel을 선정 (3가지 모델)  arg = {'llm': 'chatgpt', 'algo': 'gdpzero' }
	if args.llm in ['code-davinci-002', 'text-davinci-003']:
		backbone_model = OpenAIModel(args.llm)
	elif args.llm in ['gpt-3.5-turbo']:
		backbone_model = OpenAIChatModel(args.llm, args.gen_sentences)
	elif args.llm == 'chatgpt':
		backbone_model = AzureOpenAIChatModel(args.llm, args.gen_sentences)

	# argue에서 받은 algo값에 따라 gdpzero planning algorithm 사용 여부 결정
	if args.algo == 'gdpzero':
		print("using GDPZero as planning algorithm")
		play_gdpzero(backbone_model, args)
	elif args.algo == 'raw-prompt':
		print("using raw prompting as planning")
		play_raw_prompt(backbone_model)
	return


if __name__ == "__main__":
	# logging mode
	parser = argparse.ArgumentParser()
	parser.add_argument("--log", type=int, default=logging.WARNING, help="logging mode", choices=[logging.INFO, logging.DEBUG, logging.WARNING])
	parser.add_argument("--algo", type=str, default='gdpzero', choices=['gdpzero', 'raw-prompt'], help="planning algorithm")
	# used by PDP-Zero
	parser.add_argument('--llm', type=str, default="gpt-3.5-turbo", choices=["code-davinci-002", "gpt-3.5-turbo", "text-davinci-002", "chatgpt"], help='OpenAI model name')
	parser.add_argument('--gen_sentences', type=int, default=3, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.') # 문장 길이를 지정할수 있음 
	parser.add_argument('--num_mcts_sims', type=int, default=10, help='number of mcts simulations') # mcts 시뮬레이션 하는 횟수 
	parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state') # state 하나당 realizaiton이 몇번 되는지 (하나의 state에 몇개의 노드를 생성할 수 있을지)
	parser.add_argument('--Q_0', type=float, default=0.25, help='initial Q value for unitialized states. to control exploration') # puct : 
	args = parser.parse_args()
	logging.basicConfig(level=args.log)
	logger.setLevel(args.log)

	main(args)