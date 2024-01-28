import numpy as np
import logging
import pickle
import argparse
import numpy as np

from tqdm.auto import tqdm
from core.gen_models import (
	LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
from core.players import (
	PersuadeeModel, PersuaderModel, P4GSystemPlanner,
	PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.game import PersuasionGame
from core.mcts import OpenLoopMCTS
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
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
	
	# 1-3. llm에 따라 backbonemodel정의  system, user, planner 별로 chatmodel 불러와서 정의 
	if cmd_args.llm in ['code-davinci-002']:
		backbone_model = OpenAIModel(cmd_args.llm)
		SysModel = PersuaderModel
		UsrModel = PersuadeeModel
		SysPlanner = P4GSystemPlanner
	elif cmd_args.llm in ['gpt-3.5-turbo']:
		backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'chatgpt':
		backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	
	system = SysModel(
		sys_da,
		backbone_model, 
		conv_examples=[exp_1],
		inference_args={
			"temperature": 0.7,
			"do_sample": True,  # for MCTS open loop
			"return_full_text": False,
		}
	)
	user = UsrModel(
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
	planner = SysPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=[exp_1]
	)

	game = PersuasionGame(system, user)

	print(f"System dialog acts: {system.dialog_acts}")
	print(f"User dialog acts: {user.dialog_acts}")

	with open("data/p4g/300_dialog_turn_based.pkl", "rb") as f:
		all_dialogs = pickle.load(f)

	num_dialogs = cmd_args.num_dialogs # number of dialogs to test MCTS on
	args = dotdict({   # # mcts에 들어가는 arguments 집합 
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
	})

	output = []  # for evaluation. [{did, context, ori_resp, new_resp, debug}, ...]
	# those dialogs has inappropriated content and will throw an error/be filtered with OPENAI models. See raw_prompting.py file for more details
	bad_dialogs = ['20180808-024552_152_live', '20180723-100140_767_live', '20180825-080802_964_live']  # throws exception due to ChatGPT API filtering
	num_done = 0
	pbar = tqdm(total=num_dialogs, desc="evaluating")
	for did in all_dialogs.keys():  #### 2. 여기에서 부터 모든 대화 set에 대해서 시도 
		if did in bad_dialogs: 
			print("skipping dialog id: ", did)
			continue
		if num_done == num_dialogs:
			break

		print("evaluating dialog id: ", did)
		context = ""
		dialog = all_dialogs[did] ##### 모든 대화 set중에서 하나의 대화 set 
		
		state = game.init_dialog() # state 초기화 
		for t, turn in enumerate(dialog["dialog"]): # 라벨 되어 있지 않은 대화 set - turn 0 {'ee': utt, 'er': utt } pair
			if len(turn["ee"]) == 0:  # ended 
				break
			# also skip last turn as there is no evaluation
			if t == len(dialog["dialog"]) - 1:
				break

			# 데이터셋에서의 user 발화와 label을 현재 system에서의 dialog act로 맵핑
			usr_utt = " ".join(turn["ee"]).strip() 
			usr_da = dialog["label"][t]["ee"][-1]

			# map to our dialog act
			if usr_da == "disagree-donation":
				usr_da = PersuasionGame.U_NoDonation
			elif usr_da == "negative-reaction-to-donation":
				usr_da = PersuasionGame.U_NegativeReaction
			elif usr_da == "positive-reaction-to-donation":
				usr_da = PersuasionGame.U_PositiveReaction
			elif usr_da == "agree-donation":
				usr_da = PersuasionGame.U_Donate
			else:
				usr_da = PersuasionGame.U_Neutral

			# game ended
			if usr_da == PersuasionGame.U_Donate:
				break

			# 데이터셋에서의 user 발화와 label을 현재 system에서의 dialog act로 맵핑
			# map sys as well
			sys_utt = " ".join(turn["er"]).strip()
			sys_da = set(dialog["label"][t]["er"])
			intersected_das = sys_da.intersection(system.dialog_acts)
			if len(intersected_das) == 0:
				sys_da = "other"
			else:
				sys_da = list(intersected_das)[-1]
			
			# 하나의 turn(sys, usr)에 대해 state에 추가 
			state.add_single(PersuasionGame.SYS, sys_da, sys_utt) 
			state.add_single(PersuasionGame.USR, usr_da, usr_utt)

			# update context for evaluation => # 현재 state에서 가장 마지막 system utt와 user 발화를 context로 정의 
			context = f"""
			{context}
			Persuader: {sys_utt}
			Persuadee: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			
			# mcts policy
			if isinstance(backbone_model, OpenAIModel):
				backbone_model._cached_generate.cache_clear()
			## openloop mcts 객체 생성
			dialog_planner = OpenLoopMCTS(game, planner, args) 
			print("searching")
			for i in tqdm(range(args.num_MCTS_sims)):  # # args.num_MCTS_sims mcts 시뮬레이션 횟수만큼 search를 반복하여 Q값 등을 업데이트 
				dialog_planner.search(state) # 이때 search 할때 지금의 state이 활용됨 

			mcts_policy = dialog_planner.get_action_prob(state) # 해당 statd에서 action 유효한지여부를 체크하고 이를  prob으로 바꿈 
			mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)] # prob이 가장큰 dialog action을 선택 

			## fetch the generated utterance from simulation
			mcts_pred_rep = dialog_planner.get_best_realization(state, np.argmax(mcts_policy)) ## get_best_realizations ??? 

			# next ground truth utterance : 주어진 데이터셋에서의 다음 sys-usr 대화를 next ground truth utterance라고 지정 
			human_resp = " ".join(dialog["dialog"][t+1]["er"]).strip()
			next_sys_das = set(dialog["label"][t+1]["er"])
			next_intersected_das = next_sys_das.intersection(system.dialog_acts)
			if len(next_intersected_das) == 0:
				next_sys_da = "other"
			else:
				next_sys_da = list(next_intersected_das)[-1]

			# logging for debug
			debug_data = {
				"probs": mcts_policy,
				"da": mcts_policy_next_da,
				"search_tree": {
					"Ns": dialog_planner.Ns,
					"Nsa": dialog_planner.Nsa,
					"Q": dialog_planner.Q,
					"P": dialog_planner.P,
					"Vs": dialog_planner.Vs,
					"realizations": dialog_planner.realizations,
					"realizations_Vs": dialog_planner.realizations_Vs,
					"realizations_Ns": dialog_planner.realizations_Ns,
				},
			}

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': mcts_pred_rep,
				'new_da': mcts_policy_next_da,
				"debug": debug_data,
			}
			output.append(cmp_data)

			if cmd_args.debug:
				print(context)
				print("human resp: ", human_resp)
				print("human da: ", next_sys_da)
				print("mcts resp: ", mcts_pred_rep)
				print("mcts da: ", mcts_policy_next_da)
		with open(cmd_args.output, "wb") as f:
			pickle.dump(output, f)
		num_done += 1
		pbar.update(1)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--output', type=str, default="outputs/gdpzero.pkl", help='output file')
	parser.add_argument('--llm', type=str, default="code-davinci-002", choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo"], help='OpenAI model name')
	parser.add_argument('--gen_sentences', type=int, default=-1, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
	parser.add_argument('--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
	parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
	parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
	parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test MCTS on')
	parser.add_argument('--debug', action='store_true', help='debug mode')
	parser.parse_args()
	cmd_args = parser.parse_args()
	print("saving to", cmd_args.output)

	main(cmd_args)