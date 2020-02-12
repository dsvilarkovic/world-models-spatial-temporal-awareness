#python 04_train_rnn.py --new_model --batch_size 200
# python 04_train_rnn.py --new_model --batch_size 100

from rnn.arch import RNN
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ROOT_DIR_NAME = './data/'
SERIES_DIR_NAME = './data/series_doom/'


def get_filelist(S, N):
    filelist = os.listdir(SERIES_DIR_NAME)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)

    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    return filelist[S:], N


def get_batch(filelist, step):
	N_data = len(filelist)

	z_listS = []
	z_listB = []
	action_list = []
	rew_list = []

	try:
		new_data = np.load(SERIES_DIR_NAME + filelist[step])

		muS = new_data['muS']
		muB = new_data['muB']
		log_varS = new_data['log_varS']
		log_varB = new_data['log_varB']
		action = new_data['action']
		reward = new_data['reward']
    
		reward = np.expand_dims(reward, axis=2)
		action = np.hstack([action, np.zeros((action.shape[0],1))])
		s = log_varS.shape

		zS = muS + np.exp(log_varS/2.0) * np.random.randn(*s)
		zB = muB + np.exp(log_varB/2.0) * np.random.randn(*s)

		z_listS.append(zS[:min(len(muS), len(action))])
		z_listB.append(zB[:min(len(muS), len(action))])
		action_list.append(action[:min(len(muS), len(action))])
		rew_list.append(reward[:min(len(muS), len(action))])
	except:
		pass

	z_listS = np.array(z_listS)
	z_listB = np.array(z_listB)
	action_list = np.array(action_list)
	rew_list = np.array(rew_list)
	
	return z_listS, z_listB, action_list, rew_list

def main(args):
	
	new_model = args.new_model
	S = int(args.S)
	N = int(args.N)
	model_name = str(args.model_name)

	rnn = RNN()

	if not new_model:
		try:
			rnn.set_weights('./rnn/'+model_name+'/'+model_name+'.h5')
		except:
			print("Either set --new_model or ensure ./rnn/weights.h5 exists")
			raise
	elif not os.path.isdir('./rnn/'+model_name):
		os.mkdir('./rnn/'+model_name)
		os.mkdir('./rnn/'+model_name+'/log/')


	filelist, N = get_filelist(S, N)

	for step in range(N):
		print('STEP ' + str(step))

		zS, zB, action, rew = get_batch(filelist, step)
		rnn_input = np.concatenate([zS[:, :-1, :].reshape(1, -1, 32), action[:, :-1, :], rew[:, :-1, :]], axis = 2)
		rnn_output = np.concatenate([zB[:, 1:, :].reshape(1, -1, 32), rew[:, 1:, :]], axis = 2) 

		if step == 0:
			np.savez_compressed(ROOT_DIR_NAME + 'rnn_files.npz', rnn_input = rnn_input, rnn_output = rnn_output)

		rnn.train(rnn_input, rnn_output, model_name)

		if step % 10 == 0:

			rnn.model.save_weights('./rnn/'+model_name+'/'+model_name+'_weights.h5')

	rnn.model.save_weights('./rnn/'+model_name+'/'+model_name+'_weights.h5')




if __name__ == "__main__":
		parser = argparse.ArgumentParser(description=('Train RNN'))
		parser.add_argument('--S',default = 0, help='number of episodes to skip to train')
		parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
		parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
		parser.add_argument('--model_name', default = 'rnn', help='name of model to use')

		args = parser.parse_args()

		main(args)
