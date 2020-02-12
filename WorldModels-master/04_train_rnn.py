from rnn.arch import RNN
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR_NAME = './data/'
SERIES_DIR_NAME = './data/series_'


def get_filelist(S, N, model_name):
    filelist = os.listdir(SERIES_DIR_NAME+model_name)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)


    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    return filelist[S:], N


def random_batch(filelist, batch_size, model_name):
	N_data = len(filelist)
	indices = np.random.permutation(N_data)[0:batch_size]

	z_listS = []
	z_listB = []
	action_list = []
	rew_list = []
	done_list = []

	for i in indices:
		try:
			new_data = np.load(SERIES_DIR_NAME + model_name + '/' + filelist[i])

			muS = new_data['muS']
			muB = new_data['muB']
			log_varS = new_data['log_varS']
			log_varB = new_data['log_varB']
			action = new_data['action']
			reward = new_data['reward']
			done = new_data['done']

			reward = np.expand_dims(reward, axis=2)
			done = np.expand_dims(done, axis=2)


			s = log_varS.shape

			zS = muS + np.exp(log_varS/2.0) * np.random.randn(*s)
			zB = muB + np.exp(log_varB/2.0) * np.random.randn(*s)

			z_listS.append(zS)
			z_listB.append(zB)
			action_list.append(action)
			rew_list.append(reward)
			done_list.append(done)
		except:
			print("Ignoring", filelist[i])

	z_listS = np.array(z_listS)
	z_listB = np.array(z_listB)
	action_list = np.array(action_list)
	rew_list = np.array(rew_list)
	done_list = np.array(done_list)

	return z_listS, z_listB, action_list, rew_list, done_list

def main(args):
	
	new_model = args.new_model
	S = int(args.S)
	N = int(args.N)
	steps = int(args.steps)
	batch_size = int(args.batch_size)
	model_name = str(args.model_name)

	rnn = RNN()

	if not new_model:
		try:
			rnn.set_weights('./rnn/'+model_name+'_weights.h5')
		except:
			print("Either set --new_model or ensure ./rnn/"+model_name+"_weights.h5 exists")
			raise


	filelist, N = get_filelist(S, N, model_name)


	for step in range(steps):
		print('STEP ' + str(step))

		zS, zB, action, rew ,done = random_batch(filelist, batch_size, model_name)

		rnn_input = np.concatenate([zS[:, :-1, :], action[:, :-1, :], rew[:, :-1, :]], axis = 2)
		rnn_output = np.concatenate([zB[:, 1:, :], rew[:, 1:, :]], axis = 2) 

		if step == 0:
			np.savez_compressed(ROOT_DIR_NAME + 'rnn_files.npz', rnn_input = rnn_input, rnn_output = rnn_output)

		rnn.train(rnn_input, rnn_output, model_name)

		rnn.model.save_weights('./rnn/'+model_name+'/'+model_name+'_weights.h5')



if __name__ == "__main__":
		parser = argparse.ArgumentParser(description=('Train RNN'))
		parser.add_argument('--S',default = 0, help='number of episodes to skip to train')
		parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
		parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
		parser.add_argument('--steps', default = 4000, help='how many rnn batches to train over')
		parser.add_argument('--batch_size', default = 100, help='how many episodes in a batch?')
		parser.add_argument('--model_name', default = "default", help='name of the model')

		args = parser.parse_args()

		main(args)
