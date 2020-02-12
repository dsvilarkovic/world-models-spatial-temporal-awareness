from vae.arch import VAE
import argparse
import config
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ROOT_DIR_NAME = "./data/"
ROLLOUT_DIR_NAME = "./data/rollout_"
SERIES_DIR_NAME = "./data/series_"


def get_filelist(S, N, model_name):
    filelist = os.listdir(ROLLOUT_DIR_NAME+model_name)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)


    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    return filelist[S:], N

def encode_episode(vae, obs, action, reward, done):

    done = done.astype(int)  
    reward = np.where(reward>0, 1, 0) * np.where(done==0, 1, 0)

    mu, log_var = vae.encoder_mu_log_var.predict(obs)
    
    initial_mu = mu[0, :]
    initial_log_var = log_var[0, :]

    return (mu, log_var, action, reward, done, initial_mu, initial_log_var)



def main(args):

    N = int(args.N)
    S = int(args.S)
    model_name = str(args.model_name)
    vae = VAE()

    try:
      vae.set_weights('./vae/'+model_name+'/'+model_name+'_weights.h5')
    except:
      print("./vae/"+model_name+"_weights.h5 does not exist - ensure you have run 02_train_vae.py first")
      raise


    filelist, N = get_filelist(S, N, model_name)

    file_count = 0

    initial_musS = []
    initial_log_varsS = []
    initial_musB = []
    initial_log_varsB = []

    for file in filelist:
      try:
      
        rollout_data = np.load(ROLLOUT_DIR_NAME + model_name + '/' + file)

        muS, log_varS, action, reward, done, initial_muS, initial_log_varS = \
                encode_episode(vae, rollout_data['obsS'], rollout_data['action'], rollout_data['reward'], rollout_data['done'])
        muB, log_varB, action, reward, done, initial_muB, initial_log_varB = \
                encode_episode(vae, rollout_data['obsB'], rollout_data['action'], rollout_data['reward'], rollout_data['done'])

        if not os.path.isdir(SERIES_DIR_NAME + model_name):
          os.mkdir(SERIES_DIR_NAME + model_name)
        np.savez_compressed(SERIES_DIR_NAME + model_name + "/" + file, muS=muS, log_varS=log_varS, muB=muB, log_varB=log_varB, action = action, reward = reward, done = done)
        initial_musS.append(initial_muS)
        initial_log_varsS.append(initial_log_varS)
        initial_musB.append(initial_muB)
        initial_log_varsB.append(initial_log_varB)

        file_count += 1

        if file_count%50==0:
          print('Encoded {} / {} episodes'.format(S+file_count, N))

      except:
        print('Skipped {}...'.format(file))

    print('Encoded {} / {} episodes'.format(S+file_count, N))

    initial_musS = np.array(initial_musS)
    initial_log_varsS = np.array(initial_log_varsS)
    initial_musB = np.array(initial_musB)
    initial_log_varsB = np.array(initial_log_varsB)

    print('ONE MU SHAPE = {}'.format(initial_musS.shape))
    print('INITIAL MU SHAPE = {}'.format(initial_musS.shape))

    np.savez_compressed(ROOT_DIR_NAME + 'initial_zS.npz', initial_muS=initial_musS, initial_log_varS=initial_log_varsS)
    np.savez_compressed(ROOT_DIR_NAME + 'initial_zB.npz', initial_muB=initial_musB, initial_log_varB=initial_log_varsB)
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Generate RNN data'))
  parser.add_argument('--S',default = 0, help='number of episodes to skip to train')
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  parser.add_argument('--model_name', type=str, default = "default", help='name of the model')
  args = parser.parse_args()

  main(args)
