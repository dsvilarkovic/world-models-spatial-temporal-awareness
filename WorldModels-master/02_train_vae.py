from vae.arch import VAE
import argparse
import numpy as np
import config
import os
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DIR_NAME = './data/rollout_'
M = 300
SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64
  
def import_data(S, N, filelist, model_name):
  length_filelist = len(filelist)

  if length_filelist < N:
    N = length_filelist
    
  filelist = filelist[S:N]
  file_count = 0
  dataS = np.asarray([]).astype(np.float32)
  dataB = np.asarray([]).astype(np.float32)
  for file in filelist:
      try:
        new_dataS = np.load(DIR_NAME + model_name + '/' + file)['obsS'][::,::,::(1 if np.random.rand() < .5 else -1),::]
        new_dataB = np.load(DIR_NAME + model_name + '/' + file)['obsB'][::,::,::(1 if np.random.rand() < .5 else -1),::]
        if file_count == 0:
          dataS = new_dataS
          dataB = new_dataB
        else :
          dataS = np.vstack([dataS, new_dataS])
          dataB = np.vstack([dataB, new_dataB])
          
        file_count += 1
      except:
        print('Skipped {}...'.format(file))

  return dataS, dataB



def main(args):

  new_model = args.new_model
  S = int(args.S)
  N = int(args.N)
  batch = int(args.batch)
  epochs = int(args.epochs)
  model_name = str(args.model_name)

  vae = VAE()

  if not new_model:
    try:
      vae.set_weights('./vae/'+model_name+'_weights.h5')
    except:
      print("Either set --new_model or ensure ./vae/"+model_name+"_weights.h5 exists")
      raise
  elif not os.path.isdir('./vae/'+model_name):
    os.mkdir('./vae/'+model_name)
    os.mkdir('./vae/'+model_name+'/log/')
  
  filelist = os.listdir(DIR_NAME+model_name)
  filelist = [x for x in filelist if x != '.DS_Store' and x != '.gitignore']
  filelist.sort()
  N = max(N, len(filelist))

  for i in range(int(round(float(N-S)/batch)+1)):
    dataS, dataB = import_data(S+i*batch, S+(i+1)*batch, filelist, model_name)
    for epoch in range(epochs):
      vae.train(dataS, dataB, model_name) # uncomment this to train augmenting VAE, simple RNN (2)
      #vae.train(np.vstack([dataS, dataB]), np.vstack([dataS, dataB]), model_name) # uncomment this to train simple VAE, RNN (1)
    vae.save_weights('./vae/'+model_name+'/'+model_name+'_weights.h5')
    
    print('Imported {} / {}'.format(S+(i+1)*batch, N))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  parser.add_argument('--S',default = 0, help='number of episodes to skip to train')
  parser.add_argument('--batch',default = 50, help='number of episodes to load in one iteration (not overload RAM)')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--epochs', default = 1, help='number of epochs to train for')
  parser.add_argument('--model_name', default = "default", help='Name of the model')
  args = parser.parse_args()

  main(args)
