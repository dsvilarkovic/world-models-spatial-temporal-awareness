#python 02_train_vae.py --new_model

from vae.arch import VAE
import argparse
import numpy as np
import config
import os
import cv2
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DIR_NAME = './data/rollout_doom/'
SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64

def crop(image, scale):
  size = len(image)
  newsize = int(np.round(size * scale))
  border = int(round((size-newsize) / 2))
  left = border
  right = left + newsize
  top = border
  bottom = top + newsize
  return image[top:bottom, left:right]
  
def import_data(S, N, filelist):
  length_filelist = len(filelist)

  if length_filelist < N:
    N = length_filelist
    
  filelist = filelist[S:N]
  file_count = 0
  data = np.asarray([]).astype(np.float32)
  for file in filelist:
    try:
      new_data = np.load(DIR_NAME + file)['frames']/255.
      if file_count == 0:
        data = new_data
      else :
        data = np.vstack([data, new_data])

      file_count += 1
    except:
      print('Skipped {}...'.format(file))

  return data



def main(args):

  new_model = args.new_model
  S = int(args.S)
  N = int(args.N)
  batch = int(args.batch)
  model_name = str(args.model_name)
  print(args.alpha)
  alpha = float(args.alpha)
  vae = VAE()

  if not new_model:
    try:
      vae.set_weights('./vae/'+model_name+'/'+model_name+'_weights.h5')

    except:
      print("Either set --new_model or ensure ./vae/weights.h5 exists")
      raise
  else:
    if os.path.isdir('./vae/'+model_name):
      print("A model with this name already exists")
    else:
      os.mkdir('./vae/'+model_name)
      os.mkdir('./vae/'+model_name+'/log/')
  
  filelist = os.listdir(DIR_NAME)
  filelist = [x for x in filelist if x != '.DS_Store' and x != '.gitignore']
  filelist.sort()
  
  for i in range(round(float(N-S)/batch)):
    data = import_data(S+i*batch, S+(i+1)*batch, filelist)
    dataS = []
    dataB = []
    for d in data:
      beta = alpha + np.random.rand() * (1-alpha)
      dataS.append(cv2.resize(crop(d, alpha*beta), dsize=(SCREEN_SIZE_X, SCREEN_SIZE_Y), interpolation=cv2.INTER_CUBIC))
      dataB.append(cv2.resize(crop(d, beta), dsize=(SCREEN_SIZE_X, SCREEN_SIZE_Y), interpolation=cv2.INTER_CUBIC))
    
    dataS = np.asarray(dataS)
    dataB = np.asarray(dataB)

    vae.train(dataS, dataB, model_name) # uncomment this to train augmenting VAE, simple RNN (2)
    #vae.train(np.vstack([dataS, dataB]), np.vstack([dataS, dataB]), model_name) # uncomment this to train simple VAE, RNN (1)
        
    vae.save_weights('./vae/'+model_name+'/'+model_name+'_weights.h5')
    
    print('Imported {} / {}'.format(S+(i+1)*batch, N))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE'))
  parser.add_argument('--N',default = 10000, help='number of episodes to use to train')
  parser.add_argument('--S',default = 0, help='number of episodes to skip to train')
  parser.add_argument('--batch',default = 10, help='number of episodes to load in one iteration (not overload RAM)')
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--model_name', default = 'vae_train', help='name of directory where results will be saved')
  parser.add_argument('--alpha', default = 0.82, help='scaling factor')
  
  args = parser.parse_args()

  main(args)
