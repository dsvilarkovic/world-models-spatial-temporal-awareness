Analysis of spacial and temporal awareness of the World Models Agent
======
[Report research paper is available](./ReportPaper.pdf)

## Abstract 
We explore a variation of the World Models architecture to visualize if intuitive human spatial and temporal representations are indeed captured by the reinforcement learning agent. We propose to enhance
the VAE training such that it learns how to enlarge,
in essence zooming out, given input frames. Using a
scaling factor on input frames the VAE managed to
learn how to scale an image and even predict a curve
that was outside of the original image. Additionally,
we managed to isolate the latent feature that provoked
the scaling. We also verified that the RNN leaned the
temporal information, and is for instance able to remember a fire ball that left the frame but reappears
when enlarging the image, which the VAE failed to
do.
## Run instructions

If you are currently on local, use the link to go to the colab folder: [1] to run the model and experiments, where all the rollouts are, and where the dependencies are already satisfied. To run the code, you have to add the DL folder to “My Drive”

You can train the Car racing model in the control.ipynb notebook and the Doom model in control_doom.ipynb. Each of those contains the rollout generation commands, VAE pretraining and RNN training (for the Doom model). You need to set the name of the model and the cropping factor (alpha). If you are on colab, you don’t need to generate new rollouts. The “Analyse VAE/RNN” boxes draw plots of the predicted enlarged frames by both VAE and RNN.

You can have access to experiments shown in the report in Report Results, in the order they appear in the report (first VAE Experiments, then Image scaling in latent space, then RNN Experiments). The models used there are already pretrained.

The code provided is based on the combination of the 2 following github repositories:
https://github.com/AppliedDataSciencePartners/WorldModels
https://github.com/hardmaru/WorldModelsExperiments/tree/master/doomrnn
where we added our contribution.

The VizDoom folder and game is the following:
https://github.com/shakenes/vizdoomgym

[1]: https://drive.google.com/drive/folders/18EM1C7Z20uIOKThzCs_vV0zz1f6p5DM3?usp=sharing


## Contributors

Rafael Bischof <br/>
Constantin Le Cleï <br/>
[Dušan Svilarković](https://github.com/dsvilarkovic) <br/>
Steven Battilana
