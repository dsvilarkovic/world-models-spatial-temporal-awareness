Run instructions
======

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
