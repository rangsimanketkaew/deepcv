# Analyze Metastable States

## Visualize the latent space

The latent space of the compressed data is saved every N-th epoch while training (N is defined by `save_every_n_epoch` key).
Each saved latent space visualizes feature representations that can be used to determine the ability of the sampling to find newly unvisitted regions on the conformational space.

After model training is completed, check `autoencoder` folder in the output folder and you will find latent space plots.

```sh
├── autoencoder
│   ├── ...
│   ├── latent_space_0_epochs.png
│   ├── latent_space_10_epochs.png
│   ├── latent_space_20_epochs.png
│   ├── latent_space_30_epochs.png
│   ├── latent_space_40_epochs.png
│   ├── ...
```
