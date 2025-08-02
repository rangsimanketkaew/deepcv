# Workflow

To help the users easily understand the whole procedure of DAENN we provide an end-to-end workshop pipeline of DAENN below. The process starts from the step where we use unbiased molecular dynamics to generate a trajectory of reactant conformers. Then it proceeds to feature calculation, model training, and CV generation, respectively. The model outputs are then used as CVs functions for metadynamics simulation (or any other enhanced sampling methods) to calculate free energy surfaces (FESs) of the chemical system of interest.

<figure markdown>
  ![Image title](../img/daenn-workflow.png){ width="1200" }
  <figcaption>A complete workflow of DAENN and FES reconstruction.</figcaption>
</figure>
