# Free energy surface and metastable states sampling analysis <!-- omit in toc -->

- [Step 1: Convergence of FES](#step-1-convergence-of-fes)
- [Step 2: Visualizing DAENN model's latent space](#step-2-visualizing-daenn-models-latent-space)

## Step 1: Convergence of FES

Committor analysis (example)
```python
import numpy as np
committor = np.where(dat[1] > 1.65, 1, (np.where(dat[1] < 1.45, -1, 0)))  # R3
```

## Step 2: Visualizing DAENN model's latent space
