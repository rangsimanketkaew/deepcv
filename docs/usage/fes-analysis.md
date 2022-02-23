# Free energy surface analysis

## Convergence of FES

Committor analysis (example)

```Python
import numpy as np
committor = np.where(dat[1] > 1.65, 1, (np.where(dat[1] < 1.45, -1, 0)))  # R3
```

## Visualizing DAENN model's latent space

Check `au_visual.py` script. It can be used to visualize feature representations (latent space) of while training a model.
