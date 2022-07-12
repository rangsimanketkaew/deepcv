# Free energy surface analysis

## Convergence of FES

Committor analysis (example)

```Python
import numpy as np
committor = np.where(dat[1] > 1.65, 1, (np.where(dat[1] < 1.45, -1, 0)))  # R3
```
