# Analyzing free energy surface

## Convergence of FES

### Simple committor analysis

```Python
import numpy as np
committor = np.where(dat[1] > 1.65, 1, (np.where(dat[1] < 1.45, -1, 0)))  # R3
```

### Committor analysis with colormap

A script to calculate the sum of six torsion angles of carbons in the cyclohexene ring. 
Each sampling points is highlighted in different shade color, e.g., in this case, winter colormap.

```Python
#!/usr/bin/env python3

# Plot sum of torsion angles as a function of time
# highlighted with symmetric colormap (normal colormap + its reverse)

import numpy as np
import matplotlib.pyplot as plt

# Filename of files to read, e.g., multiplot_1.dat, multiplot_2.dat, ...
f = [np.loadtxt("multiplot_" + str(i+1) + ".dat") for i in range(6)]
f_sum = f[0].T[1] + f[1].T[1] + f[2].T[1] + f[3].T[1] + f[4].T[1] + f[5].T[1]
# fs to ps
x = f[0].T[0][200000:300000] * 0.5 / 1000
y = f_sum[300000:400000]

plt.figure(figsize=(10,8))
# plt.scatter(x, y, s=1, c=['r' if v > 8 or v < -8 else 'b' for v in y])

# upper
x1 = x[y >= 0]
y1 = y[y >= 0]
plt.scatter(x1, y1, s=0.5, c=y1, cmap="winter_r")
# lower
x2 = x[y < 0]
y2 = y[y < 0]
plt.scatter(x2, y2, s=0.5, c=y2, cmap="winter")

plt.xlabel("Simulation time (ps)", fontsize=20)
plt.ylabel("Sum of torsion angles ($\degree$)", fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlim(100,130)
plt.ylim(-32,32)
plt.tight_layout()
plt.show()
```