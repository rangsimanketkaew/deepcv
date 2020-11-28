import numpy as np

weight = np.load("output/model_weights.npz")
print(weight.files)

print(weight['layer5'].shape)
print(weight['layer6'].shape)
print(weight['layer7'].shape)
print(weight['layer8'].shape)
print(weight['layer9'].shape)
print(weight['layer10'].shape)

print(weight['layer7'])
