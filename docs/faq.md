# Frequently Asked Questions

## Installation

**Q: `ImportError: .../tensorflow/.../_pywrap_tensorflow_internal.so: undefined symbol: _ZN10ten...cEEES7_`**

**A:** You may get this error if you installed TensorFlow using pip. There is no workaround for this error. 
Try using conda to install TensorFlow instead with:
```
conda install tensorflow
```

**Q: TensorFlow `libdevice not found`**

**A:** Try installing `cudatoolkit` with
```
conda install -c anaconda cudatoolkit
```
