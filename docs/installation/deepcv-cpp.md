# Install DeepCV C++

## Compile TensorFlow C++ API

You can follow the step-by-step instruction to build TensorFlow C++ API from source using GCC 9.3.1 for C++ 14:

1. Install Python 3.7 or a newer version and Bazel 3.7.2
    
2. Install ProtoBuf 3.9.2 package with `-D_GLIBCXX_USE_CXX11_ABI=0` flag for ABI compatibility
    
3. Download the tarball from https://github.com/tensorflow/tensorflow/releases/tag/v2.7.4 to a local machine
    ```sh
    tar -xzvf v2.7.4.tar.gz
    cd tensorflow-2.7.4/
    ```
    
4. Compile TensorFlow framework
    ```sh
    bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
        --config=opt -c opt \
        //tensorflow:libtensorflow.so \
        //tensorflow:libtensorflow_cc.so \
        //tensorflow:libtensorflow_framework.so \
        //tensorflow:install_headers
    ```

5. Create a single TensorFlow directory for C++ linkage
    ```sh
    export LIB_TF="/usr/local/tensorflow/"
    sudo mkdir $LIB_TF
    sudo cp -r bazel-bin/tensorflow/include/ $LIB_TF
    sudo cp -r /home/rangsiman/protobuf-3.9.2/include/google/ $LIB_TF/include/
    sudo mkdir $LIB_TF/lib
    sudo cp -r bazel-bin/tensorflow/*.so* $LIB_TF/lib/
    sudo cp -r /home/rangsiman/protobuf-3.9.2/lib/*.so* $LIB_TF/lib/
    ```
    
More details on package dependencies installation and compilation of TensorFlow C++ API are available at https://github.com/rangsimanketkaew/tensorflow_cpp_api.

> The procedure above should also work with a new version or Python and Bazel.

## Compile PLUMED with DeepCV and TensorFlow

DeepCV is implemented in PLUMED as a standalone module `deepcv`. The source code of a modified version of PLUMED is available at https://gitlab.uzh.ch/lubergroup/plumed2-deepcv. The following is instruction for compiling PLUMED with TensorFlow and DeepCV:

```sh
# Clone PLUMED2 DeepCV repository to local machine
git clone git@gitlab.uzh.ch:lubergroup/plumed2-deepcv.git
cd plumed2-deepcv/

# Choose variable to directory to install PLUMED2, e.g.
export PREFIX="/home/rangsiman/plumed2-deepcv-install/"
# Set variable to TensorFlow directory for linkage, e.g.
export LIB_TF="/usr/local/tensorflow/"

# Configuring libraries
./configure --prefix=$PREFIX --enable-rpath \
    CXXFLAGS="-O3 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC" \
    CPPFLAGS="-I${LIB_TF}/include/ -L${LIB_TF}/lib/" \
    LDFLAGS="-L${LIB_TF}/lib/ -ltensorflow_cc -ltensorflow_framework -Wl,-rpath,$LIB_TF/lib/" \
    --disable-external-lapack --disable-external-blas \
    --disable-python --disable-libsearch --disable-static-patch \
    --disable-static-archive --enable-mpi --enable-cxx=14 \
    --disable-modules --enable-modules=adjmat+deepcv

# Compile and install in parallel using, e.g., 8 processors
make -j 8 CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -fPIC" 2>&1 | tee make.log
make -j 8 install 2>&1 | tee make_install.log
export LD_LIBRARY_PATH="${LIB_TF}/lib/":"$PREFIX/lib/":$LD_LIBRARY_PATH
```

Note that the above procedure will compile PLUMED with MPI. Use `--disable-mpi` instead if you prefer serial compilation to parallel compilation.

## Compile DeepCV C++ module

1. Go to DeepCV directory:
    ```sh
    cd /path/to/plumed2/src/deepcv/
    ```

2. The users can compile source code (`deepcv.cpp`) using either GCC compiler `g++`:
    ```sh
    # Set variable to TensorFlow directory for linkage, e.g.
    export LIB_TF=/usr/local/tensorflow/

    # compile
    g++ -Wall -fPIC -o deepcv.o deepcv.cpp \
        -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -fPIC \
        -I${LIB_TF}/include/ -L${LIB_TF}/lib \
        -ltensorflow -ltensorflow_cc -ltensorflow_framework
    ```
    or use Make (`make`):
    ```sh
    make CXXFLAGS="-std=c++14 -fPIC"
    ```
    Related header files are required to be in the same directory of source codes, else one needs to edit the absolute or relative path in Makefile.

3. Then build a shared object file (`deepcv.so`):
    ```sh
    g++ -shared -o deepcv.so deepcv.o
    ```
