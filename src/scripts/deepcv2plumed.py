"""
Deep learning-based collective variables (DeepCV)
https://gitlab.uzh.ch/LuberGroup/deepcv

Info:
28/11/2020 : Rangsiman Ketkaew
"""

"""
Generate PLUMED input file using neural-network-based collective variable for metadynamics simulation
"""

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import argparse
import numpy as np
from utils import util
from datetime import datetime


class WritePlumed:
    def __init__(self, output_plumed="plumed-NN.dat", colvar_file="COLVAR.log"):
        """Start writing PLUMED input file (.dat)

        Args:
            output_plumed (str): Plumed file name. Defaults to "plumed-NN.dat".
            colvar_file (str): Name of collective variable file. Defaults to "COLVAR.log".
        """
        self.output_plumed = output_plumed
        self.colvar_file = colvar_file
        self.num_feat = 0
        f = open(self.output_plumed, "w")
        f.write("# Deep learning-based collective variables for metadynamics simulation.\n")
        today = datetime.now().strftime("%b-%d-%y %H:%M:%S")
        f.write(f"# This file was generated by deepcv2plumed subroutine on {today}\n\n")
        f.write("# RESTART\n\n")
        f.write("UNITS LENGTH=A TIME=fs\n\n")
        f.close()

    def write_Zmat(self, num_atoms: int, stride=10):
        """Write internal coordinate (Z-matrix) based on the successive indexes i.e. 1, 2, 3, ..., N.

        Args:
            num_atoms (int): Number of atoms in the simulation system
            stride (int): Frequencies of the quantity to be printed. Defaults to 10.
        """
        f = open(self.output_plumed, "a")
        f.write(f"# Total atoms = {num_atoms}\n")
        # f.write(f"WHOLEMOLECULES ENTITY0=1-{num_atoms}\n\n")
        f.write("# Distance\n")
        f.write("d1: DISTANCE ATOMS=2,1\n")
        f.write("d2: DISTANCE ATOMS=3,1\n")
        self.num_feat += 2
        for i in range(num_atoms - 3):
            f.write(f"d{i+3}: DISTANCE ATOMS={i+4},{i+1}\n")
            self.num_feat += 1
        f.write("# Angle (radian)\n")
        f.write("a1: ANGLE ATOMS=3,1,2\n")
        self.num_feat += 1
        for i in range(num_atoms - 3):
            f.write(f"a{i+2}: ANGLE ATOMS={i+4},{i+1},{i+2}\n")
            self.num_feat += 1
        f.write("# Torsional angle (radian)\n")
        for i in range(num_atoms - 3):
            f.write(f"t{i+1}: TORSION ATOMS={i+4},{i+1},{i+2},{i+3}\n")
            self.num_feat += 1
        print(f">>> A number of Z-matrix inputs: {self.num_feat}")
        self.arg_input = [f'd{j+1}' for j in range(num_atoms - 1)] \
            + [f'a{j+1}' for j in range(num_atoms - 2)] \
            + [f't{j+1}' for j in range(num_atoms - 3)]
        arg = ",".join(self.arg_input)
        f.write(f"\nPRINT FILE=input_Zmat.log STRIDE={stride} ARG={arg}\n")
        f.close()
    
    def write_Zmat_by_index(self, index: list, stride=10):
        """Write internal coordinate (Z-matrix) based on user-defined atom index.

        Args:
            index (list): List of atom index
            stride (int): Frequencies of the quantity to be printed. Defaults to 10.
        """
        f = open(self.output_plumed, "a")
        f.write(f"# Total atoms = {len(index)}\n")
        index_ = ",".join(map(str, index))
        f.write(f"# Index: {index_}\n")
        # Define input (features)
        f.write("# Distance\n")
        f.write(f"d1: DISTANCE ATOMS={index[1]},{index[0]}\n")
        f.write(f"d2: DISTANCE ATOMS={index[2]},{index[0]}\n")
        self.num_feat += 2
        for i in range(len(index) - 3):
            f.write(f"d{i+3}: DISTANCE ATOMS={index[i+3]},{index[i]}\n")
            self.num_feat += 1
        f.write("# Angle (radian)\n")
        f.write("a1: ANGLE ATOMS=3,1,2\n")
        self.num_feat += 1
        for i in range(len(index) - 3):
            f.write(f"a{i+2}: ANGLE ATOMS={index[i+3]},{index[i]},{index[i+1]}\n")
            self.num_feat += 1
        f.write("# Torsional angle (radian)\n")
        for i in range(len(index) - 3):
            f.write(f"t{i+1}: TORSION ATOMS={index[i+3]},{index[i]},{index[i+1]},{index[i+2]}\n")
            self.num_feat += 1
        print(f">>> A number of Z-matrix inputs: {self.num_feat}")
        self.arg_input = [f'd{j+1}' for j in range(len(index) - 1)] \
            + [f'a{j+1}' for j in range(len(index) - 2)] \
            + [f't{j+1}' for j in range(len(index) - 3)]
        arg = ",".join(self.arg_input)
        f.write(f"\nPRINT FILE=input_Zmat.log STRIDE={stride} ARG={arg}\n")
        f.close()
    
    def write_SPRINT(self, sprint_index, stride=10):
        """Write function to calculate SPRINT coordinate of specified atom index.

        Args:
            sprint_index (list): Atomic type (symbol) and index
            stride (int): Frequencies of the quantity to be printed. Defaults to 10.
        """
        f = open(self.output_plumed, "a")
        f.write("\n# SPRINT coordinate\n")
        self.num_sprint = 0
        for i in sprint_index:
            self.num_sprint += len(i.split('=')[1].split(','))
            f.write(f"DENSITY LABEL={i.split('=')[0]} SPECIES={i.split('=')[1]}\n")
        print(f">>> A number of SPRINT inputs: {self.num_sprint}")
        self.num_feat += self.num_sprint
        f.write("\nCONTACT_MATRIX ...\n")
        f.write("!!!----------------------------------------!!!\n")
        f.write("!!! You have to define contact matrix here !!!\n")
        f.write("!!!----------------------------------------!!!\n")
        f.write("... CONTACT_MATRIX\n")
        f.write("SPRINT MATRIX=mat LABEL=ss\n\n")
        f.write(f"PRINT ARG=ss.* FILE=input_SPRINT.log STRIDE={stride}\n")
        print(f">>> WARINING!!! You need to define contact matrix in SPRINT deck in {self.output_plumed}")
        arg = [f'ss.coord-{i}' for i in range(self.num_sprint)]
        self.arg_input += arg
        f.close()

    def write_NeuralNetwork(self, weight, bias, kw, func_1: str, func_2: str, func_3: str, leakyrelu_coeff=0.1, 
                            elu_coeff=0.1, stride=10, stride_flush=50):
        """Initialize class with a set of required parameters.
        Input vector will be multiplied by weights for each node. Bias will also be added.
        Activation function will be applied for each node. 
        
        Available activation functions
        ------------------------------
        linear: y = x
        binary: step(x)
        sigmoid: y = 1.0/(11.0+exp(-x))
        tanh: y = exp(x) - exp(-x) / exp(x) + exp(-x)
        relu: y = step(x)*x
        leakyrelu: y = a*step(-x) + step(x)*x
        elu: y = a*(exp(step(-x)) - 1) + step(x)*x

        Args:
            weight (array): Trained neural network model.
            bias (array): Name of PLUMED input file.
            ke (list): List of keywords of array in weight and bias arrays.
            func_1 (str): Activation function of hidden layer 1.
            func_2 (str): Activation function of hidden layer 2.
            func_3 (str): Activation function of hidden layer 3 (encoded layer).
            leakyrely_coeff (int, float): Coefficient of LeakyReLU activation function. Defaults to 0.1.
            elu_coeff (int, float): Coefficient of ELU activation function. Defaults to 0.1.
        """
        kw_1, kw_2, kw_3 = kw
        act_func = {
            "linear": lambda v: f"(x{v:+.8f})",
            "binary": lambda v: f"step(x{v:+.8f})",
            "sigmoid": lambda v: f"1.0/(1.0+exp(-x{v:+.8f}))",
            "tanh": lambda v: f"exp(x{v:+.8f})-exp(-x{v:+.8f}))/(exp(x{v:+.8f})+exp(-x{v:+.8f})",
            "relu": lambda v: f"step(x{v:+.8f})*(x{v:+.8f})",
            "leakyrelu": lambda v: f"{leakyrelu_coeff}*step(-(x{v:+.8f}))+step(x{v:+.8f})*(x{v:+.8f})",
            "elu": lambda v: f"{leakyrelu_coeff}*(exp(step(-(x{v:+.8f})))-1)+step(x{v:+.8f})*(x{v:+.8f})"
        }
        print(f">>> A number of total inputs: {self.num_feat}")
        print(f">>> Activation functions: {func_1.lower()}, {func_2}, {func_3}")
        func_1 = act_func[func_1.lower()]
        func_2 = act_func[func_2.lower()]
        func_3 = act_func[func_3.lower()]
        size_layer1 = weight[kw_1].shape[1]
        size_layer2 = weight[kw_2].shape[1]
        size_layer3 = weight[kw_3].shape[1]
        print(f">>> Size of layers: {size_layer1}, {size_layer2}, {size_layer3}")

        # Check if number of inputs and weight of the first layer are corresponding
        if self.num_feat != weight[kw_1].shape[0]:
            exit(f"Error: Input size ({self.num_feat}) and weight size ({weight[kw_1].shape[0]}) are not corresponding")

        f = open(self.output_plumed, "a")
        f.write(f"\n# Total inputs = {self.num_feat}\n")
        f.write("\n# Neural network\n")
        self.arg_input = ",".join(self.arg_input)
        #----------- LAYER 1 -----------#
        f.write("#===== Hidden layer 1 =====#\n")
        # Loop over nodes (neurons) in the present hidden layer
        for i in range(size_layer1):
            # Multiply input by weight
            f.write("COMBINE ...\n")
            f.write(f"\tLABEL=hl1_n{i+1}_mult\n")
            f.write(f"\tARG={self.arg_input}\n")
            w = map(str, weight[kw_1][:,i])
            # print(len(list(w)))
            w = ",".join(w)
            f.write(f"\tCOEFFICIENTS={w}\n")
            f.write(f"\tPOWERS={','.join(['1'] * self.num_feat)}\n")
            f.write("\tPERIODIC=NO\n")
            f.write("... COMBINE\n")
        f.write("\n")
        for i in range(size_layer1):
            # Add bias and apply activation function
            f.write("MATHEVAL ...\n")
            f.write(f"\tLABEL=hl1_n{i+1}_out\n")
            f.write(f"\tARG=hl1_n{i+1}_mult\n")
            f.write(f"\tFUNC=({func_1(bias[kw_1][i])})\n")
            f.write("\tPERIODIC=NO\n")
            f.write("... MATHEVAL\n")
        arg = ','.join([f'hl1_n{i+1}_out' for i in range(size_layer1)])
        f.write(f"\nPRINT FILE=layer1.log STRIDE={stride} ARG={arg}\n")

        #----------- LAYER 2 -----------#
        f.write("\n#===== Hidden layer 2 =====#\n")
        # Loop over nodes (neurons) in the present hidden layer
        for i in range(size_layer2):
            # Multiply input by weight
            f.write("COMBINE ...\n")
            f.write(f"\tLABEL=hl2_n{i+1}_mult\n")
            arg = [f'hl1_n{j+1}_out' for j in range(size_layer1)]
            arg = ",".join(arg)
            f.write(f"\tARG={arg}\n")
            w = map(str, weight[kw_2][:,i])
            w = ",".join(w)
            f.write(f"\tCOEFFICIENTS={w}\n")
            f.write(f"\tPOWERS={','.join(['1'] * size_layer1)}\n")
            f.write("\tPERIODIC=NO\n")
            f.write("... COMBINE\n")
        f.write("\n")
        for i in range(size_layer2):
            # Add bias and apply activation function
            f.write("MATHEVAL ...\n")
            f.write(f"\tLABEL=hl2_n{i+1}_out\n")
            f.write(f"\tARG=hl2_n{i+1}_mult\n")
            f.write(f"\tFUNC=({func_2(bias[kw_2][i])})\n")
            f.write("\tPERIODIC=NO\n")
            f.write("... MATHEVAL\n")
        arg = ','.join([f'hl2_n{i+1}_out' for i in range(size_layer2)])
        f.write(f"\nPRINT FILE=layer2.log STRIDE={stride} ARG={arg}\n")

        #----------- LAYER 3 -----------#
        f.write("\n#===== Hidden layer 3 =====#\n")
        # Loop over nodes (neurons) in the present hidden layer
        for i in range(size_layer3):
            # Multiply input by weight
            f.write("COMBINE ...\n")
            f.write(f"\tLABEL=hl3_n{i+1}_mult\n")
            arg = [f'hl2_n{j+1}_out' for j in range(size_layer2)]
            arg = ",".join(arg)
            f.write(f"\tARG={arg}\n")
            w = map(str, weight[kw_3][:,i])
            w = ",".join(w)
            f.write(f"\tCOEFFICIENTS={w}\n")
            f.write(f"\tPOWERS={','.join(['1'] * size_layer2)}\n")
            f.write("\tPERIODIC=NO\n")
            f.write("... COMBINE\n")
        f.write("\n")
        for i in range(size_layer3):
            # Add bias and apply activation function
            f.write("MATHEVAL ...\n")
            f.write(f"\tLABEL=hl3_n{i+1}_out\n")
            f.write(f"\tARG=hl3_n{i+1}_mult\n")
            f.write(f"\tFUNC=({func_3(bias[kw_3][i])})\n")
            f.write("\tPERIODIC=NO\n")
            f.write("... MATHEVAL\n")
        arg = ','.join([f'hl3_n{i+1}_out' for i in range(size_layer3)])
        f.write(f"\nPRINT FILE=layer3.log STRIDE={stride} ARG={arg}\n")

        # Update PLUMED output files
        f.write(f"\n# Update all files every {stride_flush} steps\n")
        f.write(f"FLUSH STRIDE={stride_flush}\n")

        # Save/Print CVs
        self.colvar = [f'hl3_n{j+1}_out' for j in range(size_layer3)]
        colvar = ",".join(self.colvar)
        f.write(f"\nPRINT ARG={colvar} STRIDE=1 FILE={self.colvar_file}\n")
        f.write(f"\n# You can use the following variables as CVs: {colvar}\n")
        f.close()

    def write_MetaD(self, metad_label="metad", sigma=0.05, height=2.0, pace=50, well_tempered=True, temp=300, 
                    bias_factor=25, hill_name="HILLS", bias_name="bias.log", stride_metad=1):
        """Write input for well-tempered metadynamics simulation

        Args:
            metad_label (str, optional): Label of metadynamics object. Defaults to "metad".
            sigma (float, optional): Width of Gaussian potential. Defaults to 0.05.
            height (float, optional): Height of Gaussian potential. Defaults to 2.0.
            pace (int, optional): Frequency of Gaussian potential deposition. Defaults to 50.
            well_tempered (bool, optional): Turn on/off well-tempered metadynamics. Defaults to True.
            temp (int, optional): Temperature (in Kelvin). Defaults to 300.
            bias_factor (int, optional): Factor of well-tempered bias. Defaults to 25.
            hill_name (str, optional): Name of Gaussian HILLS file. Defaults to "HILLS".
            bias_name (str, optional): Name of bias output. Defaults to "bias.log".
            stride_metad (str, optional): Value of stride parameter for printing metadynamics bias. Defaults to 1.
        """
        # Metadynamics input
        f = open(self.output_plumed, "a")
        f.write("\nMETAD ...\n")
        f.write("\tLABEL=metad\n")
        colvar = ",".join(self.colvar)
        f.write(f"\tARG={colvar}\n")
        f.write(f"\tSIGMA={','.join([str(sigma)] * len(self.colvar))}\n")
        f.write(f"\tHEIGHT={height}\n")
        f.write(f"\tPACE={pace}\n")
        if well_tempered:
            f.write(f"\tTEMP={temp}\n")
            f.write(f"\tBIASFACTOR={bias_factor}\n")
        f.write(f"\tFILE={hill_name}\n")
        f.write(f"... METAD\n")
        f.write(f"\nPRINT ARG={colvar},{metad_label}.bias FILE={bias_name} STRIDE={stride_metad}\n\n")
        f.close()


if __name__ == "__main__":
    info = ("Generate PLUMED input file (.dat) from neural network model (weights and biases) trained by DeepCV.")
    parser = argparse.ArgumentParser(description=info)
    parser.add_argument("--input", "-i", metavar="INPUT.json", type=str, required=True,
        help="Input file (.json) that is used for training model. \
        The input file must contain either absolute or relative path of the directory \
        in which the weight and bias NumPy's compresses file format (.npz) stored.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--num-atoms", "-n", metavar="N", type=int, default=None,
        help="The number of atoms in the simulation system that is used to train the model. \
        Cannot be used with -a option.")
    group.add_argument("--atom-index", "-a", metavar="INDEX", type=int, nargs="+", default=None,
        help="List of atomic index (1-based array index) that will be used. \
        If the specified index is 0-based index, please use '--increment-index' to change from 0-based to 1-based index. \
        Cannot be used with -n option.")
    parser.add_argument("--sprint-index", "-s", metavar="LABEL_INDEX", type=str, default=None,
        help="Atomic type (symbol) and index of atoms to compute SPRINT coordinate. \
        Separator between index is ',' and between atomic type is ':'.\
        For example, '--sprint-index O=3,5,7,9:H=2,4,6,8,10'.")
    parser.add_argument("--increment-index", "-ii", action='store_true',
        help="If this argument is set, the index will be changed from 0-based to 1-based index. \
        This means that the index will be incremented by 1. For example, [3, 4, 5, 6] will become [4, 5, 6, 7]. \
        This option is useful when the atomic index is 0-based.")
    parser.add_argument("--output", "-o", metavar="FILENAME.dat", type=str, default="plumed_NN.dat", 
        help="Name of a PLUMED input file (.dat). Default to 'plumed_NN.dat'")

    args = parser.parse_args()

    #------- Read input -------#
    if not os.path.isfile(args.input): exit(f'Error: No such file "{args.input}"')
    json = util.load_json(args.input)
    func_1 = json["network"]["func_1"]
    func_2 = json["network"]["func_2"]
    func_3 = json["network"]["func_3"]
    folder = json["output"]["out_dir"]
    weight = json["output"]["out_weights_npz"]
    bias = json["output"]["out_biases_npz"]
    weight = folder + "/" + weight
    bias = folder + "/" + bias
    if not os.path.isfile(weight): exit(f'Error: No such file "{weight}"')
    if not os.path.isfile(bias): exit(f'Error: No such file "{bias}"')

    #------- Print info -------#
    print(">>> Reading input file ...")
    print(f">>> Weight: {weight}")
    print(f">>> Bias  : {bias}")

    #------- Check file -------#
    weight = np.load(weight)
    bias = np.load(bias)
    kw = weight.files[:3] # get keywords of first three layers

    #------- Start writing -------#
    p = WritePlumed(output_plumed=args.output)
    if not args.num_atoms and not args.atom_index:
        parser.error("either --num-atoms (-n) or --atom-index (-a) must be specified.")
    elif args.num_atoms and args.atom_index:
        parser.error("--num-atoms (-n) or --atom-index (-a) cannot be specified at the same time.")
    if args.num_atoms:
        p.write_Zmat(args.num_atoms)
    elif args.atom_index:
        if args.increment_index:
            print(">>> Increment all atomic index by 1")
            args.atom_index = [i+1 for i in args.atom_index]
        else:
            print(">>> Atomic index incrementation is turned off")
        p.write_Zmat_by_index(args.atom_index)
    if args.sprint_index:
        p.write_SPRINT(args.sprint_index.split(":"))
    p.write_NeuralNetwork(weight, bias, kw, func_1, func_2, func_3)
    p.write_MetaD()
    print(f">>> Plumed data have been successfully written to '{os.path.abspath(args.output)}'")
