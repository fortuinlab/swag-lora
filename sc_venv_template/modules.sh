module --force purge

module load Stages/2023  GCC/11.3.0  OpenMPI/4.1.4
module load tensorboard #(could be logged with 2024)

module load Stages/2024 
module load GCC OpenMPI 

# Some base modules commonly used in AI
module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
# module load Flask Seaborn
# This module brings many dependencies we will use later
# module load PyQuil

# ML Frameworks
module load  PyTorch scikit-learn
#  torchvision PyTorch-Lightning
module load dill

# module load Stages/2024 
# module load Stages/2023
# module load GCC OpenMPI 
# # Some base modules commonly used in AI
# module load mpi4py numba tqdm matplotlib IPython SciPy-Stack bokeh git
# module load Flask Seaborn
# # This module brings many dependencies we will use later
# module load PyQuil

# # ML Frameworks
# # module load  PyTorch scikit-learn torchvision PyTorch-Lightning
# module load  PyTorch  scikit-learn  
# # torchvision 

# #updates modules  to the ones that are simialr to ours
# module load Stages/2024  GCCcore/.12.3.0 
# module load dill



# module load    PyTorch/2.1.2
# module load     tornado/6.3.2
# module load    tqdm/4.66.1
# module load     PyYAML/6.0
# module load  scikit-learn/1.3.1
# module load  networkx/3.1
# module load  IPython/8.14.0
# module load  dill/0.3.7
# module load  sympy/1.12

# module load    PyTorch/2.1.2
# module load     tornado
# module load    tqdm
# module load     PyYAML
# module load  scikit-learn
# module load  networkx
# module load  IPython
# module load  dill
# module load  sympy