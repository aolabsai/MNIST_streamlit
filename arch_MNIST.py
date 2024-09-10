import ao_arch as ar


description = "Basic MNIST"

neurons_x = 28  # Number of neurons in the x direction (global variable)
neurons_y = 28
# 1 channel, 28x28=784 neurons, each corresponding to a
# point in MNIST (downsampled to 28x28 bitmap)
arch_i = [1 for x in range(neurons_x * neurons_y)]  
# 1 channel or dimension of output, 4 neurons, corresponding to 2^4=16 binary to code for 0-9 int, the MNIST labels
arch_z = [4]
# No control neurons used here
arch_c = []
# specifies how the neurons are connected;
connector_function = "nearest_neighbour_conn"
Z2I_connections = True #wether want Z to I connection or not. If not specified, by default it's False. 
random = True #if the arch_z is not equal to size of arch_i, then pick this random function, by default it's faslse
z_ne_conn = 4 #z_in_conn -- int of random Q-input connections for Z neurons



connector_parameters = [4, 4, neurons_x, neurons_y, Z2I_connections, random, z_ne_conn] #[along axis neighbours, along diagonal neighbours, ... ]
                       

arch = ar.Arch(
    arch_i, arch_z, arch_c, connector_function, connector_parameters, description
)
