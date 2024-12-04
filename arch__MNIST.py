import ao_arch as ar


description = "Basic MNIST"
# 1 channel, 28x28=784 neurons, each corresponding to a
# point in MNIST (downsampled to 28x28 bitmap)
arch_i_bw = [28 * 28]
arch_i_gr = [8 for i in range(28 * 28)]
# 1 channel or dimension of output, 4 neurons, corresponding to 2^4=16 binary to code for 0-9 int, the MNIST labels
arch_z = [4]
# No control neurons used here
arch_c = []
# specifies how the neurons are connected;
# in this case, all neurons are connected randomly to a set number of others
connector_function = "rand_conn"
# used 360, 180 before to good success
connector_parameters = [392, 261, 784, 4]
arch_bw = ar.Arch(
    arch_i_bw, arch_z, arch_c, connector_function, connector_parameters, description
)
arch_gr = ar.Arch(
    arch_i_gr, arch_z, arch_c, connector_function, connector_parameters, description
)
