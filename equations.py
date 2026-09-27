import numpy as np

def flops(image_size, batch):

    def conv_flops(cin, cout, k, out_size):
        return 2 * batch * cout * out_size**2 * cin * k**2

    total = 0

    total += conv_flops(3, 32,  7, image_size // 2)
    total += conv_flops(32, 64,  5, image_size // 4)
    total += conv_flops(64, 128, 3, image_size // 8)
    total += conv_flops(128, 256, 1, image_size // 8)
    total += conv_flops(256, 256, 3, image_size // 16)
    total += conv_flops(256, 512, 1, image_size // 16)

    total += 2 * batch * 512 * 256
    total += 2 * batch * 256 * 100

    return total

def memory(image_size, batch):

    def activation_mem(cout, out_size):
            return batch * cout * out_size**2
    
    total = 0

    total += activation_mem(3, image_size)
    total += activation_mem(32, image_size // 4)
    total += activation_mem(32, image_size // 2)
    total += activation_mem(64, image_size // 4)
    total += activation_mem(128, image_size // 8)
    total += activation_mem(256, image_size // 8)
    total += activation_mem(256, image_size // 16)
    total += activation_mem(512, image_size // 16)
    total += batch * (512 + 256 + 100)

    total += 3*7**2*32 + 32*5**2*64 + 64*3**2*128 + 128*1**2*256 + 256*3**2*256 + 256*1**2*512
    total +=  512*256 + 256*100 + 256 + 100

    return total * 4

def bytes_moved(image_size, batch):
    def conv_bytes(cin, cout, k, in_size, out_size):
         return 4 * (batch * cin * in_size**2 + cout * cin * k**2 + batch * cout * out_size**2)
    
    def linear_bytes(cin, cout):
         return 4 * (batch * cin + cin * cout + cout + batch * cout)

    def relu(n):
         return 4 * batch * (n + n)
    
    def op_bytes(n_in, n_out):
        return 4 * batch * (n_in + n_out)

    total = 0
    total += conv_bytes(3, 32, 7, image_size, image_size // 2)
    total += relu(32 * (image_size // 2)**2)
    total += op_bytes(32 * (image_size // 2)**2, 32 * (image_size // 4)**2)
    total += conv_bytes(32, 64, 5, image_size // 4, image_size // 4)
    total += relu(64 * (image_size // 4)**2)
    total += conv_bytes(64, 128, 3, image_size // 4, image_size // 8)
    total += relu(128 * (image_size // 8)**2)
    total += conv_bytes(128, 256, 1, image_size // 8, image_size // 8)
    total += relu(256 * (image_size // 8)**2)
    total += conv_bytes(256, 256, 3, image_size // 8, image_size // 16)
    total += relu(256 * (image_size // 16)**2)
    total += conv_bytes(256, 512, 1, image_size // 16, image_size // 16)
    total += relu(512 * (image_size // 16)**2)
    total += op_bytes(512 * (image_size // 16)**2, 512)
    total += linear_bytes(512, 256)
    total += relu(256)
    total += linear_bytes(256, 100)

    return total

def latency(image_size, batch, theta):
    launch = theta["launch"]
    compute = flops(image_size, batch) / theta["compute"]
    memory = bytes_moved(image_size, batch) / theta["bandwidth"]

    return launch + np.maximum(compute, memory)

def energy(image_size, batch, theta_energy):
    energy = theta_energy["base_power"] * latency(image_size, batch, theta_energy["latency"])
    energy += theta_energy["joule_per_gflop"] * flops(image_size, batch) / 1e9
    energy += theta_energy["joule_per_gbyte"] * bytes_moved(image_size, batch) / 1e9
    
    return energy