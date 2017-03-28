import autograd.numpy as np
movie_latent_size = 80
user_latent_size = 40
hyp_user_network_sizes = [movie_latent_size+1, 120, 40]
rating_network_sizes = [movie_latent_size+user_latent_size,120,1]
scale = .1

def build_params(col_size):
    parameters = init_random_params(scale,hyp_user_network_sizes)#Neural Net Parameters
    NET_DEPTH = len(parameters)
    parameters.append(scale * np.random.rand(movie_latent_size,col_size))#Column Latents
    parameters.append(scale *  np.ones(hyp_user_network_sizes[-1]))#Attention Latent
    parameters = list(parameters)

    return parameters, NET_DEPTH

#Credit to David Duvenaud for sleek init code
def init_random_params(scale, layer_sizes, rs=np.random.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
              scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
