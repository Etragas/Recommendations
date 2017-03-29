import autograd.numpy as np
movie_latent_size = 40
user_latent_size = 40
hyp_user_network_sizes = [movie_latent_size+1, 100, user_latent_size]
hyp_movie_network_sizes = [user_latent_size+1,100,movie_latent_size]
rating_network_sizes = [movie_latent_size+user_latent_size,10,1]
scale = .1

def build_params((row_size,col_size)):
    parameters = []
    parameters=(init_random_params(scale,hyp_user_network_sizes))#Neural Net Parameters
    parameters+=(init_random_params(scale,hyp_movie_network_sizes))#Neural Net Parameters
    l1_size = len(parameters)
    parameters+=(init_random_params(scale,rating_network_sizes))#Neural Net Parameters
    l2_size = len(parameters)
    parameters.append(scale * np.random.rand(movie_latent_size,50))#Column Latents
    parameters.append(scale * np.random.rand(50,user_latent_size))#user_latent_size,row_size))#Column Latents
    parameters = list(parameters)

    return parameters, l1_size, l2_size


#Credit to David Duvenaud for sleek init code
def init_random_params(scale, layer_sizes, rs=np.random.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
              scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
