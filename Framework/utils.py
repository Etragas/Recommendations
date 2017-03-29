import autograd.numpy as np
movie_latent_size = 40
user_latent_size = 40
hyp_user_network_sizes = [movie_latent_size+1, 100, user_latent_size]
hyp_movie_network_sizes = [user_latent_size+1,100,movie_latent_size]
rating_network_sizes = [movie_latent_size+user_latent_size,10,1]
scale = .1

def build_params((row_size,col_size)):
    parameters = {}
    parameters[keys_movie_to_user_net]=(init_random_params(scale,hyp_user_network_sizes))#Neural Net Parameters
    parameters[keys_user_to_movie_net]=(init_random_params(scale,hyp_movie_network_sizes))#Neural Net Parameters
    parameters[keys_rating_net]=(init_random_params(scale, rating_network_sizes))#Neural Net Parameters
    parameters[keys_col_latents]=(scale * np.random.rand(movie_latent_size,50))#Column Latents
    parameters[keys_row_latents]=(scale * np.random.rand(50,user_latent_size))#user_latent_size,row_size))#Row Latents

    return parameters


#Credit to David Duvenaud for sleek init code
def init_random_params(scale, layer_sizes, rs=np.random.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
              scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

keys_movie_to_user_net = "MovieToUser"
keys_user_to_movie_net = "UserToMovie"
keys_row_latents = "RowLatents"
keys_col_latents = "ColLatents"
keys_rating_net = "PredNet"