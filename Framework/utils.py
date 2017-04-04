import autograd.numpy as np
num_movie_latents = 10
movie_latent_size = 40
num_user_latents = 20
user_latent_size = 40
hyp_user_network_sizes = [movie_latent_size+1, 100, user_latent_size]
hyp_movie_network_sizes = [user_latent_size+1,100,movie_latent_size]
rating_network_sizes = [movie_latent_size+user_latent_size,50,1]
scale = .1

def build_params((row_size,col_size)):
    parameters = {}
    parameters[keys_movie_to_user_net]=(init_random_params(scale,hyp_user_network_sizes))#Neural Net Parameters
    parameters[keys_user_to_movie_net]=(init_random_params(scale,hyp_movie_network_sizes))#Neural Net Parameters
    parameters[keys_rating_net]=(init_random_params(scale, rating_network_sizes))#Neural Net Parameters
    parameters[keys_col_latents]=(scale * np.random.rand(movie_latent_size,num_movie_latents))#Column Latents
    parameters[keys_row_latents]=(scale * np.random.rand(num_user_latents,user_latent_size))#user_latent_size,row_size))#Row Latents

    return parameters


#Credit to David Duvenaud for sleek init code
def init_random_params(scale, layer_sizes, rs=np.random.RandomState(0)):
    """Build a list of (weights, biases) tuples,
       one for each layer in the net."""
    return [(scale * rs.randn(m, n),   # weight matrix
              scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def get_canonical_indices(data,latent_sizes):
    indicators = data > 0
    user_rating_counts = indicators.sum(axis = 1) #Bug one found
    movie_rating_counts = indicators.sum(axis = 0) #Bug one found
    user_indices = get_top_n(user_rating_counts,latent_sizes[0])
    movie_indices = get_top_n(movie_rating_counts,latent_sizes[1])
    return user_indices, movie_indices

def fill_in_gaps(canonical_indices,new_indices,full_data):
    can_sizes = map(lambda x: x.size,canonical_indices)
    #Sort the indices before.
    new_axis_sizes = tuple([canonical_indices[x].size+new_indices[x].size for x in range(len(canonical_indices))])
    print new_axis_sizes
    training_block = np.zeros(new_axis_sizes)
    #To load data properly, cross product every axis with every other of new indices.
    #Fill in gaps along diagonal
    training_block[:can_sizes[0],:can_sizes[1]]= full_data[np.ix_(*canonical_indices)]
    training_block[can_sizes[0]:,can_sizes[1]:] = full_data[np.ix_(*new_indices)]
    print training_block
    #Fill in gaps in between.
    training_block[:can_sizes[0],can_sizes[1]:] = full_data[np.ix_(canonical_indices[0],new_indices[1])]
    training_block[can_sizes[0]:,:can_sizes[1]] = full_data[np.ix_(new_indices[0],canonical_indices[1])]
    return training_block

def get_top_n(data,n):
    indices = np.argpartition(-data, n)[:n]
    print indices
    return indices


def splitData(data):
    row, col = data.shape
    row_indices, col_indices = [np.random.choice(range(row),row),np.random.choice(range(col),col)]
    print row_indices
    print row,col
    row_split, col_split = int(.8*row), int(.8*col)
    return [row_indices[:row_split],col_indices[:col_split]],[row_indices[row_split:],col_indices[col_split:]]

keys_movie_to_user_net = "MovieToUser"
keys_user_to_movie_net = "UserToMovie"
keys_row_latents = "RowLatents"
keys_col_latents = "ColLatents"
keys_rating_net = "PredNet"
