import utils
from DataLoader import *
from train import *

# Load the data using DataLoader
full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.MOVIELENS)
full_data = full_data[:100,:100]
#full_data = full_data[:200,:200]
#full_data = np.random.randint(0,5,full_data.shape,dtype = 'int')
print full_data
print full_data.shape
print np.mean(np.sum(full_data > 0,axis = 1))
# Our dataset only has 1000 users and 1700 movies

# Reduce the matrix to toy size
nrows, ncols = full_data.shape
utils.num_user_latents = int(.1 * nrows)
utils.num_movie_latents = int(.1 * ncols)

# [Model Parameters
# Initialize our train matrix with given size
train_data, test_data = splitData(full_data)
#train_idx, test_idx = splitData(full_data)
can_idx = get_canonical_indices(train_data, [utils.num_user_latents, utils.num_movie_latents])
train_idx = test_idx = np.array([np.array(range(nrows)),np.array(range(ncols))])
train_user_size, train_movie_size = map(lambda x: x.size, train_idx)

# Training Parameters
step_size = 0.005
num_users_per_batch = 100
batches_per_epoch = nrows / num_users_per_batch
num_epochs = 40
hyperp = [step_size, num_epochs, 1]
hypert = [step_size / 2, num_epochs, batches_per_epoch]

# Build the dictionary of parameters for the nets, etc.
parameters = build_params([train_user_size + num_user_latents, train_movie_size + num_movie_latents])

# Train the parameters.  Pretraining the nets and canon latents are optional.
parameters = train(train_data, test_data, can_idx, train_idx, test_idx, parameters,
                   p1=True, p1Args=hyperp, p2=True, p2Args=hyperp, trainArgs=hypert)
