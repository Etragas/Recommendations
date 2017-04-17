import utils
from DataLoader import *
from train import *

# Load the data using DataLoader
full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.MOVIELENS)
full_data = full_data[:100,:100]
print full_data[0,:].sum()
print np.mean(np.sum(full_data > 0,axis = 1))
# Our dataset only has 1000 users and 1700 movies

# Reduce the matrix to toy size
nrows, ncols = full_data.shape
utils.num_user_latents = int(.1 * nrows)
utils.num_movie_latents = int(.1 * ncols)

# [Model Parameters
# Initialize our train matrix with given size
can_idx = get_canonical_indices(full_data, [utils.num_user_latents, utils.num_movie_latents])
full_data = full_data[:,can_idx[1]]
full_data = full_data[can_idx[0],:]
train_data, test_data = splitData(full_data)
print "can idx, " ,can_idx
print train_data[9,:]
print test_data[9,:]
train_idx = test_idx = np.array([np.array(range(nrows)),np.array(range(ncols))])

# Training Parameters
step_size = 0.0005
num_users_per_batch = 30
batches_per_epoch = int(np.ceil(float(nrows) / num_users_per_batch))
batches_per_can_epoch = int(np.ceil(float(utils.num_user_latents)/ num_users_per_batch))

num_epochs = 40
hyperp1 = [step_size*10, 30, batches_per_can_epoch]
hyperp2 = [step_size*10, 10, batches_per_can_epoch]
hypert = [step_size, num_epochs, batches_per_epoch]

# Build the dictionary of parameters for the nets, etc.
parameters = build_params()

# Train the parameters.  Pretraining the nets and canon latents are optional.
parameters = train(train_data, test_data, can_idx, train_idx, test_idx, parameters,
                   p1=True, p1Args=hyperp1, p2=True, p2Args=hyperp2, trainArgs=hypert)
