import utils
from DataLoader import *
from train import *

# Load the data using DataLoader
#full_data = DataLoader().LoadData(file_path="../Data/ml-1m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS)
full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.MOVIELENS)
print full_data.shape
print np.mean(np.sum(full_data > 0,axis = 1)) #Check average ratings per user

# Reduce the matrix to toy size
full_data = full_data[:100,:100]
rows = [x for x in range((full_data.shape[0])) if full_data[x,:].sum() > 0]
cols = [x for x in range((full_data.shape[1])) if full_data[:,x].sum() > 0]
full_data = full_data[rows,:]
full_data = full_data[:,cols]

#Determine number of latents for movie/user
print full_data.shape
nrows, ncols = full_data.shape

utils.num_user_latents = int(np.ceil(.1*nrows))
utils.num_movie_latents = int(np.ceil(.1*ncols))
print utils.num_user_latents, utils.num_movie_latents
can_idx = get_canonical_indices(full_data, [utils.num_user_latents, utils.num_movie_latents])

#Resort data so that canonical users and movies are in top left
full_data = full_data[:,can_idx[1]]
full_data = full_data[can_idx[0],:]

train_data, test_data = splitData(full_data)
train_idx = test_idx = np.array([np.array(range(nrows)),np.array(range(ncols))])

# Training Parameters
step_size = 0.0001
num_users_per_batch = 10
batches_per_epoch = int(np.ceil(float(nrows) / num_users_per_batch))
batches_per_can_epoch = int(np.ceil(float(utils.num_user_latents)/ num_users_per_batch))

num_epochs = 40
hyperp1 = [step_size*10, 20, batches_per_can_epoch]
hyperp2 = [step_size*10, 20, batches_per_can_epoch]
hypert = [step_size, num_epochs, batches_per_epoch]

# Build the dictionary of parameters for the nets, etc.
parameters = build_params()

# Train the parameters.  Pretraining the nets and canon latents are optional.
parameters = train(train_data, test_data, can_idx, train_idx, test_idx, parameters,
                   p1=True, p1Args=hyperp1, p2=True, p2Args=hyperp2, trainArgs=hypert, use_cache=True)
