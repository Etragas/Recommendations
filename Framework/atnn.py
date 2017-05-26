import utils
from DataLoader import *
from train import *
from autograd import core
print core.__file__
# Load the data using DataLoader
netflix_full_data = DataLoader().LoadData(file_path="../Data/download/user_first.txt", data_type=DataLoader.NETFLIX, size= (490000,18000))
#full_data = DataLoader().LoadData(file_path="../Data/ml-10m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (72000,11000))
#netflix_full_data= DataLoader().LoadData(file_path="../Data/ml-1m/ratingsbetter.dat", data_type=DataLoader.NETFLIX, size= (6100,4000))
full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.MOVIELENS, size= (1200,2000))
#netflix_full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.NETFLIX, size= (1200,2000))
# Reduce the matrix to toy size
#full_data = full_data[:100,:100]

#Determine number of latents for movie/user
print full_data.shape
nrows, ncols = full_data.shape

utils.num_user_latents = int(np.ceil(40))
utils.num_movie_latents = int(np.ceil(20))

print utils.num_user_latents, utils.num_movie_latents
can_idx = get_canonical_indices(full_data, [utils.num_user_latents, utils.num_movie_latents])
can_idx_netflix = get_canonical_indices_from_list(netflix_full_data, [utils.num_user_latents, utils.num_movie_latents])

# for x in range(len(can_idx_netflix[0])):
#     if can_idx[0][x] == can_idx_netflix[0][x]:
#         print "all_good"
#         continue
#     else:
#         print x
#         print "What"
#
#     raw_input()
#Resort data so that canonical users and movies are in top left
full_data = full_data[:,can_idx[1]]
full_data = full_data[can_idx[0],:]
netflix_full_data = index_sort(netflix_full_data,can_idx_netflix)

for x in range(11):
    print len(netflix_full_data[keys_row_first][x][get_items])
    print x, netflix_full_data[keys_row_first][x][get_items]
print netflix_full_data[keys_col_first][0][get_items]
print netflix_full_data[keys_row_first][0][get_items]

for x in range(10):
    print len(netflix_full_data[keys_col_first][x][get_items])

train_data, test_data = splitData(full_data)
net_train, net_test = splitDataList(netflix_full_data,.8)
train_idx = test_idx = np.array([np.array(range(nrows)),np.array(range(ncols))])

# Training Parameters
step_size = 0.0001
num_users_per_batch = 20
batches_per_epoch = int(np.ceil(float(nrows) / num_users_per_batch))
batches_per_can_epoch = int(np.ceil(float(utils.num_user_latents)/ num_users_per_batch))

num_epochs = 40
hyperp1 = [step_size*10, 40, batches_per_can_epoch]
hyperp2 = [step_size*10, 40 , batches_per_can_epoch]
hypert = [step_size, num_epochs, batches_per_epoch]

# Build the dictionary of parameters for the nets, etc.
parameters = build_params()

# Train the parameters.  Pretraining the nets and canon latents are optional.
parameters = train(net_train, net_test, can_idx, train_idx, test_idx, parameters,
                   p1=False, p1Args=hyperp1, p2=False, p2Args=hyperp2, trainArgs=hypert, use_cache=False)
