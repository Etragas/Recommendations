import utils
from DataLoader import *
from train import *
from GenGraph import draw_stuff
from autograd import core
print core.__file__
# Load the data using DataLoader
#full_data = DataLoader().LoadData(file_path="../Data/ml-10m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (72000,11000))
#full_data = DataLoader().LoadData(file_path="../Data/ml-1m/ratingsbetter.dat", data_type=DataLoader.MOVIELENS, size= (6100,4000))
full_data = DataLoader().LoadData(file_path="../Data/ml-100k/u.data", data_type=DataLoader.MOVIELENS, size= (1200,2000))
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

utils.num_user_latents = int(np.ceil(10))
utils.num_movie_latents = int(np.ceil(10))

print utils.num_user_latents, utils.num_movie_latents
can_idx = get_canonical_indices(full_data, [utils.num_user_latents, utils.num_movie_latents])
#Resort data so that canonical users and movies are in top left
full_data = full_data[:,can_idx[1]]
full_data = full_data[can_idx[0],:]

train_data, test_data = splitData(full_data)
train_idx = test_idx = np.array([np.array(range(nrows)),np.array(range(ncols))])

# Training Parameters
step_size = 0.0001
num_users_per_batch = 5
batches_per_epoch = int(np.ceil(float(nrows) / num_users_per_batch))
batches_per_can_epoch = int(np.ceil(float(utils.num_user_latents)/ num_users_per_batch))

num_epochs = 40
hyperp1 = [step_size*10, 1, batches_per_can_epoch]
hyperp2 = [step_size*10, 1, batches_per_can_epoch]
hypert = [step_size, num_epochs, batches_per_epoch]

# Build the dictionary of parameters for the nets, etc.
parameters = build_params()
print can_idx
with open("../Data/ml-100k/u.item") as f:
    names = [x.split('|')[1] for x in f.readlines()]

names = [names[x] for x in can_idx[1]]
unames = ["user #" + str(i) for i in range(88)]
names = [unames,names]
print names

call_train = {13:[]}
#train_data[15,9]=0
ltrain = listify(train_data)
setup_caches(ltrain)
getUserLatent(parameters,ltrain,13,call_train = call_train)
draw_stuff(names,call_train)
global TRAININGMODE
TRAININGMODE = True
print_train(0,0,names,*[call_train])
raw_input()
# Train the parameters.  Pretraining the nets and canon latents are optional.
parameters = train(train_data, test_data, can_idx, train_idx, test_idx, parameters,
                   p1=True, p1Args=hyperp1, p2=True, p2Args=hyperp2, trainArgs=hypert, use_cache=True)
