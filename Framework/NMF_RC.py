import autograd.numpy as np
from autograd import grad
from NMF import NMF

class NMF_RC(NMF):
	
	def __init__()
	#Initialize our model and parameters
	'''
	user_components
	movie_components
	user_attn #size of matrix?
	movie_atnn #size is of movie_components
	'''

	def create_canonical_set()
	#Decide the rating matrix of canonical users and data, preferably dense.

	def generate_latents_from_canonical()
	#I'm thinking our current rowless (or columnless)  might work, or bpmf

	def get_user_latents()
	#base case - get from canonical set
	#calculate user latent for non-canonical user.
	#for ever movie he has rated, get their latents (recursively) and aggregate with associated attention.
	'''
	if user in canonical:
		return user_latent
	else:
		for every movie in user.rated_movies (rows)
			append get_movie_latents(movie) + rating to user_parameters
			movie_atnn = get_movie_atnn()
			user_latent = combine_atnn_movies(user_parameters, movie_atnn)
	'''

	def get_movie_latents()
	#base case - get from canonical set
	#calculate movie latent for non-canonical user.
	#for every user that has rated, get their latents (recursively) and aggregate with associated attention.
	'''
	if movie in canonical:
		return movie_latent
	else:
		for every user in movie.rated_users (columns)
			append get_user_latents(user) + rating to movie_parameters
			user_atnn = get_user_atnn()
			movie_latent = combine_atnn_users(movie_parameters, user_atnn)
	'''

	def get_user_atnn()
	#NN function to get the user attention from the latents + ratings

	def get_movie_atnn()
	#NN function to get the movie attention from the latents + ratings

	def combine_attn_users(movie_parameters, user_atnn)
	#NN function to combine the user attention and user latents

	def combine_attn_movies(user_parameters, movie_atnn)
	#NN function to combine the movie attention and movie latents

	def combine_user_movie(user_latent, movie_latent)
	#our final inference step - calculate rating


