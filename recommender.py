#!/usr/bin/env python
import scipy.stats as scistats
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# This helpful method will print the jokes
def print_jokes(p, n_jokes):
    for i in range(N_clusters):
        print '\n------------------------------'
        print '     cluster ' + str(i) + '   '
        print '------------------------------'
        for j in idx[p == i][:n_jokes]:
            print jokes[j] + '\n'

data = pickle.load(open('test_data.p'))
# rows are users, columns are jokes
data_test = data['data_test']
data_train = data['data_train']

# Here are the joke texts, idx is used to index in print_jokes
jokes = pickle.load(open('jokes.pck', 'rb'))
idx = np.arange(len(jokes))

# Compute the similarity matrix. 
d_user_user = np.zeros([data_test.shape[0],data_train.shape[0]])
d_item_item = np.zeros([data_train.shape[1],data_train.shape[1]])
# d_user_user = pickle.load( open('user_user.pkl', 'rb'))
# d_item_item = pickle.load( open('item_item.pkl', 'rb'))	

#These calculations take a while, should consider saving matrices
for i in range(data_test.shape[0]):
	ri = data_test[i]
	for j in range(data_train.shape[0]):
		rj = data_train[j]
		# use elements for which both users have given ratings
		inds = np.logical_and(ri != 0, rj != 0)
		# some users gave the same rating to all jokes :(
		if np.std(ri[inds])==0 or np.std(rj[inds])==0:
			continue
		d_user_user[i,j] = scistats.pearsonr(ri[inds],rj[inds])[0]

for i in range(data_train.shape[1]):
	ri = data_train[:,i]
	d_item_item[i,i] = 1
	for j in range(i+1, data_train.shape[1]):
		rj = data_train[:,j]
		# consider only those users who have given ratings
		inds = np.logical_and(ri != 0, rj != 0)
		d_item_item[i,j] = scistats.pearsonr(ri[inds],rj[inds])[0]
		d_item_item[j,i] = d_item_item[i,j]

userfile = open('user_user.pkl', 'wb')
itemfile = open('item_item.pkl', 'wb')
pickle.dump(d_user_user, userfile)
pickle.dump(d_item_item, itemfile)

# If the rating is 0, then the user has not rated that item
# Mask to select for rated or unrated jokes
d_mask = (data_test == 0)

print data_test.shape
# print d_item_item
# print "---------" * 40

# ------------------ item item ----------------------- #
print "\n*******Item Item similarity*******"
rmse = 0
totalSquaredError = 0
totalPredicted = 0

for user in range(data_test.shape[0]):
	predictedJokes = {}
	for joke in range(data_test[0].shape[0]):
		if(d_mask[user][joke] == False):
			weightedRatingsSum = 0
			weightSum = 0
			for i in range(d_item_item.shape[1]):
				if(data_test[user][i] != 0):
					weightedRatingsSum += d_item_item[joke][i] * data_test[user][i]
					weightSum += d_item_item[joke][i]

			prediction = weightedRatingsSum / weightSum
			predictedJokes[str(joke)] = prediction
			error = data_test[user][joke] - prediction
			totalSquaredError += error ** 2
			totalPredicted += 1

	joke_rec = max(predictedJokes, key=predictedJokes.get)
	print predictedJokes
	print "Test instance "+str(user)+" , Recommend joke: "+str(joke_rec)

print totalSquaredError
print totalPredicted
mse = totalSquaredError / totalPredicted
rmse = mse ** 0.5

print "RMSE for all predictions: " + str(rmse)
# print "RMSE for all predictions: " + str(rmse)

# ------------------- user user --------------------- #
print "\n*******User User similarity*******"
rmse = 0
totalSquaredError = 0
totalPredicted = 0

for user in range(data_test.shape[0]):
	predictedJokes = {}	
	for joke in range(data_test[0].shape[0]):
		if(d_mask[user][joke] == False):
			weightedRatingsSum = 0
			weightSum = 0
			for i in range(len(data_train)):
				if(data_train[i][joke] != 0):		
					weightedRatingsSum += d_user_user[user][i] * data_train[i][joke] # correlation * rating
					weightSum += d_user_user[user][i] # sum of correlations

			prediction = weightedRatingsSum / weightSum
			predictedJokes[str(joke)] = prediction			
			error = data_test[user][joke] - prediction
			totalSquaredError += error ** 2
			totalPredicted += 1

	print predictedJokes
	joke_rec = max(predictedJokes, key = predictedJokes.get)
	print "Test instance "+str(user)+" , Recommend joke: "+str(joke_rec)

print totalSquaredError
print totalPredicted
mse = totalSquaredError / totalPredicted
rmse = mse ** 0.5

print "RMSE for all predictions: " + str(rmse)
