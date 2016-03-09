from NeuralNetwork import Neural_Network,trainer
import numpy as np
import csv
from numpy.lib.npyio import genfromtxt
import os
import pandas as pd
#season_01 = genfromtxt('season 01.csv', skip_header=1, dtype = float, delimiter=',')
#season_2 = genfromtxt('season 2.csv', skip_header=1, dtype = float, delimiter=',')

def vectorOfStats(tempRow, predictors):
#Get the correct predictors for the stats and return a vector of them
	vector = []
	for p in predictors:
		vector.append(int(tempRow[p]))
	return vector

'''The main function.  This will loop through the players within
the player csv.  It will pull necessary training data from the all-seasons csv.
It will add them correctly to the x and y train/test sets, and then run 
the neural network and output the predicted stat.'''
def predictStat(playerStats, allStats, stat1, predictors):
	testVector = []
	copyTest = []
	seasonNum = len(playerStats.index)-1
	myPos = playerStats.iloc[[seasonNum]]['Pos'].to_string()[-1]
	if(myPos == 'F'):
		pos=0 
	elif(myPos == 'C'):
		pos=1
	else:
		pos=2

	#---------Handling the player's stats-----------#
	for i in range(seasonNum):
		tempRow0 = playerStats.iloc[[i]]
		testVector.append(vectorOfStats(tempRow0, predictors))
		copyTest.append([int(tempRow0[stat1])])

	testVector = np.array(testVector)
	testVector = testVector.flatten()
	numInputs = (seasonNum)*len(predictors)

	#How many previous seasons to use for training/testing data
	if(seasonNum > 16):
		copyTest = copyTest[len(copyTest)-2:]
	elif(seasonNum > 8):
		copyTest = copyTest[len(copyTest)-3:]
	elif(seasonNum > 5):
		copyTest = copyTest[len(copyTest)-5:]

	if(seasonNum > 16):
		numInputs = len(predictors)*3
		testVector = testVector[(len(testVector)-numInputs):]
	elif(seasonNum > 9):
		numInputs = len(predictors)*6
		testVector = testVector[(len(testVector)-numInputs):]

	#---------End Handling the player's stats-----------#	


	trainingX = []
	trainingY = []


	#---------Getting stats from all players-----------#
	numPlayers = len(allStats.index)	
	prevId = "null"
	tempX = []
	samePlayer = True
	prevSeason = 0

	# Loop through all players
	for i in range(numPlayers):
	    tempRow0 = allStats.iloc[[i]]
	    myId = tempRow0["ilkid"].to_string()[8:]
	    mySeason = int(tempRow0["seasonNum"])

	    # Use position unless entering 13th season or greater
	    if(int(tempRow0["position"]) == pos or seasonNum > 12):
	    	# Is this the same player as the last row
		    if(prevId != myId):
		    	tempX = []
		    	samePlayer = False
		    else:
		    	samePlayer = True


		    # Include if this season is less than current season for player
		    if(mySeason < seasonNum):
		    	# If is a new player
		    	if(samePlayer == False):
		    		# Add 0s to first season if they missed it
		    		if(mySeason != 0):
		    			if(mySeason-1 == 0):
			    			tempX.append([0]*len(predictors))
			    			tempX.append(vectorOfStats(tempRow0, predictors))
			    	elif(mySeason == 0):
			    		tempX.append(vectorOfStats(tempRow0, predictors))
			    	else:
			    		tempX = []

			    # If not a new player
		    	else:
		    		if(mySeason-1 != prevSeason):
		    			# Add 0s to missing seasons
		    			for x in range(mySeason-1-prevSeason):
		    				tempX.append([0]*len(predictors))
		    		tempX.append(vectorOfStats(tempRow0, predictors))


		    # If we have reached last season needed
		    elif(mySeason == seasonNum):
		    	
		    	tempX.append(vectorOfStats(tempRow0, predictors))
		    	tempX = np.array(tempX)
		    	tempX = tempX.flatten()
		    	# If we have a vector of proper size
		    	if(len(tempX) == seasonNum*len(predictors)):
		    		tempX = tempX.tolist()
		    		if(seasonNum > 16):
		    			tempX = tempX[(len(tempX)-numInputs):]
		    		elif(seasonNum > 9):
		    			tempX = tempX[(len(tempX)-numInputs):]

		    		# Appending to the trainingX and trainingY
			    	trainingX.append(tempX)
			    	trainingY.append([int(tempRow0[stat1])])
		    	tempX = []
		    # Otherwise clear our vector
		    else:
		    	tempX = []


	    prevId = myId
	    prevSeason = mySeason
	#---------End of getting stats from all players-----------#




	#Training Data:
	trainingX = np.asarray(trainingX)
	trainingY = np.asarray(trainingY)

	#Normalize:
	xMax = np.amax(trainingX, axis=0)
	yMax = np.amax(trainingY, axis=0)
	copyTest = np.array(copyTest)
	playerMax = np.amax(copyTest, axis=0)

	# statAdjustDict1 = {'G':0, 'MP':4, 'PTS':5, 'ORB':.8, 'DRB':1.2, 'AST':1.6, 'STL':.3, 'BLK':.5, 'TOV':.25,'3P':.8}
	# statAdjustDict2 = {'G':0, 'MP':2, 'PTS':2.5, 'ORB':.4, 'DRB':.8, 'AST':.8, 'STL':.15, 'BLK':.25, 'TOV':.15,'3P':.4}

	def improvementFunction(num):
	#Function to aid in improvement of younger players
		return (seasonNum-7)*(-num/5)

	statAdjustDict = {
						'G':0,
						'MP':improvementFunction(300), 
						'PTS':improvementFunction(400), 
						'ORB':improvementFunction(35),
						'DRB':improvementFunction(80), 
						'AST':improvementFunction(65), 
						'STL':improvementFunction(30), 
						'BLK':improvementFunction(30), 
						'TOV':improvementFunction(20),
						'3P':improvementFunction(40)}

	if(seasonNum < 8):
		playerMax[0] += statAdjustDict[stat1]


	trainingX = trainingX/xMax
	trainingY = trainingY/yMax

	NN = Neural_Network(Lambda = 0.001, inputSize = numInputs, hiddenSize = int((numInputs)/1.5), outputSize = 1)
	T = trainer(NN)
	T.train(trainingX, trainingY, trainingX, trainingY)

	result = NN.propogation(testVector)
	adjustedResult = (result*playerMax)
	returnStat = round(adjustedResult[0], 1)


	return returnStat

	

directory = 'player_stats/'
csvFile = 'all-seasons.csv'
allStats = pd.read_csv(csvFile)

for f in sorted(os.listdir(directory)):
#Run the predictStat method on all players within directory
#Need the csv file containing the player's previous stats
	playerCSV = directory+f
	playerStats = pd.read_csv(playerCSV)
	firstName = f[:f.index('_')]
	lastName = f[f.index('_')+1:f.index('.')]
	playerName = (firstName + ' ' + lastName)
	print(playerName + ' starting.')


	#These are the stats we want to predict
	stats = ['G', 'MP', 'PTS', '3P', 'ORB', 'DRB', 'AST', 'TOV', 'STL', 'BLK']

	#These are the categories used to predict the corresponding index in 'stats'
	predictors = [['G', 'MP', 'PTS'],
				['MP', 'G', 'PTS'],
				['PTS', 'G', '3P'],
				[ '3P', 'G', 'PTS'],
				['ORB', 'G', 'DRB', 'BLK'],
				['DRB', 'G', 'ORB', 'BLK'],
				['AST', 'G', 'TOV', 'STL'],
				['TOV', 'G', 'AST', 'STL'],
				['STL', 'G', 'AST', 'TOV'],
				['BLK', 'G', 'ORB', 'DRB'],
				]
	finalStats = []

	#Predict for each stat within 'stats' list
	for s in range(len(stats)):
		statPrediction = predictStat(playerStats, allStats, stats[s], predictors[s])
		finalStats.append(statPrediction)	
		print(stats[s] + ': ' + str(statPrediction))




	#Used for outputting/printing the predicitons
	gamesPlayed = finalStats[0]
	index = [playerName]
	d = {'G' : pd.Series([round(gamesPlayed, 0)], index=index),
		'MP' : pd.Series([round(finalStats[1]/gamesPlayed,1)], index=index),
		'PTS' : pd.Series([round(finalStats[2]/gamesPlayed,1)], index=index),
		'3P' : pd.Series([round(finalStats[3]/gamesPlayed,1)], index=index),
		'ORB' : pd.Series([round(finalStats[4]/gamesPlayed,1)], index=index),
		'DRB' : pd.Series([round(finalStats[5]/gamesPlayed,1)], index=index),
		'AST' : pd.Series([round(finalStats[6]/gamesPlayed,1)], index=index),
		'TOV' : pd.Series([round(finalStats[7]/gamesPlayed,1)], index=index),
		'STL' : pd.Series([round(finalStats[8]/gamesPlayed,1)], index=index),
		'BLK' : pd.Series([round(finalStats[9]/gamesPlayed,1)], index=index)
		}
	
	#Output prediction to a csv file
	df = pd.DataFrame(d, columns=['G', 'MP', 'PTS', '3P', 'ORB', 'DRB', 'AST', 'TOV', 'STL', 'BLK'])
	with open('output2.csv', 'a') as out:
		df.to_csv(out, header=False)
	print(playerName + ' stats predicted.')
