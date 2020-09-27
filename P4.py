import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def g(x,sigma):
	return float(1/(sigma* math.sqrt(2*math.pi))*math.exp(-(x**2)/(2*sigma**2)))



if __name__ == '__main__':
	df = pd.DataFrame(columns=['s','x','y'])

	x = [i/10 for i in range(-30,30)]
	#s = [(.5+i/40) for i in range(40)]
	s = [1, 1.1]
	for sigma in s:
		for val in x:
			#df.append(pd.DataFrame.from_dict({'x':val,'s':sigma,'y':g(val,sigma)},orient='columns'))
			df = df.append({'x':val,'s':sigma,'y':g(val,sigma)},ignore_index=True)


	der = np.array(df.loc[df['s'] == 1]['y'].to_list()) - np.array( df.loc[df['s'] == 1.1]['y'].to_list())


	df['yx'] = df.apply(lambda x: g(x.x,1),axis=1)
	df['dyx'] = df['yx'] - df['yx'].shift()
	df['ddyx'] = df['dyx'] - df['dyx'].shift()
	#df['ddyx'] = df.apply(lambda x: x.ddyx/x.s, axis=1)




	plt.plot(x,df.loc[df['s']==1]['ddyx'])

	plt.show()

	plt.plot(x,der)
	plt.show()

