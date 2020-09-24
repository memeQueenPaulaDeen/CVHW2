import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def g(x,sigma):
	return float(1/(sigma* math.sqrt(2*math.pi)*math.exp(-(x**2)/(2*sigma**2))))



if __name__ == '__main__':
	df = pd.DataFrame()

	x = [i/100 for i in range(-30,30)]
	#s = [(.5+i/40) for i in range(40)]
	s = [i/5+.5 for i in range(-30,30)]
	# domain = []
	# for x in range (10,50):
	#
	# 	for s in range(10):
	#
	# 		domain.append([x/10,(.5+s/10)])

	# df['d'] = domain
	# df['y'] = df.apply(lambda x: g(x.d[0], x.d[1]))
	# df['x'] = df.apply(lambda x: x.d[0])
	# df['s'] = df.apply(lambda x: x.d[1])



	df['x'] = x
	df['s'] = s
	#df['s'] = -df['x']


	df['yx'] = df.apply(lambda x: g(x.x,1),axis=1)
	df['dyx'] = df['yx'] - df['yx'].shift()
	df['ddyx'] = df['dyx'] - df['dyx'].shift()
	#df['ddyx'] = df.apply(lambda x: x.ddyx/x.s, axis=1)

	df['dys'] = 0
	for ss in s:
		#print(ss)
		df['ys'] = df.apply(lambda x: g(x.x,ss),axis=1)
		df['dys'] = df['ys'] - df['dys']
	#df['dys'] = df.apply(lambda x: x.dys / x.s, axis=1)


	plt.plot(x,df['ddyx'])
	plt.show()

	plt.plot(s,df['dys'])
	plt.show()

