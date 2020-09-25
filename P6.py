
import cv2
import math
import numpy as np
from scipy import ndimage

class Transform:

	class Affine:

		xIdx = 0
		yIdx = 1

		def __init__(self,arr):

			assert np.shape(arr) == (2,3)
			self.arr = np.append(arr,[0,0,1]).reshape(3,3)

		def getCoord(self,x,y):
			xy = np.array([[x],[y],[1]])
			result = np.matmul(self.arr,xy)
			return result.astype('int64')

		def inBoundsPredicate(self,numRows,NumCols,x,y):
			return 0 <= self.getCoord(x,y)[self.xIdx]  and self.getCoord(x,y)[self.xIdx] <= NumCols -1 \
		and 0 <= self.getCoord(x,y)[self.yIdx]  and self.getCoord(x,y)[self.yIdx] <= numRows -1


if __name__ == '__main__':
	fpath = 'boat.png'

	I = cv2.imread(fpath, 0)
	t = Transform.Affine(np.array([[.9, -.1, 100],[.1, .9, 0]]))

	out = np.zeros(shape=np.shape(I))

	rows = len(I)
	cols = len(I[0])

	y=0
	for row in I:
		x=0
		for intensity in row:
			if t.inBoundsPredicate(rows,cols,x,y):
				txy = t.getCoord(x,y).flatten()
				tx = txy[t.xIdx]
				ty = txy[t.yIdx]
				out[ty][tx] = intensity
			x = x+1
		y = y+1

	out = out.astype('uint8')
	print(I)
	print(out)

	cv2.imshow("original", I)
	cv2.imshow("transformed", out)
	cv2.waitKey(0)
