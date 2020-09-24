
###Adapted from homework #1


import cv2
import math
import numpy as np
from scipy import ndimage

class Kernal:

	class Gausinan:

		def __init__(self,N,sigma):
			self.N = N
			self.sigma = sigma
			self.frontBit = 1 / (sigma ** 2 * 2 * math.pi)
			self.k = np.zeros((N, N))
			self.idxMin = int(N / 2)
			for i in range(N):
				for j in range(N):
					x, y = self.getXYfromIDX(i, j)
					self.k[i, j] = self.frontBit * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))



		def getXYfromIDX(self,i,j):
			x = i-self.idxMin
			y = j-self.idxMin
			return x, y

	class Sobel:

		Gx = None
		Gy = None

		def __init__(self):
			self.N = 3
			self.x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])*1/4
			self.y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])*1/4
			self.idxMin = int(self.N / 2)

		def getGXfromIM(self,image):
			image.setPadding(self.idxMin)
			self.Gx = image.convolve(self,self.x)
			#self.Gx = ndimage.convolve(image.image, self.x, mode='constant', cval=0.0)

		def getGYfromIM(self,image):
			image.setPadding(self.idxMin)
			self.Gy = image.convolve(self,self.y)

		def getGradMag(self,Ix,Iy,scale=255):
			print(np.shape(Ix))
			print(np.shape(Iy))
			assert np.shape(Ix) == np.shape(Iy)
			grad = np.zeros(np.shape(Ix))
			j = 0
			for row in Ix:
				i = 0
				for x in row:
					grad[j][i] = math.sqrt(x**2+Iy[j][i]**2)
					i = i + 1
				j = j + 1

			#simple scale
			if scale != 0:
				grad = grad.astype('float64')
				#grad  *= 255.0/grad.max()
				grad = (255 * (grad - np.min(grad)) / np.ptp(grad))#addapted from https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
				grad = grad.astype('uint8')

			return grad




class Image:

	paddingOffset = None

	def __init__(self,fpath):
		self.image = cv2.imread(fpath, 0) #np.array([[1,2,3],[4,5,6],[7,8,9]])#
		self.height = len(self.image)
		self.width = len(self.image[0])

	def setPadding(self,width):

		#from numpy docs @ https://numpy.org/doc/stable/reference/generated/numpy.pad.html
		def pad_with(vector, pad_width, iaxis, kwargs):
			pad_value = kwargs.get('padder', 10)
			vector[:pad_width[0]] = pad_value
			vector[-pad_width[1]:] = pad_value

		padded = np.pad(self.image, width, pad_with, padder=0)
		self.paddingOffset = width
		self.padded = padded




	def convolve(self,k,m):
		#m is kernal matrix
		#thankfully kernal is symetric so no flipy bois
		for r in range(self.height):
			#print(r)
			for c in range(self.width):
				cellVal= 0
				for y in range (k.N):
					for x in range(k.N):
						cellVal += self.padded[r+x,c+y] * m[x,y]
				self.image[r,c] = cellVal
		# print("plz")
		# print(self.image)
		return self.image



if __name__ == '__main__':
	fpath = 'boat.png'

	Iog = Image(fpath)
	can = cv2.Canny(Iog.image, 0, 1,3)

	Ix = Image(fpath)
	print(np.max(Ix.image))
	Iy = Image(fpath)
	print(np.max(Iy.image))
	kern = Kernal.Sobel()
	kern.getGXfromIM(Ix)
	kern.getGYfromIM(Iy)
	grad = kern.getGradMag(kern.Gx, kern.Gy)
	print(np.max(grad))



	#sobelx = cv2.Sobel(Iog.image, cv2.CV_64F, 1, 0, ksize=3)
	#sobely = cv2.Sobel(Iog.image, cv2.CV_64F, 0, 1, ksize=3)



	cv2.imwrite('outputP5sobel.png',grad)
	cv2.imwrite('outputP6canny.png',can)

	cv2.imshow("preconvolve", Iog.image)
	cv2.imshow("postconvolveSx", kern.Gx)
	cv2.imshow("postconvolveSy", kern.Gy)
	cv2.imshow("postconvolveGrad", grad)

	cv2.imshow("canny", can)

	#cv2.imshow("postconvolveSxCVtest", sobelx)
	#cv2.imshow("postconvolveSyCVtest", sobely)


	cv2.waitKey(0)