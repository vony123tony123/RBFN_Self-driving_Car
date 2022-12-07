import numpy as np
import random
import matplotlib.pyplot as plt
from toolkit import toolkit

class RBFN:

	def readFile(file):
		dataset = list()
		answers = list()
		with open(file, 'r') as f:
		    lines = f.readlines()
		    for line in lines:
		        data_list = line.split(" ")
		        dataset.append(list(map(float,data_list[:-1])))
		        answers.append(float(data_list[-1]))
		dataset = np.array(dataset)
		answers = np.array(list(map(lambda x: (x+40)/80, answers)))
		return dataset, answers

	def guass_function(self, x, m, std):
	    return np.exp(-1 * (toolkit.euclid_distance(x, m)**2 / (2 * std**2)))

	def Kmeans(self, n_clusters,points, max_epochs):
	    randomIntArray = [random.randint(0,len(points)-1) for k in range(n_clusters)]
	    m = points[randomIntArray]
	    for epoch in range(max_epochs):
	        d = toolkit.euclid_distance_2d(points, m)
	        clusters = np.argmin(d, axis=-1)
	        m = [np.mean(points[clusters==k], axis=0) for k in range(n_clusters)]
	        m = np.array(m)
	    std = np.array([np.mean(toolkit.euclid_distance(points[clusters==k],m[k]), axis=0) for k in range(n_clusters)])
	    return m, std, clusters

	def forward(self, x, m, std, w, delta):
	    guass = self.guass_function(x,m,std)
	    return guass, np.dot(guass, w) + delta

	def optimize(self, lr, data, ans, F, guass_list, m_list, std_list, w_list, delta):
		m_gradient = (ans - F) * w_list * guass_list / std_list**2
		m_gradient = np.array([m_gradient]).T
		m_gradient = m_gradient * (data - m_list)
		m_after = m_list + lr * m_gradient

		std_graident = (ans - F) * w_list * guass_list / std_list**3
		std_graident = std_graident * toolkit.euclid_distance([data], m_list)[0]**2
		std_after = std_list + lr * std_graident
		
		w_after = w_list + lr * (ans - F) * guass_list
		delta_after = delta + lr * (ans - F)

		return m_after, std_after, w_after, delta_after

	def train(self, dataset, answers, K=3, lr=0.1, max_epochs=100):
		w_list = np.random.randn(K)
		delta = np.random.randn()
		m_list,std_list ,clusters = self.Kmeans(K, dataset, 100)

		# print("m_list = ", m_list)
		# print("w_list = ",w_list)
		# print("delta = ", delta)
		# print("std_list = ", std_list)

		for epoch in range(max_epochs):
			ME = 0
			for data, ans in zip(dataset, answers):
				guass_list, F = self.forward(data, m_list, std_list, w_list, delta)
				E = (ans - F)**2/2
				ME = ME + E

				# print("data = ", data)
				# print("m_list = ", m_list)
				# print("guass_list = ",guass_list)
				# print("w_list = ",w_list)
				# print("delta = ", delta)
				# print("std_list = ", std_list)
				# print("ans = ", ans)
				# print("F = ",F)

				m_gradient = (ans - F) * w_list * guass_list / std_list**2
				m_gradient = np.array([m_gradient]).T
				m_gradient = m_gradient * (data - m_list)
				m_after = m_list + lr * m_gradient

				std_graident = (ans - F) * w_list * guass_list / std_list**3
				std_graident = std_graident *np.sum((data - m_list)**2, axis = -1)
				std_after = std_list + lr * std_graident
				
				w_after = w_list + lr * (ans - F) * guass_list
				delta_after = delta + lr * (ans - F)

				m_list = m_after
				std_list = std_after
				w_list = w_after
				delta = delta_after

				# print("---------------------------")
				# print("m_list = ", m_list)
				# print("w_list = ",w_list)
				# print("delta = ", delta)
				# print("std_list = ", std_list)
				# break
			ME = ME / len(dataset)
			# if epoch % 100 == 0:
			# 	print("Epoch {} : mean loss = {}".format(epoch, ME))
			print("Epoch {} : mean loss = {}".format(epoch, ME))

		self.m_list = m_list
		self.std_list = std_list
		self.w_list = w_list
		self.delta = delta

	def predict(self, data):
		guass = self.guass_function(data,self.m_list,self.std_list)
		result = np.dot(guass, self.w_list) + self.delta
		return result.item() * 80 - 40

if __name__ == "__main__":
	read_file = "D:/Project/NNHomework/NN_HW2/train4dAll.txt"

	RBFN = RBFN()

	dataset, answers = RBFN.readFile(read_file)

	RBFN.train(dataset, answers, max_epochs = 1)
