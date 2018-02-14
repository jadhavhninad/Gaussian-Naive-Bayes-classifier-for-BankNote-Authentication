import numpy as np
import collections
import matplotlib as plt
import random

def model(X,Y):
	cp={}
	cp = collections.Counter(Y)
	
	#print(cp[0],cp[1])
	mean={}
	for i in range(0,4,1):
		mean[i]={}
		for z in range(0,2,1):
			m_sum=0
			for j in range(0,X.shape[0],1):
				m_sum +=  X[j][i] if Y[j]==z else 0
			
			mean[i][z] = m_sum/cp[z]
			
	variance={}
	for i in range(0,4,1):
		variance[i]={}
		for z in range(0,2,1):
			var_sum=0
			for j in range(0,X.shape[0],1):
				var_sum +=  (X[j][i] - mean[i][z])**2 if Y[j]==z else 0
			
			variance[i][z] = var_sum/(cp[z]-1)
	
	#print(mean.items())
	#print(variance.items())
	return mean,variance,cp

def classify(X,mean,variance,cp):
	
	pred = np.zeros((1,X.shape[0]))
	#print(pred.shape)
	
	for j in range(0,X.shape[1],1):
		temp_pred=np.zeros((2,1))
		
		for z in range(0,1,1):
			temp_pred[z] = 1
			for i in range(0,X.shape[1],1):
				val1 = 1/(2*3.14*float(variance[i][z]))**0.5
				val2 = -(X[j][i] - mean[i][z])**2
				temp_pred[z] *= float(val1 * np.exp(val2 / (2*variance[i][z])))
				#print(temp_pred[z])
		
		#print("final pred")
		pred_one_j = float(temp_pred[1]*(cp[1]/(cp[0]+cp[1])) / (temp_pred[1]*(cp[1]/(cp[0]+cp[1])) + temp_pred[0]*(cp[0]/(cp[0]+cp[1]))))
		pred[0,j] = 1 if pred_one_j >= 0.5 else 0
	
	#print( pred.shape)
	return pred
	
def main():
		data = np.array(np.genfromtxt('./data.csv',delimiter=','))
		cv_fold = 3
		
		np.random.shuffle(data)
		samples=data.shape[0]
		batch_size=(samples/3)
		acc_plot={}
		
		for i in range(0,cv_fold,1):
			start = batch_size*i
			end = (start + batch_size) if (start+batch_size)<samples else samples
			test_labels = data[int(start):int(end),4]
			test_data = data[int(start):int(end),0:4]
			
			train_set = np.append(data[0:int(start),:],data[int(end):samples,:],axis=0)
			
			data_fracts = [0.05,1]
			for j in data_fracts:
				sample_fract = j*train_set.shape[0]
				print(sample_fract)
				testAcc = 0
				
				for k in range(0,5,1):
					fstart = random.randint(0,int(train_set.shape[0]-sample_fract))
					fend = fstart+sample_fract
					
					#print(train_set[int(fstart):int(fend),:])
					s_train_data = train_set[int(fstart):int(fend), 0:4]
					s_train_label = train_set[int(fstart):int(fend),4]
					
					param_mean, param_var, class_prob = model(s_train_data,s_train_label)
					test_Pred = classify(test_data,param_mean,param_var,class_prob)
					testAcc += 100 - (np.sum(np.abs(test_Pred - test_labels))/test_Pred.shape[1])*100
			
				acc_plot[j] = testAcc/5
				print("for %f, testAcc = %f"%(j,acc_plot[j]))
			break

if __name__ == "__main__":
	main()


	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
