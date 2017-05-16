#evaluates and returns X-value (for 2d input)
import numpy as np

def obj(inp):	
	#x1= inp[:,0]
	#x2 = inp[:,1]    
	#f =  (x1+(2*x2)-7)**2 + (2*x1 + x2 - 5)**2 #+  np.random.rand()/40.0;          #+ np.sin(x) + np.cos(3*x)
	f = np.power(inp-2,2)# + 3*np.power(inp,3) 
	return f.reshape(-1,1)