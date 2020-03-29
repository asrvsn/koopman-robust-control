''' Various 2x2 LTI systems '''
import numpy as np

from sampler.utils import *

systems = {
	'spiral_sink': np.array([[-1.,1.],[-1.25, -0.45]]), 
	'spiral_sink2': np.array([[-1.65,-4.],[0.5, -0.4]]), 
	'spiral_source': np.array([[1.1,0.75],[-3.25, -0.3]]), 
	# 'nodal_source': np.array([[3.35,0.65],[-1.45, 0.05]]), 
	'nodal_sink ': np.array([[-0.05,-3.00],[0.35, -3.45]]), 
	'center1': np.array([[-1.05,-3.60],[1.10, 1.05]]), 
	'center2': np.array([[-0.4,-1.40],[0.50, 0.4]]), 
	'saddle1': np.array([[0.70,-2.55],[-0.10, -2.50]]), 
	'saddle2': np.array([[0.95,2.25],[0.20, -2.10]]), 
	'saddle3': np.array([[-0.30,1.40],[0.55, 0.40]]), 
	'defective_source': np.array([[0.1, 0.8], [-1.38, 2.2]]),
	'defective_sink': np.array([[0.15, 1.8], [-0.87, -2.35]]),
}

semistable_systems = {
	'spiral_sink1': np.array([[-1.,1.],[-1.25, -0.45]]), 
	'spiral_sink2': np.array([[-1.65,-4.],[0.5, -0.4]]), 
	'nodal_sink ': np.array([[-0.05,-3.00],[0.35, -3.45]]), 
	'center1': np.array([[-1.05,-3.60],[1.10, 1.05]]), 
	'center2': np.array([[-0.4,-1.40],[0.50, 0.4]]), 
}

if __name__ == '__main__': 
	import matplotlib.pyplot as plt
	import scipy.linalg as linalg

	# Trace/determinant plots	
	fig, axs = plt.subplots(2, len(systems), figsize=(12,4))
	for i, (name, A) in enumerate(systems.items()):
		axs[0,i].set_title(name)
		plot_flow_field(axs[0,i], lambda x: A@x, (-4,4), (-4,4))
		plot_trace_determinant(axs[1,i], A, (-4,4), (-4,4))

	plt.show()
