import numpy as np
from rgx import *

dfile = 'dataset/Hierarchical_7.csv'
csv = np.genfromtxt(dfile, delimiter=',')
data = csv[:,0:2]

plot = visualize(data, title="visualisasi data asli", mode="unsupervised")
methods = ["single","complete","centroid","group"]

for m in methods:
	linkage, clusters = agglomerative(data, method=m)
	plot_hie_title = "hirarki cluster dengan " + m
	plot_nest_title = "nested cluster dengan " + m
	
	visualize_dendogram(linkage, title=plot_hie_title)
	visualize_nested(data, clusters, title=plot_nest_title)
	
plot.show()



