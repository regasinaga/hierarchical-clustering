import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram

colors = np.array(['#ff3333', '#ff6633','#ff9933',
					'#ffcc33', '#ffff33', '#ccff33',
					'#99ff33', '#33ff33', '#33ff99',
					'#33ffff', '#3399ff', '#3333ff',
					'#9933ff', '#ff33ff', '#ff3366'])
							
def visualize(d, t=None, title="plot", mode="supervised"):
	global colors
	plot.figure()
	mark = 'o'
	if mode=="supervised":
		tn = len(np.unique(t))
		for tc in range(0, tn):
			ind = np.where(t==tc+1)
			plot.scatter(d[ind,0], d[ind,1], marker=mark, color=colors[tc])
	elif mode=="unsupervised":
		plot.scatter(d[:,0], d[:,1], marker=mark, color=colors[2])
	plot.title(title)
	return plot
	
def visualize_dendogram(linkage, title="hirarki"):
	plot.figure(figsize=(10,5))
	plot.title(title)
	plot.xlabel("sample index")
	plot.ylabel("dissimilarity")
	dendrogram(
		linkage,
		leaf_rotation=90.0,
		leaf_font_size=10.0,
	)
	return plot
	
def visualize_nested(x, clusters, title="nested"):
	global colors
	plot.figure()
	plot.title(title)
	mark='o'
	start_draw = len(clusters) - 1
	end_draw = len(x) - 1

	for c in range(start_draw, end_draw, -1):
		attr1_diff = np.max(clusters[c][:,0]) - np.min(clusters[c][:,0])
		attr2_diff = np.max(clusters[c][:,1]) - np.min(clusters[c][:,1])
		centroid = np.mean(clusters[c], axis=0)
		
		radius = max(attr1_diff, attr2_diff) / 2 
		center = (centroid[0], centroid[1])
		
		add = np.power(1.1, 1.1*len(clusters[c]))
		circle = plot.Circle(center, radius + add, edgecolor='b', facecolor='#ffffff', zorder=1)
		plot.gca().add_patch(circle)
	
	plot.axis("scaled")
	
	plot.scatter(x[:,0], x[:,1], marker=mark, color=colors[2], zorder=2)
	return plot
	
def single_link(mcp, mcq):
	repeatp = len(mcq)
	repeatq = len(mcp)
	dissimilarities = euclid( np.repeat(mcp, repeatp, axis=0), np.tile(mcq, (repeatq, 1)))
	return np.min(dissimilarities)

def complete_link(mcp, mcq):
	repeatp = len(mcq)
	repeatq = len(mcp)
	dissimilarities = euclid( np.repeat(mcp, repeatp, axis=0), np.tile(mcq, (repeatq, 1)))
	return np.max(dissimilarities)

def group_average(mcp, mcq):
	repeatp = len(mcq)
	repeatq = len(mcp)
	dissimilarities = euclid( np.repeat(mcp, repeatp, axis=0), np.tile(mcq, (repeatq, 1)))
	return (np.sum(dissimilarities)) / (len(mcp) * len(mcq))

def centroid_based(mcp, mcq):
	cp = np.array([np.mean(mcp, axis=0)])
	cq = np.array([np.mean(mcq, axis=0)])
	
	return euclid(cp, cq)

def agglomerative(x, method="single"):
	measures = {
		"single": single_link,
		"complete": complete_link,
		"group": group_average,
		"centroid": centroid_based
	}
	
	clusters = []
	stack = [-1]
	linkage = np.array([[]]); linkage.shape = (0,4)
	for xi in x: clusters.append(np.array([xi]))
	
	# saving cluster index
	cluster_indx = list(range(0, len(clusters)))
	active = None
	# ALGORITHM : NEAREST NEIGHBOR CHAIN
	while len(cluster_indx) > 1:

		if len(stack) == 1:
			stack.append(cluster_indx[0])
		active = stack[-1]
		# finding closest neighbor
		dissimilar = np.ones(len(cluster_indx)) * np.Inf
		assign_to = 0;
		for ci in cluster_indx:
			if ci != active: 
				dissimilar[assign_to] = measures[method](clusters[active], clusters[ci])
			assign_to += 1
			
		D = cluster_indx[np.argmin(dissimilar)]
		closest_dissimilar = np.min(dissimilar) 

		# if these two are mutual nearest neighbor
		if D in stack and D == stack[-2]:
			indq = stack.pop(-1)
			indp = stack.pop(-1)
			cluster_indx.pop(cluster_indx.index(indp))
			cluster_indx.pop(cluster_indx.index(indq))
			
			# merging clusters
			nu_cluster = np.concatenate((clusters[indp], clusters[indq]), axis=0)
			# updating clusters array and accessible cluster index
			clusters.append(nu_cluster)
			cluster_indx.append(len(clusters)-1)
			
			#updating linkage 
			nu_linkage = np.array([[indp, indq, closest_dissimilar, len(clusters[indp]) + len(clusters[indq])]])
			linkage = np.concatenate((linkage, nu_linkage), axis=0)
		else:
			stack.append(D)
		del dissimilar

	return linkage, clusters

def euclid(p, q):
	return np.sqrt(np.sum(np.power(p - q, 2.0), axis=1))
