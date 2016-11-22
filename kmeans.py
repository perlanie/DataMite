import math
import random
import sys
import numpy as np

def centroids(points, closest, k):
    clusters = []
    for i in range(k):
        clusters.append(np.ndarray((0,len(points[0])))) #create a list of numpy arrays, one for each of the k centers, which we will add points to

    for i in range(len(points)):
        clusters[closest[i]] = np.append(clusters[closest[i]], [points[i]], axis=0) #add the point to an array for its closest center
        #now each center has a seperate numpy array containing all the points in its cluster
    avg = []
    for i in range(k):
        avg.append(tuple(np.mean(clusters[i], axis = 0))) #put the centroid of the cluster in an array called averages
    return avg


def kMeans(k, points):
    assert k < len(points)
    random.seed()
    centers = []
    while (len(centers) < k):
        p = points[random.randint(0, len(points) - 1)]
        if (p not in centers): #ensure we dont get duplicates
            centers.append(p)
	
    minDist = [sys.maxsize for x in range(len(points))]
    closest= [sys.maxsize for x in range(len(points))]
    #print "Initial centers: ", centers
    oldCenters = []
    while centers != oldCenters:
        print "iteration"
        oldCenters = centers
        for i in range(len(centers)):
            for j in range(len(points)):
                dist = distance (centers[i], points[j])
                if minDist[j] > dist:
                    minDist[j] = dist
                    closest[j] = i
        centers = centroids(points, closest, k)

    return centers, closest
        
#computes the distance between 2 n dimensional tuples
def distance(a, b):
    assert len(a) == len(b)
    d = 0
    for i in range(len(a)):
        d += (a[i]-b[i])*(a[i]-b[i])
    return math.sqrt(d)

pts = []
for i in range(100000):
    pts.append((random.randint(-1000,1000), random.randint(-1000,1000)))

print kMeans(100, pts)

