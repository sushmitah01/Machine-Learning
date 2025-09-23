import numpy
import matplotlib.pyplot as plt
from scipy import stats
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x=numpy.mean(speed)
y=numpy.median(speed)
z=stats.mode(speed)
sd=numpy.std(speed)#standard Deviation low is close to mean(average) and high is in wider range
variance=numpy.var(speed) #sqr root variance is Sd
percentile=numpy.percentile(speed,75)


print(x)
print(y)
print(z)
print(sd)
print (variance)
print(percentile)
