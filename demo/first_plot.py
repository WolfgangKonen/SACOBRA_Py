import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# import seaborn as sns
#df = sns.load_dataset('iris')

fn = lambda x: 3 * np.sum(x ** 2)

s = 2
x1 = np.linspace(-s,s,30)
x2 = np.linspace(-s,s,30)
X, Y = np.meshgrid(x1, x2)
Z = 2*(X**2 + Y**2)

fig, ax = plt.subplots()

#CS = ax.contour(X, Y, Z)
#ax.clabel(CS, fontsize=10)
#ax.set_title('Simplest default with labels')

# filled contour with colormap cmap:
cmap = plt.colormaps['viridis']         # alt: cm.gray
im = ax.imshow(Z, interpolation='bilinear', origin='lower',
               cmap=cmap, extent=(-3, 3, -3, 3))
# contour lines at certain levels with colormap gray:
levels = np.concat((np.arange(0,2,0.5)-0.05, np.arange(2,10,1)))
CS = ax.contour(Z, levels, origin='lower', cmap='gray', extend='both',
                linewidths=1, extent=(-3, 3, -3, 3))
# 'extent' defines the extent of the plot (xmin,xmax,ymin,ymax)

# additional lines: constraint border (thick red), axes (thin black)
plt.plot([-2,3],[1+2,1-3], 'r-')
plt.plot([0,0],[-3,3],'k-',linewidth=0.5)
plt.plot([-3,3],[0,0],'k-',linewidth=0.5)

plt.savefig('../demo/contour_plot.png')
plt.show()