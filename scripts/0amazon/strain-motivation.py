import matplotlib.pyplot as plt

from pylab import mpl

# xmajorLocator = MultipleLocator(24 * 3) #x 24 * 3
# ymajorLocator = MultipleLocator(100 * 2) #y 100 * 2


'''
Heterogeneity =[1, 2, 3, 4, 5, 6, 10]
ConvergenceTime_s40 = [7037, 12503, 14542, 23371, 26194, 28174, 48473]
WaitingTime = [400, 4777, 6346, 11522, 13651, 15430, 28593]
WaitingRatio = [0.0568, 0.3821, 0.4364, 0.4930, 0.5211, 0.5477, 0.5899]
# sizes=15,20,45,10
# colors='yellowgreen','gold','lightskyblue','lightcoral'
# explode=0,0.1,0,0
# plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)

#Lab1
ssp_s = [0, 10, 20, 30, 40, 50]
ConvergenceTime_homo = [9141, 4773, 7892, 6861, 7037, 6194]
ConvergenceTime_hete = [10784, 15648, 15151, 17491, 14542, 17185]
c_homo = [4390, 421, 378, 223, 73, 127]
c_hete = [2618, 507, 250, 196, 62, 117]
#############################################


# Draw
font_size = 15

plt.figure(num=1, figsize=(8, 3))

ax1 = plt.subplot(121)
# ax1.xaxis.set_major_locator(xmajorLocator) 
# ax1.yaxis.set_major_locator(ymajorLocator) 
ax1.xaxis.grid(True, which='major') #x
ax1.yaxis.grid(True, which='major') #x
# plt.axis('Heterogeneity')
# plt.title("Convergence Time with Different Heterogeneity", fontsize=font_size)
plt.ylabel("Convergence Time", fontsize = font_size)
plt.xlim(0, 12)
plt.ylim(0, 60000)
line1 = ax1.plot(Heterogeneity, ConvergenceTime_s40, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='s=40')
plt.xlabel("Heterogeneity(Vavg/Vmin)", fontsize = font_size)
# plt.savefig("strain-motivation2.jpg")

# plt.figure(num=2, figsize=(8, 6))
ax2 = plt.subplot(122)
ax2.xaxis.grid(True, which='major') #x
ax2.yaxis.grid(True, which='major') #x
# plt.title("Waiting Time Ratio with Different Heterogeneity", fontsize=font_size)
plt.ylabel("Waiting Time Ratio", fontsize = font_size)
plt.xlim(0, 12)
plt.ylim(0, 0.7)
line1 = ax2.plot(Heterogeneity, WaitingRatio, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='s=40')

# plt.show()
plt.xlabel("Heterogeneity(Vavg/Vmin)", fontsize = font_size)
plt.subplots_adjust(top=0.92, bottom=0.2, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.2)
plt.savefig("/Users/hhp/Dropbox/Hu Hanpeng/paper draft/fig/strain-motivation3.jpg")



#############################################
plt.figure(num=2, figsize=(8, 3))

ax3 = plt.subplot(121)
ax3.xaxis.grid(True, which='major') #x
ax3.yaxis.grid(True, which='major') #x
# plt.title("Performance Difference with Heterogeneity or Homogeneity", fontsize=18)
plt.xlabel("SSP threshold s", fontsize = font_size)
plt.ylabel("Convergence Time", fontsize = font_size)
plt.xlim(0, 60)
plt.ylim(0, 20000)
line1 = ax3.plot(ssp_s, ConvergenceTime_homo, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='homogeneous')

line2 = ax3.plot(ssp_s, ConvergenceTime_hete, 
	color='lightskyblue', 
	linestyle='--', 
	marker='x', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='heterogeneous')

ax4 = plt.subplot(122)
ax4.xaxis.grid(True, which='major') #x
ax4.yaxis.grid(True, which='major') #x
# plt.title("Performance Difference with Heterogeneity or Homogeneity", fontsize=font_size)
plt.xlabel("SSP threshold s", fontsize = font_size)
plt.ylabel("Average Commit Number", fontsize = font_size)
plt.xlim(0, 60)
plt.ylim(0, 3000)
line1 = ax4.plot(ssp_s, c_homo, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='homogeneous')
line2 = ax4.plot(ssp_s, c_hete, 
	color='lightskyblue', 
	linestyle='--', 
	marker='x', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='heterogeneous')

# plt.show()
plt.legend(ncol=1, loc = 9)
plt.subplots_adjust(top=0.92, bottom=0.2, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.3)
plt.savefig("/Users/hhp/Dropbox/Hu Hanpeng/paper draft/fig/strain-motivation1.jpg")
'''

Heterogeneity =[1, 2, 3, 4, 5, 6, 10]
ssp_s40_time = [2947, 5470, 7331, 8110, 11354, 13095, 21504]
usp_nobeta_time = [3069, 3072, 2960, 3915, 3429, 2771, 2394]
usp_beta_time = [6262, 5194, 5902, 6454, 7828, 6505, 6092]
usp_nobeta_gc = [0.721, 0.702, 0.709, 0.709, 0.700, 0.731,0.7055]
usp_nobeta_lc = [0.403, 0.366, 0.351, 0.398, 0.308, 0.581,0.4165]
usp_nobeta_glr = [y / x for (x, y) in zip(usp_nobeta_gc, usp_nobeta_lc)]
usp_beta_gc = [0.748, 0.708, 0.725, 0.736, 0.718, 0.70875, 0.737]
usp_beta_lc = [0.809, 0.812, 0.605, 0.622, 0.7395, 0.699, 0.601]
usp_beta_glr = [y / x for (x, y) in zip(usp_beta_gc, usp_beta_lc)]
ssp_s40_gc = [0.757, 0.760, 0.742, 0.750, 0.755, 0.749, 0.728]
ssp_s40_lc = [0.699, 0.673, 0.555, 0.800, 0.682, 0.668, 0.464]
ssp_s40_glr = [y / x for (x, y) in zip(ssp_s40_gc, ssp_s40_lc)]

ssp_s40_c = [73, 68, 62, 51, 57, 55, 131]
usp_nobeta_c = [120, 145, 262, 199, 91, 123, 131]
usp_beta_c = [237, 471, 268, 402, 313.5, 286, 310]

font_size = 15


'''
plt.figure(num=1, figsize=(8, 8))

ax1 = plt.subplot(221)
# ax1.xaxis.set_major_locator(xmajorLocator) 
# ax1.yaxis.set_major_locator(ymajorLocator) 
ax1.xaxis.grid(True, which='major') #x
ax1.yaxis.grid(True, which='major') #x
# plt.axis('Heterogeneity')
# plt.title("Convergence Time with Different Heterogeneity", fontsize=font_size)
plt.ylabel("Convergence Time(s)", fontsize = font_size)
plt.xlim(0, 12)
plt.ylim(0, 40000)
line1 = ax1.plot(Heterogeneity, ssp_s40_time, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='SSP s=40')
line2 = ax1.plot(Heterogeneity, usp_nobeta_time, 
	color='lightskyblue', 
	linestyle='--', 
	marker='x', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain')
ax1.legend(ncol=1)
ax1.set_title("(a)")
# plt.savefig("strain-motivation2.jpg")

# plt.figure(num=2, figsize=(8, 6))
ax2 = plt.subplot(222)
ax2.xaxis.grid(True, which='major') #x
ax2.yaxis.grid(True, which='major') #x
# plt.title("Waiting Time Ratio with Different Heterogeneity", fontsize=font_size)
plt.ylabel("Global Accuracy", fontsize = font_size)
plt.xlim(0, 12)
plt.ylim(0, 1.5)
line3 = ax2.plot(Heterogeneity, ssp_s40_gc, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='SSP s=40')
line4 = ax2.plot(Heterogeneity, usp_nobeta_gc, 
	color='lightskyblue', 
	linestyle='--', 
	marker='x', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain')
ax2.legend(ncol=1)
ax2.set_title("(b)")

ax3 = plt.subplot(223)
ax3.xaxis.grid(True, which='major') #x
ax3.yaxis.grid(True, which='major') #x
# plt.title("Waiting Time Ratio with Different Heterogeneity", fontsize=font_size)
plt.ylabel("Minority Accuracy", fontsize = font_size)
plt.xlim(0, 12)
plt.ylim(0, 1)
line1 = ax3.plot(Heterogeneity, ssp_s40_lc, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='SSP s=40')
line2 = ax3.plot(Heterogeneity, usp_nobeta_lc, 
	color='lightskyblue', 
	linestyle='--', 
	marker='x', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain')
ax3.legend(ncol=1)
ax3.set_title("(c)")
plt.xlabel("Heterogeneity(Vavg/Vmin)", fontsize = font_size)

ax4 = plt.subplot(224)
ax4.xaxis.grid(True, which='major') #x
ax4.yaxis.grid(True, which='major') #x
# plt.title("Waiting Time Ratio with Different Heterogeneity", fontsize=font_size)
plt.ylabel("Minority Accuracy : Global Accuracy", fontsize = font_size)
plt.xlim(0, 12)
plt.ylim(0, 1.5)
line1 = ax4.plot(Heterogeneity, ssp_s40_glr, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='SSP s=40')
line2 = ax4.plot(Heterogeneity, usp_nobeta_glr, 
	color='lightskyblue', 
	linestyle='--', 
	marker='x', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain')
ax4.legend(ncol=1, loc='best')
ax4.set_title("(d)")
plt.xlabel("Heterogeneity(Vavg/Vmin)", fontsize = font_size)
# plt.show()


plt.subplots_adjust(top=0.92, bottom=0.1, left=0.10, right=0.95, hspace=0.25, wspace=0.2)
plt.savefig("/Users/hhp/Dropbox/Hu Hanpeng/paper draft/fig/strain_ssp40_contrast.jpg")

'''








# plt.figure(num=1, figsize=(8, 5))
# ax1 = plt.subplot(111)
# ax1.xaxis.grid(True, which='major') #x
# ax1.yaxis.grid(True, which='major') #x
# # plt.title("Waiting Time Ratio with Different Heterogeneity", fontsize=font_size)
# plt.ylabel("Total Commit Count", fontsize = font_size)
# plt.xlim(0, 12)
# plt.ylim(0, 500)
# line1 = ax1.plot(Heterogeneity, ssp_s40_c, 
# 	color='lightgreen', 
# 	linestyle='-', 
# 	marker='.', 
# 	markeredgecolor='red',
# 	markeredgewidth=2, 
# 	label='SSP s=40')
# line2 = ax1.plot(Heterogeneity, usp_nobeta_c, 
# 	color='lightskyblue', 
# 	linestyle='--', 
# 	marker='x', 
# 	markeredgecolor='red',
# 	lw=2,
# 	markeredgewidth=2, 
# 	label='STrain')
# line3 = ax1.plot(Heterogeneity, usp_beta_c, 
# 	color='lightskyblue', 
# 	linestyle='-', 
# 	marker='.', 
# 	markeredgecolor='red',
# 	lw=2,
# 	markeredgewidth=2, 
# 	label='STrain-beta')
# ax1.legend(ncol=3)

# plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.95, hspace=0.25, wspace=0.2)
# plt.xlabel("Heterogeneity(Vavg/Vmin)", fontsize = font_size)
# plt.savefig("/Users/hhp/Dropbox/Hu Hanpeng/paper draft/fig/commit_contrast.jpg")







plt.figure(num=1, figsize=(8, 8))
ax1 = plt.subplot(221)
ax1.xaxis.grid(True, which='major') #x
ax1.yaxis.grid(True, which='major') #x
# plt.title("Waiting Time Ratio with Different Heterogeneity", fontsize=font_size)
plt.ylabel("Global Accuracy", fontsize = font_size)
plt.xlim(0, 12)
plt.ylim(0, 1.5)
line1 = ax1.plot(Heterogeneity, ssp_s40_gc, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='SSP s=40')
line2 = ax1.plot(Heterogeneity, usp_nobeta_gc, 
	color='lightskyblue', 
	linestyle='--', 
	marker='x', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain')
line3 = ax1.plot(Heterogeneity, usp_beta_gc, 
	color='lightskyblue', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain-beta')
ax1.legend(ncol=1)
ax1.set_title("(a)")


ax2 = plt.subplot(222)
ax2.xaxis.grid(True, which='major') #x
ax2.yaxis.grid(True, which='major') #x
# plt.title("Waiting Time Ratio with Different Heterogeneity", fontsize=font_size)
plt.ylabel("Minority Accuracy", fontsize = font_size)
plt.xlim(0, 12)
plt.ylim(0, 1.2)
line1 = ax2.plot(Heterogeneity, ssp_s40_lc, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='SSP s=40')
line2 = ax2.plot(Heterogeneity, usp_nobeta_lc, 
	color='lightskyblue', 
	linestyle='--', 
	marker='x', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain')
line3 = ax2.plot(Heterogeneity, usp_beta_lc, 
	color='lightskyblue', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain-beta')
ax2.legend(ncol=1)
ax2.set_title("(b)")

ax3 = plt.subplot(212)
ax3.xaxis.grid(True, which='major') #x
ax3.yaxis.grid(True, which='major') #x
# plt.title("Waiting Time Ratio with Different Heterogeneity", fontsize=font_size)
plt.ylabel("Minority Accuracy : Global Accuracy", fontsize = font_size)
plt.xlim(0, 12)
plt.ylim(0, 1.5)
line1 = ax3.plot(Heterogeneity, ssp_s40_glr, 
	color='lightgreen', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	markeredgewidth=2, 
	label='SSP s=40')
line2 = ax3.plot(Heterogeneity, usp_nobeta_glr, 
	color='lightskyblue', 
	linestyle='--', 
	marker='x', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain')
line3 = ax3.plot(Heterogeneity, usp_beta_glr, 
	color='lightskyblue', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='red',
	lw=2,
	markeredgewidth=2, 
	label='STrain-beta')
ax3.legend(ncol=3, loc='best')
ax3.set_title("(c)")

plt.subplots_adjust(top=0.92, bottom=0.1, left=0.10, right=0.95, hspace=0.25, wspace=0.2)
plt.xlabel("Heterogeneity(Vavg/Vmin)", fontsize = font_size)
plt.savefig("/Users/hhp/Dropbox/Hu Hanpeng/paper draft/fig/strain_beta_ssp40_contrast.jpg")

