import matplotlib.pyplot as plt

def func(x):
	k = float(x)
	m = 3.0
	return 1- m / (m + k * (m-1) * m)

def f2(x):
	k = float(x)
	m = 3.0
	return 1- m / (m + k * (m-1) * m)	


k = range(1, 20, 1)
mu = [func(_k) for _k in k]
opt = [0.9 for _ in range(1, 20, 1)]
font_size = 24

plt.xlim(1, 20)
plt.ylabel("Momentum Value", fontsize = font_size)
plt.xlabel("k", fontsize = font_size)
plt.plot(k, mu, 
	color='steelblue', 
	linestyle='-', 
	marker='.', 
	markeredgecolor='steelblue',
	markeredgewidth=2, 
	label='implicit momentum')

plt.plot(k, opt,
	color='black',
	linestyle='-',
	label='optimal momentum'
	)
x = [1, 2, 3, 4, 4.5]
xx = [func(_k) for _k in x]
plt.fill_between(x, xx, opt[:5], color='orange', alpha=0.25, label='explicit momentum')

plt.legend(loc=4, fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.subplots_adjust(top=0.92, bottom=0.17, left=0.17, right=0.92, hspace=0.25, wspace=0.2)
plt.savefig("fig/implicit_momentum.png")
