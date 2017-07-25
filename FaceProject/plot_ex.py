
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# def update_line(num, data, line):
#     line.set_data(data[..., :num])
#     return line,

# fig1 = plt.figure()

# data = np.random.rand(2, 25)
# l, = plt.plot([], [], 'r-')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.xlabel('x')
# plt.title('test')
# line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
#                                    interval=50, blit=True)

# # To save the animation, use the command: line_ani.save('lines.mp4')

# fig2 = plt.figure()
# ax1 = fig2.add_subplot(1,1,1)

# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
# y = [0.16637881318707332, 0.36513194514572622, 0.28626505371563582, 0.30774320639357172, 0.33110493940331609, 0.36820933888498886, 0.32888524607686798, 0.34201991617387717, 0.30791291394335119, 0.32061087588135706, 0.32061087588135706, 0.31872385092060773, 0.32513410733086417, 0.33531346781217497, 0.31638037095599181, 0.33868996455150924, 0.30779006887307037, 0.29738677399798608, 0.33554955705862122, 0.31058311179940529, 0.32340362461991812, 0.30514102524758546, 0.34648114221226201, 0.30023040443692961, 0.29232051242707269, 0.32330475119729513, 0.34836322076515114, 0.32340362461991812, 0.35870264258353668, 0.32106373777026298, 0.3298138810301745, 0.31872385092060773, 0.28800838238485871, 0.32676068415392728, 0.32048803081107624, 0.31150688077789301, 0.33729360229101663, 0.30723915161562793, 0.30914643815842058, 0.32203979891498236, 0.29036882500433114, 0.30280113839793033, 0.2529373263201663, 0.26255271093555088, 0.2529373263201663, 0.27218389926643322, 0.27326691606421122, 0.19175342428830791, 0.20457856804048924, 0.24653706500947725, 0.16637881318707332, 0.30057689703590729, 0.3801006950921203, 0.38125475142302689, 0.37868276995760575, 0.39585558137712173, 0.28250081579745789, 0.26085289890155339] 
# base = np.hypot(x, y)
# ims = []

# ims.append((ax1.plot(x, y)))
# im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=20,blit=True)
# # To save this second animation with some metadata, use the following command:
# # im_ani.save('im.mp4', metadata={'artist':'Guido'})

# plt.show()
# 
# 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np


#style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
	EYE_AR_THRESH = 0.3
	EYE_AR_CONSEC_FRAMES = 48
	COUNTER = 0
	graph_data = open('test.txt', 'r').read()
	lines = graph_data.split('\n')
	xs = []
	ys = []
	colors = []
	for line in lines:
		if len(line) > 1:
			x, y = line.split(',')
			if float(y) < EYE_AR_THRESH:
				COUNTER += 1
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					colors.append('r')
				else:
					colors.append('y')
			else:
				colors.append('b')
				COUNTER = 0
			xs.append(x)
			ys.append(y)
	ax1.clear()
	ax1.plot(xs, ys)
	plt.axhline(y=0.30, c="red")
	ax1.scatter(xs, ys, c=colors, marker="o", picker=True)


ani = animation.FuncAnimation(fig, animate, interval=50, frames=100)
plt.show()