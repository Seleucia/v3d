import  numpy as np
import helper.dt_utils as du

mu, sigma = 0, 0.002 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)

# import matplotlib.pyplot as plt
# count, bins, ignored = plt.hist(s, 30, normed=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
# plt.show()


base_file="/home/coskun/PycharmProjects/data/pose/joints16/"
max_count=1000
p_count=50
sindex=0
mode=0
(X_D,Y_D,F_L,G_L,S_L)=du.joints_sequence_tp1(base_file,max_count,p_count,sindex,mode,get_flist=False)

print "loaded.."