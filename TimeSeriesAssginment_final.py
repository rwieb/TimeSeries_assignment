# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:00:12 2021

@author: Rutger
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.stats import norm, mode
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
from scipy.spatial.distance import squareform

######################## PREPROCESSING AND WINDOWING ##########################################################
######### Question 4.1

    #a)
    
features = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/features1.txt',delimiter='\s+',header=None)
df_train_wl = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/train/X_train.txt',sep='\s+',header=None)
df_train_wl.columns = [features.loc[0:,0]]
# df_test_wl = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/test/X_test.txt',sep='\s+',header=None)
# df_test_wl.columns = [features.loc[0:,0]]


#     #b)
    
# statistics = df_test_wl[["tBodyAcc-mean()-X","tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z","tBodyAcc-std()-X","tBodyAcc-std()-Y","tBodyAcc-std()-Z"]].describe()


# ######### Question 4.2

#     #a)
# df_train_y = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/train/y_train.txt',sep='\s+',header=None)
# df_test_y = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/test/y_test.txt',sep='\s+',header=None)

#     #b)

# x = df_test_y[0].value_counts().to_frame()
# fig, ax1 = plt.subplots()
# plt.bar(x.index,x.iloc[0:,0])
# plt.xlabel("Activity")
# plt.ylabel("number of occurences in data")

# ######### Question 4.3

# df_train = pd.merge(df_train_y,df_train_wl,left_index=True,right_index=True)
# df_test = pd.merge(df_test_y,df_test_wl,left_index=True,right_index=True)

# for x in range(1,100,10):
#     df_c1 = pd.DataFrame(data=[df_train.groupby([0]).get_group(1).iloc[0:,x],df_train.groupby([0]).get_group(2).iloc[0:,x],df_train.groupby([0]).get_group(3).iloc[0:,x],df_train.groupby([0]).get_group(4).iloc[0:,x],df_train.groupby([0]).get_group(5).iloc[0:,x],df_train.groupby([0]).get_group(6).iloc[0:,x]]).transpose()
#     df_c1.columns=['1','2','3','4','5','6']
#     fig, ax = plt.subplots()
#     sns.kdeplot(data=df_c1)
#     plt.xlabel(f'feature {df_train.columns[x]}')

######################## EXPLORING THE RAW DATA ##########################################################
######### Question 4.4

df_total_acc_x = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt',sep='\s+',header=None)
df_total_acc_y = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt',sep='\s+',header=None)
df_total_acc_z = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt',sep='\s+',header=None)
    
    #a)
df_total_acc_x.shape

    #b)

var_x_y_z = [df_total_acc_x.iloc[::2].melt().drop(labels="variable",axis=1).var().values,df_total_acc_y.iloc[::2].melt().drop(labels="variable",axis=1).var().values,df_total_acc_z.iloc[::2].melt().drop(labels="variable",axis=1).var().values]
df_bodyacc_x = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt',sep='\s+',header=None)
df_train_y = pd.read_fwf('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/UCI HAR Dataset/train/y_train.txt',sep='\s+',header=None,names=['activity'])
df_bodyacc = pd.merge(df_train_y,df_bodyacc_x, left_index=True, right_index=True)
df_bodyacc.reset_index(level=0, inplace=True)
orig_signal = pd.melt(df_bodyacc.iloc[::2],id_vars=['index','activity'],value_vars=range(2,((len(df_bodyacc.columns))-2))) #get original signal, delete every uneven column to account for 50% overlap
sort_sign = orig_signal.sort_values(by=['index','variable'],axis=0,ignore_index=True)

#plot signal
sort_sign.plot(y=['value'])

########################  THE TIME DOMAIN AND THE FREQUENCY DOMAIN ##########################################################
######### Question 4.5
    #a)
    
#set working directory
# os.chdir("C:/Users/hugob/OneDrive - Universiteit Twente/Others/Desktop/UT Courses/Q6/Data Science/Time Series/UCI HAR Dataset/UCI HAR Dataset") 
    
#read raw signals
#read inertial signals train
acc_x_train = pd.read_fwf("train/Inertial Signals/body_acc_x_train.txt", header = None)

#read label files
y_train = pd.read_fwf("train/y_train.txt", header = None)
# y_test = pd.read_fwf("test/y_test.txt", header = None)

#give summary statistics        
mean = acc_x_train.mean(axis=1).to_frame()
standard_dev = acc_x_train.std(axis=1).to_frame()
kurtosis = acc_x_train.kurtosis(axis=1).to_frame()
skewness = acc_x_train.skew(axis=1).to_frame()
median = acc_x_train.median(axis=1).to_frame()
variance =  acc_x_train.var(axis=1).to_frame()
minimun = acc_x_train.min(axis=1).to_frame()
maximun = acc_x_train.max(axis=1).to_frame()

#add label info to statistics
mean['Label'] = y_train [0]
standard_dev['Label'] = y_train [0]

#simplify statistics
# class_describe = sort_sign.groupby('activity')['value'].describe()

    #b)
    
#attribute colors to each activity
colors = {1:'red', 2:'green', 3: 'black', 4:'blue', 5:'purple', 6:'yellow'}
mean['color'] = 0
for i in mean.index:
    mean['color'][i] = colors[mean['Label'][i]]

#reconstruct normal distributions
x = np.arange(-1, 1, 0.001)
for j in range(1,7):
    for i in range(len(mean)):
        if mean['Label'][i] == j: #plot the distributions of each activity separately
            plt.plot(x, norm.pdf(x, mean[0][i], standard_dev[0][i]), linewidth = 0.5, color = mean['color'][i], label = str(j))
    plt.title('normal distribution - activity ' + str(j))
    plt.show()


######### Question 4.6
    #a)
    
    #read raw signals
acc_x_train = pd.read_fwf("train/Inertial Signals/body_acc_x_train.txt", header = None)
y_train = pd.read_fwf("train/y_train.txt", header = None) #read label files
acc_x_train['Label'] = y_train [0] #add label info to raw data

#FFT activities
for i in range(1,7):
    acc_x_train_activ = acc_x_train[acc_x_train['Label'] == i]
    acc_x_train_activ = acc_x_train_activ.drop(['Label'], axis=1) #remove label column
    # plt.plot(acc_x_train_activ6, color = 'blue') #plot of signal
    
    #sample spacing
    y = acc_x_train_activ - acc_x_train_activ.mean() # remove the DC-offset
    y = np.array(y) # convert dataframe to array 
    N = len(y) # Number of sample points
    T = 1/50 # sample spacing
    x = np.linspace(0, N*T, N, endpoint=False)
    yf = fft(y)
    xf = fftfreq(N,T)[:N//2]
    # xf = np.linspace(0, 1/(2*T), N//2)
    
    #plot spectrum
    plt.plot(xf, 2/N * np.abs(yf[0:N//2]), color = 'blue')
    plt.title('FFT of activity ' + str(i))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|DFT(K)|')
    plt.grid()
    plt.show()
    

########################  FILTERING ############################################################
######### Question 4.7
    #a)
# fig,ax5 = plt.subplots(2)
# ax5[0].plot(sort_sign.index,sort_sign.loc[:,'value'])
# ax5[0].axis([0, len(sort_sign.loc[:,'value']), -1.2, 1.2])
# #ax5.plot(range(0,len(sort_sign)),sort_sign.loc[:,'value'])
    #b)
sos_low = signal.butter(5, 4, btype='lowpass', fs=50, output='sos') #order,crit_freq,which filter, output=
filt_low = signal.sosfilt(sos_low, sort_sign.loc[:,'value'])
sos_high = signal.butter(5, 0.2, btype='highpass', fs=50, output='sos') #order,crit_freq,which filter, output=
filt_high = signal.sosfilt(sos_high, sort_sign.loc[:,'value'])
sos_band = signal.butter(5, [0.2,4], btype='bandpass', fs=50, output='sos') #order,crit_freq,which filter, output=
filt_band = signal.sosfilt(sos_band, sort_sign.loc[:,'value'])
# ax5[1].plot(sort_sign.index, filtered)
# ax5[1].set_title('After 4 Hz low-pass filter')
# ax5[1].axis([0, len(sort_sign.loc[:,'value']), -1.2, 1.2])
# ax5[1].set_xlabel('--')
# plt.tight_layout()
# plt.show()

# fig,ax6 = plt.subplots(2)
# ax6[0].plot(sort_sign.index[7900:8101],sort_sign.loc[7900:8100,'value'])
# ax6[0].axis([7900, 8100, -1.2, 1.2])
# #ax6.plot(range(0,len(sort_sign)),sort_sign.loc[:,'value'])
# ax6[1].plot(sort_sign.index[7900:8100], filtered[7900:8100])
# ax6[1].set_title('After 4 Hz low-pass filter')
# ax6[1].axis([7900, 8100, -1.2, 1.2])
# ax6[1].set_xlabel('--')
# plt.tight_layout()
# plt.show()

df_filt = pd.DataFrame(data=[sort_sign.iloc[:,0],sort_sign.iloc[:,1],filt_low,filt_high,filt_band]).T
df_filt.columns=['index','activity','low pass filter','high pass filter','bandfilter']


for j in range(2,5):
    for i in range(1,7):
        df_plot_filt = df_filt.groupby(by='activity',axis=0).get_group(i).iloc[0:,j].values
        df_plot_filt = df_plot_filt - np.mean(df_plot_filt)
        #df_plot_filt = df_plot_filt[0:5000]
        # Number of sample points
        N = len(df_plot_filt)
        # sample spacing (Hz)
        T = 1.0 / 50.0
        x = np.linspace(0.0, N*T, N, endpoint=False)
        yf = fft(df_plot_filt)
        xf = fftfreq(N, T)[:N//2]
        fig, ax7 = plt.subplots(2)
        ax7[0].plot(df_plot_filt[0:1500])
        plt.xlim([0,10])
        ax7[1].plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (-)')
        plt.suptitle(f'Plots regarding activity {i} and {df_filt.columns[j]}', fontsize=14)
        plt.show()
        
        
########################  DYNAMIC TIME WARPING AND CLASSIFICATION WITH K-NEAREST NEIGHBOURS #######################################
######### Question 4.8

#Define functions
plt.style.use('bmh')
# %matplotlib inline

try:
    from IPython.display import clear_output
    have_ipython = True
except ImportError:
    have_ipython = False

class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays
    
    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN
        
    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function
            
    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """
    
    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
    
    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels
        
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer
            
        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """
        
        self.x = x
        self.l = l
        
    def _dtw_distance(self, ts_a, ts_b, d = lambda x,y: abs(x-y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
        
        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function
        
        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window 
        return cost[-1, -1]
    
    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure
        
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
        
        y : array of shape [n_samples, n_timepoints]
        
        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """
        
        # Compute the distance matrix        
        dm_count = 0
        
        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if(np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
            
            p = ProgressBar(len(dm)[0])
            
            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])
                    
                    dm_count += 1
                    p.animate(dm_count)
            
            # Convert to squareform
            dm = squareform(dm)
            return dm
        
        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0])) 
            dm_size = x_s[0]*y_s[0]
            
            p = ProgressBar(dm_size)
        
            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)
        
            return dm
        
    def predict(self, x):
        """Predict the class labels or probability estimates for 
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified
          
        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels 
              (2) the knn label count probability
        """
        
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]
        
        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()

class ProgressBar:
    """This progress bar was taken from PYMC
    """
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        if have_ipython:
            self.animate = self.animate_ipython
        else:
            self.animate = self.animate_noipython

    def animate_ipython(self, iter):
        print ('\r', self,)
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

#### Derived features ####
# Import the HAR dataset
x_train_file = open('train/X_train.txt', 'r')
y_train_file = open('train/y_train.txt', 'r')

x_test_file = open('test/X_test.txt', 'r')
y_test_file = open('test/y_test.txt', 'r')

# Create empty lists
x_train = []
y_train = []
x_test = []
y_test = []

# Mapping table for classes
labels = {1:'WALKING', 2:'WALKING UPSTAIRS', 3:'WALKING DOWNSTAIRS',
          4:'SITTING', 5:'STANDING', 6:'LAYING'}

# Loop through datasets
for x in x_train_file:
    x_train.append([float(ts) for ts in x.split()])
    
for y in y_train_file:
    y_train.append(int(y.rstrip('\n')))
    
for x in x_test_file:
    x_test.append([float(ts) for ts in x.split()])
    
for y in y_test_file:
    y_test.append(int(y.rstrip('\n')))
    
# Convert to numpy for efficiency
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

#plot derived features
plt.figure(figsize=(11,7))
colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
          '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

for i, r in enumerate([0,27,65,100,145,172]):
    plt.subplot(3,2,i+1)
    plt.plot(x_train[r], label=labels[y_train[r]], color=colors[i], linewidth=2)
    plt.xlabel('Samples @50Hz')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
#Model performance
m = KnnDtw(n_neighbors=1, max_warping_window=10)
m.fit(x_train[::10], y_train[::10])
label, proba = m.predict(x_test[::10])

from sklearn.metrics import classification_report, confusion_matrix

print (classification_report(label, y_test[::10],
                            target_names=[l for l in labels.values()]))

conf_mat = confusion_matrix(label, y_test[::10])

fig = plt.figure(figsize=(6,6))
width = np.shape(conf_mat)[1]
height = np.shape(conf_mat)[0]

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c>0:
            plt.text(j-.2, i+.1, c, fontsize=16)
            
cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(6), [l for l in labels.values()], rotation=90)
_ = plt.yticks(range(6), [l for l in labels.values()])

#### Raw signals ####
#read raw signals
#read inertial signals train
acc_x_train = pd.read_fwf("train/Inertial Signals/body_acc_x_train.txt", header = None)
# acc_y_train = pd.read_fwf("train/Inertial Signals/body_acc_y_train.txt", header = None)
# acc_z_train = pd.read_fwf("train/Inertial Signals/body_acc_z_train.txt", header = None)
# gyro_x_train = pd.read_fwf("train/Inertial Signals/body_gyro_x_train.txt", header = None)
# gyro_y_train = pd.read_fwf("train/Inertial Signals/body_gyro_y_train.txt", header = None)
# gyro_z_train = pd.read_fwf("train/Inertial Signals/body_gyro_z_train.txt", header = None)
# Tacc_x_train = pd.read_fwf("train/Inertial Signals/total_acc_x_train.txt", header = None)
# Tacc_y_train = pd.read_fwf("train/Inertial Signals/total_acc_y_train.txt", header = None)
# Tacc_z_train = pd.read_fwf("train/Inertial Signals/total_acc_z_train.txt", header = None)

# #read inertial signals test
acc_x_test = pd.read_fwf("test/Inertial Signals/body_acc_x_test.txt", header = None)
# acc_y_test = pd.read_fwf("test/Inertial Signals/body_acc_y_test.txt", header = None)
# acc_z_test = pd.read_fwf("test/Inertial Signals/body_acc_z_test.txt", header = None)
# gyro_x_test = pd.read_fwf("test/Inertial Signals/body_gyro_x_test.txt", header = None)
# gyro_y_test = pd.read_fwf("test/Inertial Signals/body_gyro_y_test.txt", header = None)
# gyro_z_test = pd.read_fwf("test/Inertial Signals/body_gyro_z_test.txt", header = None)
# Tacc_x_test = pd.read_fwf("test/Inertial Signals/total_acc_x_test.txt", header = None)
# Tacc_y_test = pd.read_fwf("test/Inertial Signals/total_acc_y_test.txt", header = None)
# Tacc_z_test = pd.read_fwf("test/Inertial Signals/total_acc_z_test.txt", header = None)

# Convert to numpy for efficiency
acc_x_train = np.array(acc_x_train)
acc_x_test = np.array(acc_x_test)

#plot raw signals
plt.figure(figsize=(11,7))
colors = ['#D62728','#2C9F2C','#FD7F23','#1F77B4','#9467BD',
          '#8C564A','#7F7F7F','#1FBECF','#E377C2','#BCBD27']

for i, r in enumerate([0,27,65,100,145,172]):
    plt.subplot(3,2,i+1)
    plt.plot(acc_x_train[r], label=labels[y_train[r]], color=colors[i], linewidth=2)
    plt.xlabel('Samples @50Hz')
    plt.legend(loc='upper left')
    plt.tight_layout()
    
#Model performance
m = KnnDtw(n_neighbors=1, max_warping_window=10)
m.fit(acc_x_train[::10], y_train[::10])
label, proba = m.predict(acc_x_test[::10])

print (classification_report(label, y_test[::10],
                            target_names=[l for l in labels.values()]))

conf_mat = confusion_matrix(label, y_test[::10])

fig = plt.figure(figsize=(6,6))
width = np.shape(conf_mat)[1]
height = np.shape(conf_mat)[0]

res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
for i, row in enumerate(conf_mat):
    for j, c in enumerate(row):
        if c>0:
            plt.text(j-.2, i+.1, c, fontsize=16)
            
cb = fig.colorbar(res)
plt.title('Confusion Matrix')
_ = plt.xticks(range(6), [l for l in labels.values()], rotation=90)
_ = plt.yticks(range(6), [l for l in labels.values()])


######################## TIME SERIES COMPARISON AND PREDICTION ##########################################################
######### Question 4.9
#dateparse = lambda dates: datetime.strptime(dates, '%Y-%m-%d')#' %H:%M:%S')
#land_temp_country = pd.read_csv('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/GlobalLandTemperaturesByCountry.csv',parse_dates=['dt'],date_parser=dateparse,delimiter=',')
land_temp_country = pd.read_csv('C:/Users/Rutger/Documents/Master/Tijdvak6/Data Science/Assignments/TimeSeries/Data/GlobalLandTemperaturesByCountry.csv',delimiter=',')

# country = ['Norway','Finland','Singapore', 'Cambodia']
# def get_country(country):
#     t_country = land_temp_country.groupby(by="Country").get_group(f"{country}").copy() #copy to let pandas know it should not propagate into land_temp_country
#     t_country.iloc[:,0] = pd.to_datetime(t_country.iloc[:,0], infer_datetime_format=True)
#     t_country_avg = t_country.groupby(t_country['dt'].dt.year)['AverageTemperature'].mean()
#     return t_country_avg 

# t_avg=[]
# for x in country:
#     t_country_avg = get_country(x)
#     t_avg = t_avg.append(t_country_avg)
    
#Norway average yearly temperature
t_nor = land_temp_country.groupby(by="Country").get_group("Norway").copy() #copy to let pandas know it should not propagate into land_temp_country
t_nor['dt'] = pd.to_datetime(t_nor['dt'], infer_datetime_format=True)
t_nor_avg = pd.DataFrame(t_nor.groupby(t_nor['dt'].astype('datetime64[Y]'))['AverageTemperature'].mean()) ; t_nor_avg.columns=['Norway']
t_nor_m = pd.DataFrame(t_nor['AverageTemperature'])
t_nor_m = pd.DataFrame(t_nor['AverageTemperature']).set_index(t_nor['dt'])
#Finland average yearly temperature
t_fin = land_temp_country.groupby(by="Country").get_group("Finland").copy()
t_fin['dt'] = pd.to_datetime(t_fin['dt'], infer_datetime_format=True)
t_fin_avg = pd.DataFrame(t_fin.groupby(t_fin['dt'].astype('datetime64[Y]'))['AverageTemperature'].mean()) ; t_fin_avg.columns=["Finland"]
t_fin_m = pd.DataFrame(t_fin['AverageTemperature']).set_index(t_fin['dt'])
#Singapore average yearly temperature
t_sing = land_temp_country.groupby(by="Country").get_group("Singapore").copy()
t_sing['dt'] = pd.to_datetime(t_sing['dt'], infer_datetime_format=True)
t_sing_avg = pd.DataFrame(t_sing.groupby(t_sing['dt'].astype('datetime64[Y]'))['AverageTemperature'].mean()) ; t_sing_avg.columns=['Singapore']
t_sing_m = pd.DataFrame(t_sing['AverageTemperature']).set_index(t_sing['dt'])
#Cambodia average yearly temperature
t_cam = land_temp_country.groupby(by="Country").get_group("Cambodia").copy() 
t_cam['dt'] = pd.to_datetime(t_cam['dt'], infer_datetime_format=True)
t_cam_avg = pd.DataFrame(t_cam.groupby(t_cam['dt'].astype('datetime64[Y]'))['AverageTemperature'].mean()) ; t_cam_avg.columns=['Cambodia']
t_cam_m = pd.DataFrame(t_cam['AverageTemperature']).set_index(t_cam['dt'])

#Dataframe with all countries
df_years = pd.concat([t_nor_avg,t_fin_avg,t_sing_avg,t_cam_avg], axis=1, join='outer')
df_months = pd.concat([t_nor_m,t_fin_m,t_sing_m,t_cam_m], axis=1, join='outer')
df_months.columns=["Norway","Finland","Singapore","Cambodia"] 

    #a)

#plot:
fig, ax = plt.subplots()
plt.plot(pd.DatetimeIndex(df_years.index).year,df_years.iloc[:,0],label="Norway")
plt.plot(pd.DatetimeIndex(df_years.index).year,df_years.iloc[:,1],label="Finland")
plt.plot(pd.DatetimeIndex(df_years.index).year,df_years.iloc[:,2],label="Singapore")
plt.plot(pd.DatetimeIndex(df_years.index).year,df_years.iloc[:,3],label="Cambodia")
plt.legend(loc='best')
plt.title('Average yearly temperatures for different cities')
plt.xlabel("time (years)")
plt.xlim((1741,2014))
plt.ylabel("Average temperature (degrees)")

#DTW script, based on: https://github.com/kamperh/lecture_dtw_notebook/blob/main/dtw.ipynb
def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)

#setting right data type:
fin_m = np.array(np.array(df_months['Finland'].copy())); nor_m = np.array(np.array(df_months['Norway'].copy())) ; sing_m = np.array(np.array(df_months['Singapore'].copy())) ; cam_m=np.array(np.array(df_months['Cambodia'].copy())) 
#Remove NaN values:
fin_m = fin_m[~np.isnan(fin_m)] ; nor_m = nor_m[~np.isnan(nor_m)] ; sing_m = sing_m[~np.isnan(sing_m)] ; cam_m = cam_m[~np.isnan(cam_m)]
    
#compare to Finland
for cur_b in [nor_m, sing_m, cam_m]:        
    #check which country is it comapared to
    if all(item in cur_b for item in nor_m) == True:
        country_name = 'Norway'
    elif all(item in cur_b for item in sing_m) == True:
        country_name = 'Singapore'
    elif all(item in cur_b for item in cam_m) == True:
        country_name = 'Cambodia'
    # Distance matrix
    N = fin_m.shape[0]
    M = cur_b.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
    #         print(a[i], b[j], abs(a[i] - b[j]))
            dist_mat[i, j] = abs(fin_m[i] - cur_b[j])
    # DTW
    path, cost_mat = dp(dist_mat)
    print("Alignment cost for Finland compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]))
    print("Normalized alignment cost for Finland compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]/(N + M)))
    print()
    
#compare to Norway
for cur_b in [sing_m, cam_m]:
    #check which country is it comapared to
    if all(item in cur_b for item in sing_m) == True:
        country_name = 'Singapore'
    elif all(item in cur_b for item in cam_m) == True:
        country_name = 'Cambodia'
    # Distance matrix
    N = nor_m.shape[0]                                
    M = cur_b.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
    #         print(a[i], b[j], abs(a[i] - b[j]))
            dist_mat[i, j] = abs(nor_m[i] - cur_b[j])
    # DTW
    path, cost_mat = dp(dist_mat)
    print("Alignment cost for Norway compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]))
    print("Normalized alignment cost for Norway compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]/(N + M)))
    print()
    
#compare Cambodia to Singapore
for cur_b in [sing_m]:
    country_name = 'Singapore'
    # Distance matrix
    N = cam_m.shape[0]                                
    M = cur_b.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
    #         print(a[i], b[j], abs(a[i] - b[j]))
            dist_mat[i, j] = abs(cam_m[i] - cur_b[j])
    # DTW
    path, cost_mat = dp(dist_mat)
    print("Alignment cost for cambodia compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]))
    print("Normalized alignment cost for Cambodia compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]/(N + M)))
    print()

    #b)

def test_stationarity(timeseries,country):  
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test for the ' f'{country}' ' case:')
    dftest = adfuller(timeseries, autolag='AIC') #dataseries to test, 
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


#stationarity of monthly values.
test_stationarity(fin_m,'Finland') #test stationarity of an array with Dickey-Fuller test (change name of country)
test_stationarity(nor_m,'Norway') #test stationarity of an array with Dickey-Fuller test (change name of country)
test_stationarity(sing_m,'Singapore') #test stationarity of an array with Dickey-Fuller test (change name of country)
test_stationarity(cam_m,'Cambodia') #test stationarity of an array with Dickey-Fuller test (change name of country)



    #c)
yavg = [t_fin_m,t_nor_m,t_sing_m, t_cam_m]
    
t_residual = []
for x in range(len(yavg)):
    #### DETREND AND REMOVE SEASONALITY #####         
    decomposition = seasonal_decompose(yavg[x].dropna(),period=12) #determine decomposition for every country
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    ress = residual.values.tolist()
    ress = [x for x in ress if str(x) != 'nan']  #nan string, not NaN
    t_residual.append(ress) #index 0: Finland, 1:Norway, 2:Singapore, 3:Cambodia

####### DTW on detrended data! #######
#compare to Finland
for cur_b in [t_residual[1], t_residual[2], t_residual[3]]: 
    #check which country is it comapared to
    if all(item in cur_b for item in t_residual[1]) == True:
        country_name = 'Norway'
    elif all(item in cur_b for item in t_residual[2]) == True:
        country_name = 'Singapore'
    elif all(item in cur_b for item in t_residual[3]) == True:
        country_name = 'Cambodia'
    # Distance matrix
    N = len(t_residual[0])
    M = len(cur_b)
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
    #         print(a[i], b[j], abs(a[i] - b[j]))
            dist_mat[i, j] = abs(t_residual[0][i] - cur_b[j])
    # DTW
    path, cost_mat = dp(dist_mat)
    print("---Removed trend and seasonality--- Alignment cost for Finland compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]))
    print("---Removed trend and seasonality--- Normalized alignment cost for Finland compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]/(N + M)))
    print()
    
#compare to Norway
for cur_b in [t_residual[2], t_residual[3]]:
    #check which country is it comapared to
    if all(item in cur_b for item in t_residual[2]) == True:
        country_name = 'Singapore'
    elif all(item in cur_b for item in t_residual[3]) == True:
        country_name = 'Cambodia'
    # Distance matrix
    N = len(t_residual[1])                             
    M = len(cur_b)
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
    #         print(a[i], b[j], abs(a[i] - b[j]))
            dist_mat[i, j] = abs(t_residual[1][i] - cur_b[j])
    # DTW
    path, cost_mat = dp(dist_mat)
    print("---Removed trend and seasonality--- Alignment cost for Norway compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]))
    print("---Removed trend and seasonality--- Normalized alignment cost for Norway compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]/(N + M)))
    print()
    
#compare Cambodia to Singapore
for cur_b in [t_residual[2]]:
    country_name = 'Singapore'
    # Distance matrix
    N = len(t_residual[2])                               
    M = len(cur_b)
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
    #         print(a[i], b[j], abs(a[i] - b[j]))
            dist_mat[i, j] = abs(t_residual[3][i] - cur_b[j])
    # DTW
    path, cost_mat = dp(dist_mat)
    print("---Removed trend and seasonality--- Alignment cost for Cambodia compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]))
    print("---Removed trend and seasonality--- Normalized alignment cost for Cambodia compared with " f'{country_name}' ": {:.4f}".format(cost_mat[N - 1, M - 1]/(N + M)))
    print()

#stationarity of monthly detrended values.
test_stationarity(t_residual[0],'Finland') #test stationarity of an array with Dickey-Fuller test (change name of country)
test_stationarity(t_residual[1],'Norway') #test stationarity of an array with Dickey-Fuller test (change name of country)
test_stationarity(t_residual[2],'Singapore') #test stationarity of an array with Dickey-Fuller test (change name of country)
test_stationarity(t_residual[3],'Cambodia') #test stationarity of an array with Dickey-Fuller test (change name of country)

    #d)
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA

#differencing first order on Finland data
#tt_fin = pd.DataFrame(df_months['Finland'].copy()) ; tt_fin = tt_fin[~np.isnan(tt_fin)]
tt_fin = df_months['Finland'].copy() ; tt_fin = pd.DataFrame(tt_fin[~np.isnan(tt_fin)])
td_fin = tt_fin - tt_fin.shift(12) #first order
#td_fin = tt_fin - 2*tt_fin.shift(1) + tt_fin.shift(2) #second order
plt.plot(td_fin)
td_fin.dropna(inplace=True)
test_stationarity(td_fin,'Finland')

lag_acf = acf(td_fin, nlags=20,fft=False)
lag_pacf = pacf(td_fin, nlags=20, method='ols')

#visualise change of differencing technique
fig, ax = plt.subplots()
plt.plot(pd.DatetimeIndex(tt_fin.index),tt_fin.iloc[:,0],label="Original")
plt.plot(pd.DatetimeIndex(td_fin.index),td_fin.iloc[:,0],label="Differenced")
plt.legend(loc='best')
plt.title('Effect of differencing technique for Finland case')
plt.xlabel("time (years)")
plt.xlim(('1998-08-01 00:00:00','2010-08-01 00:00:00'))
plt.ylabel("Average temperature (degrees)")

#Plot ACF: 
ax = plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(td_fin)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(td_fin)),linestyle='--',color='gray')
plt.xlim(0,20)
plt.grid(b=True, which='both', color='0.65',linestyle='-')

plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(td_fin)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(td_fin)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.grid(b=True, which='both', color='0.65',linestyle='-')
plt.xlim(0,20)
plt.tight_layout()

### AR MODEL ####
model = ARIMA(tt_fin, order=(2, 1, 0))  #p,d,q
results_AR = model.fit()  
fig, ax = plt.subplots()
plt.plot(td_fin, label='Original data')
plt.plot(results_AR.fittedvalues, color='red', label='AR model')
plt.legend(loc='best')
plt.title('Original data vs. AR model for Finland case')
plt.title('standardized RSS for AR model: %.4f'% ((sum((results_AR.fittedvalues-tt_fin.iloc[1:,0])**2))))
plt.xlabel("time (years)")
plt.xlim(('1998-08-01 00:00:00','2010-08-01 00:00:00'))
plt.ylabel("Average temperature (degrees)")

### MA MODEL ####
model = ARIMA(tt_fin, order=(0, 1, 3))  #p,d,q
results_MA = model.fit()  
fig, ax = plt.subplots()
plt.plot(td_fin, label='Original data')
plt.plot(results_MA.fittedvalues, color='red', label='MA model')
plt.legend(loc='best')
plt.title('Original data vs. MA model for Finland case')
plt.title('standardized RSS for MA model: %.4f'% ((sum((results_MA.fittedvalues-tt_fin.iloc[1:,0])**2))))
plt.xlabel("time (years)")
plt.xlim(('1998-08-01 00:00:00','2010-08-01 00:00:00'))
plt.ylabel("Average temperature (degrees)")

from statsmodels.tsa.arima.model import ARIMA
### ARIMA MODEL ####
model = ARIMA(tt_fin, order=(2, 1, 3))#,enforce_invertibility=True)  #p,d,q
results_ARIMA = model.fit()  
fig, ax = plt.subplots()
plt.plot(td_fin, label='Original data')
plt.plot(results_ARIMA.fittedvalues, color='red', label='ARIMA model')
plt.legend(loc='best')
plt.title('Original data vs. ARIMA model for Finland case')
plt.title('standardized RSS for ARIMA model: %.4f'% ((sum((results_ARIMA.fittedvalues-tt_fin.iloc[1:,0])**2))))
plt.xlabel("time (years)")
plt.xlim(('1998-08-01 00:00:00','2010-08-01 00:00:00'))
plt.ylabel("Average temperature (degrees)")
    

    #e) (based on:https://machinelearningmastery.com/make-sample-forecasts-arima-python/)
    
    # invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

model_test = ARIMA(tt_fin.iloc[0:3159,0], order=(2, 1, 0))  #p,d,q #perform model without last 7 days
results_AR2 = model_test.fit()  

tt_test = results_AR2.predict() #fitted values of AR model
#tt_test['fitted']=results_AR2.fittedvalues

series = tt_fin.iloc[0:3159,0] 
X = series.values
days_in_year = 365

# 7 days forecast
start_index = len(tt_test) #can also be date type
end_index = len(tt_test) + 6 #7 day forecast
forecast = results_AR2.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast: 
	inverted = inverse_difference(history, yhat, days_in_year) #this
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1

fig, ax = plt.subplots()
plt.plot(pd.DatetimeIndex(tt_fin.index),tt_fin['Finland'], label='Original data')
plt.plot(pd.DatetimeIndex(tt_fin.index)[len(history)-8:len(history)],history[len(history)-8:len(history)], color='red', label='AR model 7m prediction')
plt.plot(pd.DatetimeIndex(tt_fin.index)[1:],results_AR.fittedvalues, color='blue', label='Fitted AR model')
plt.legend(loc='best')
plt.title('Original data vs. AR model fitted and prediction for Finland case')
plt.xlabel("time (years)")
plt.xlim(('2012-10-01 00:00:00','2013-08-01 00:00:00'))
plt.xticks(rotation=60)
plt.ylabel("Average temperature (degrees)")

##TEST ARIMA

def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

model_test2 = ARIMA(tt_fin.iloc[0:3159,0], order=(2, 1, 3))  #p,d,q #perform model without last 7 days
results_ARIMA2 = model_test2.fit()  

tt_test2 = results_ARIMA2.predict() #fitted values of AR model
#tt_test['fitted']=results_AR2.fittedvalues

series = tt_fin.iloc[0:3159,0] 
X = series.values
days_in_year = 365

# 7 days forecast
start_index = len(tt_test) #can also be date type
end_index = len(tt_test) + 6 #7 day forecast
forecast = results_ARIMA2.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable
history2 = [x for x in X]
day = 1
for yhat in forecast: 
	inverted = inverse_difference(history2, yhat, days_in_year) #this
	print('Day %d: %f' % (day, inverted))
	history2.append(inverted)
	day += 1

fig, ax = plt.subplots()
plt.plot(pd.DatetimeIndex(tt_fin.index),tt_fin['Finland'], label='Original data')
plt.plot(pd.DatetimeIndex(tt_fin.index)[len(history2)-8:len(history2)],history2[len(history2)-8:len(history2)], color='red', label='AR model 7m prediction')
plt.plot(pd.DatetimeIndex(tt_fin.index)[7:],results_ARIMA2.fittedvalues, color='blue', label='Fitted ARIMA model')
plt.legend(loc='best')
plt.title('Original data vs. ARIMA model fitted and prediction for Finland case')
plt.xlabel("time (years)")
plt.xlim(('2012-10-01 00:00:00','2013-08-01 00:00:00'))
plt.xticks(rotation=60)
plt.ylabel("Average temperature (degrees)")
