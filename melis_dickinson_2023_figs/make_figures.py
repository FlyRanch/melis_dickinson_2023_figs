import numpy as np
import scipy.io
import numpy.matlib
import os
import h5py
import math
import csv
from copy import copy

from scipy import linalg
import scipy.special
from scipy.signal import find_peaks
from scipy.interpolate import BSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
import scipy.special
from scipy import stats
from scipy.linalg import eigh
from scipy.stats import gaussian_kde
from scipy.stats import zscore

from .mpl_functions import adjust_spines

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter
from matplotlib.gridspec import GridSpec

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from sklearn import manifold, datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import manifold

import tensorflow as tf

from keras.layers import Dense, Input, Concatenate, Reshape, Lambda
from keras.layers import Conv1D, Flatten, MaxPooling1D, Activation, Dropout, GaussianNoise, BatchNormalization, LayerNormalization
from keras.layers import Conv2D, Conv2DTranspose, ThresholdedReLU, UpSampling2D, SeparableConv2D, DepthwiseConv2D, GlobalAveragePooling2D
from keras.constraints import maxnorm
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, kullback_leibler_divergence
from keras.regularizers import l1, l2, l1_l2
from keras.utils import np_utils
#from keras.utils.all_utils import Sequence
from keras import optimizers
from keras import backend as K

import seaborn as sns;

class FigureGen():

    def __init__(self):
        self.dt = 1.0/15000.0
        self.N_frames = 16375
        self.N_pol_theta = 20
        self.N_pol_eta      = 24
        self.N_pol_phi      = 16
        self.N_pol_xi      = 20
        self.N_const      = 3
        self.muscle_names = ['b1','b2','b3','i1','i2','iii1','iii24','iii3','hg1','hg2','hg3','hg4','freq']
        self.N_window      = 9
        self.g = 9800.0
        self.R_fly = 2.7
        self.f = 200.0

    def set_plot_loc(self,plot_loc_in):
        self.plot_loc = plot_loc_in

    def load_dataset(self,file_name):
        self.data_file = h5py.File(file_name,'r')
        self.data_keys = list(self.data_file.keys())
        self.N_files = len(self.data_keys)
        print('Number of files: {self.N_files}')
        # data lists:
        self.a_theta_L_mov = []
        self.a_eta_L_mov = []
        self.a_phi_L_mov = []
        self.a_xi_L_mov = []
        self.a_x_L_mov = []
        self.a_y_L_mov = []
        self.a_z_L_mov = []
        self.a_theta_R_mov = []
        self.a_eta_R_mov = []
        self.a_phi_R_mov = []
        self.a_xi_R_mov = []
        self.a_x_R_mov = []
        self.a_y_R_mov = []
        self.a_z_R_mov = []
        self.s1_s2_mov = []
        self.T_mov = []
        self.time_wb_mov = []
        self.freq_mov = []
        self.t_wbs_mov = []
        self.b1_wbs = []
        self.b2_wbs = []
        self.b3_wbs = []
        self.i1_wbs = []
        self.i2_wbs = []
        self.iii1_wbs = []
        self.iii24_wbs = []
        self.iii3_wbs = []
        self.hg1_wbs = []
        self.hg2_wbs = []
        self.hg3_wbs = []
        self.hg4_wbs = []
        for i in range(self.N_files):
            print('file nr: '+str(i+1))
            mov_group = self.data_file[self.data_keys[i]]
            n_triggers = np.squeeze(np.copy(mov_group['N_triggers']))
            m_keys = list(mov_group.keys())
            mov_keys = [k for k in m_keys if 'mov_' in k]
            n_mov = len(mov_keys)
            for j in range(n_mov):
                a_theta_L_i = np.copy(mov_group[mov_keys[j]]['a_theta_L'])
                a_eta_L_i = np.copy(mov_group[mov_keys[j]]['a_eta_L'])
                a_phi_L_i = np.copy(mov_group[mov_keys[j]]['a_phi_L'])
                a_xi_L_i = np.copy(mov_group[mov_keys[j]]['a_xi_L'])
                a_x_L_i = np.copy(mov_group[mov_keys[j]]['a_x_L'])
                a_y_L_i = np.copy(mov_group[mov_keys[j]]['a_y_L'])
                a_z_L_i = np.copy(mov_group[mov_keys[j]]['a_z_L'])
                a_theta_R_i = np.copy(mov_group[mov_keys[j]]['a_theta_R'])
                a_eta_R_i = np.copy(mov_group[mov_keys[j]]['a_eta_R'])
                a_phi_R_i = np.copy(mov_group[mov_keys[j]]['a_phi_R'])
                a_xi_R_i = np.copy(mov_group[mov_keys[j]]['a_xi_R'])
                a_x_R_i = np.copy(mov_group[mov_keys[j]]['a_x_R'])
                a_y_R_i = np.copy(mov_group[mov_keys[j]]['a_y_R'])
                a_z_R_i = np.copy(mov_group[mov_keys[j]]['a_z_R'])
                s1_s2_i = np.copy(mov_group[mov_keys[j]]['s1_s2'])
                T_i = np.copy(mov_group[mov_keys[j]]['T'])
                time_wb_i = np.copy(mov_group[mov_keys[j]]['time_wb'])
                freq_i = np.copy(mov_group[mov_keys[j]]['freq'])
                t_wbs_i = np.copy(mov_group[mov_keys[j]]['t_wbs'])
                if a_theta_L_i.size==0:
                    print('mov nr: '+str(j+1))
                    print('error empty array')
                else:
                    self.a_theta_L_mov.append(a_theta_L_i)
                    self.a_eta_L_mov.append(a_eta_L_i)
                    self.a_phi_L_mov.append(a_phi_L_i)
                    self.a_xi_L_mov.append(a_xi_L_i)
                    self.a_x_L_mov.append(a_x_L_i)
                    self.a_y_L_mov.append(a_y_L_i)
                    self.a_z_L_mov.append(a_z_L_i)
                    self.a_theta_R_mov.append(a_theta_R_i)
                    self.a_eta_R_mov.append(a_eta_R_i)
                    self.a_phi_R_mov.append(a_phi_R_i)
                    self.a_xi_R_mov.append(a_xi_R_i)
                    self.a_x_R_mov.append(a_x_R_i)
                    self.a_y_R_mov.append(a_y_R_i)
                    self.a_z_R_mov.append(a_z_R_i)
                    self.s1_s2_mov.append(s1_s2_i)
                    self.T_mov.append(T_i)
                    self.time_wb_mov.append(time_wb_i)
                    self.freq_mov.append(freq_i)
                    self.t_wbs_mov.append(t_wbs_i)
                    b1_i = np.copy(mov_group[mov_keys[j]]['b1_wbs'])
                    b2_i = np.copy(mov_group[mov_keys[j]]['b2_wbs'])
                    b3_i = np.copy(mov_group[mov_keys[j]]['b3_wbs'])
                    i1_i = np.copy(mov_group[mov_keys[j]]['i1_wbs'])
                    i2_i = np.copy(mov_group[mov_keys[j]]['i2_wbs'])
                    iii1_i = np.copy(mov_group[mov_keys[j]]['iii1_wbs'])
                    iii24_i = np.copy(mov_group[mov_keys[j]]['iii24_wbs'])
                    iii3_i = np.copy(mov_group[mov_keys[j]]['iii3_wbs'])
                    hg1_i = np.copy(mov_group[mov_keys[j]]['hg1_wbs'])
                    hg2_i = np.copy(mov_group[mov_keys[j]]['hg2_wbs'])
                    hg3_i = np.copy(mov_group[mov_keys[j]]['hg3_wbs'])
                    hg4_i = np.copy(mov_group[mov_keys[j]]['hg4_wbs'])
                    self.b1_wbs.append(b1_i)
                    self.b2_wbs.append(b2_i)
                    self.b3_wbs.append(b3_i)
                    self.i1_wbs.append(i1_i)
                    self.i2_wbs.append(i2_i)
                    self.iii1_wbs.append(iii1_i)
                    self.iii24_wbs.append(iii24_i)
                    self.iii3_wbs.append(iii3_i)
                    self.hg1_wbs.append(hg1_i)
                    self.hg2_wbs.append(hg2_i)
                    self.hg3_wbs.append(hg3_i)
                    self.hg4_wbs.append(hg4_i)
        self.data_file.close()

    def create_dataset(self,outliers):
        self.a_mod_theta_L = []
        self.a_mod_eta_L = []
        self.a_mod_phi_L = []
        self.a_mod_xi_L = []
        self.a_mod_x_L = []
        self.a_mod_y_L = []
        self.a_mod_z_L = []
        self.a_mod_theta_R = []
        self.a_mod_eta_R = []
        self.a_mod_phi_R = []
        self.a_mod_xi_R = []
        self.a_mod_x_R = []
        self.a_mod_y_R = []
        self.a_mod_z_R = []
        self.T_vec = []
        self.freq_vec = []
        self.ca_traces = []
        self.ca_mod_traces = []
        self.ca_spline = []
        self.N_movs_total = len(self.a_theta_L_mov)
        print('total number of movies: '+str(self.N_movs_total))
        for i in range(self.N_movs_total):
            self.a_mod_theta_L.append(np.gradient(self.a_theta_L_mov[i],axis=1))
            self.a_mod_eta_L.append(np.gradient(self.a_eta_L_mov[i],axis=1))
            self.a_mod_phi_L.append(np.gradient(self.a_phi_L_mov[i],axis=1))
            self.a_mod_xi_L.append(np.gradient(self.a_xi_L_mov[i],axis=1))
            self.a_mod_x_L.append(np.gradient(self.a_x_L_mov[i],axis=1))
            self.a_mod_y_L.append(np.gradient(self.a_y_L_mov[i],axis=1))
            self.a_mod_z_L.append(np.gradient(self.a_z_L_mov[i],axis=1))
            self.a_mod_theta_R.append(np.gradient(self.a_theta_R_mov[i],axis=1))
            self.a_mod_eta_R.append(np.gradient(self.a_eta_R_mov[i],axis=1))
            self.a_mod_phi_R.append(np.gradient(self.a_phi_R_mov[i],axis=1))
            self.a_mod_xi_R.append(np.gradient(self.a_xi_R_mov[i],axis=1))
            self.a_mod_x_R.append(np.gradient(self.a_x_R_mov[i],axis=1))
            self.a_mod_y_R.append(np.gradient(self.a_y_R_mov[i],axis=1))
            self.a_mod_z_R.append(np.gradient(self.a_z_R_mov[i],axis=1))
            n_wbs_i = self.a_theta_L_mov[i].shape[1]
            #ca_traces_i = np.zeros((12,n_wbs_i))
            ca_traces_i = np.zeros((13,n_wbs_i))
            ca_traces_i[0,:] = self.b1_wbs[i]
            ca_traces_i[1,:] = self.b2_wbs[i]
            ca_traces_i[2,:] = self.b3_wbs[i]
            ca_traces_i[3,:] = self.i1_wbs[i]
            ca_traces_i[4,:] = self.i2_wbs[i]
            ca_traces_i[5,:] = self.iii1_wbs[i]
            ca_traces_i[6,:] = self.iii24_wbs[i]
            ca_traces_i[7,:] = self.iii3_wbs[i]
            ca_traces_i[8,:] = self.hg1_wbs[i]
            ca_traces_i[9,:] = self.hg2_wbs[i]
            ca_traces_i[10,:] = self.hg3_wbs[i]
            ca_traces_i[11,:] = self.hg4_wbs[i]
            ca_traces_i[12,:] = self.freq_mov[i]
            self.ca_traces.append(ca_traces_i)
            ca_mod_traces_i = np.diff(ca_traces_i,axis=1)
            self.ca_mod_traces.append(ca_mod_traces_i)
            # spline smoothing
            ca_spline_smooth = self.spline_fit(ca_traces_i,0.2)
            self.ca_spline.append(ca_spline_smooth)
        # create dataset
        self.X_data_list = []
        self.Y_data_list = []
        self.X_mean_list = []
        self.X_gradient_list = []
        train_inds_list = []
        test_inds_list  = []
        wb_cntr = 0
        for i in range(self.N_movs_total):
            n_wbs_i = self.a_theta_L_mov[i].shape[1]
            n_half = int(np.floor(self.N_window))
            N_i = n_wbs_i-2*n_half
            if N_i>2:
                if np.sum(outliers[:,0]==i)<1:
                    X_i = np.zeros((N_i,self.N_window,13))
                    Y_i = np.zeros((N_i,80))
                    x_i = np.transpose(self.ca_traces[i])
                    x_i = self.Muscle_scale(x_i)
                    y_i = np.transpose(np.concatenate((self.a_theta_L_mov[i],self.a_eta_L_mov[i],self.a_phi_L_mov[i],self.a_xi_L_mov[i]),axis=0))
                    y_i = self.Wingkin_scale(y_i)
                    for j in range(N_i):
                        X_i[j,:,:] = x_i[j:(j+self.N_window),:]
                        Y_i[j,:] = y_i[j,:]
                        # check if frequency is within range
                        if self.freq_mov[i][j]>150 and self.freq_mov[i][j]<250:
                            # Check if wing kinematics are within range
                            wingkin_max = np.zeros(4)
                            wingkin_max[0] = (np.amax(np.abs(self.a_theta_L_mov[i][:,j]))>((60/180)*np.pi))
                            wingkin_max[1] = (np.amax(np.abs(self.a_eta_L_mov[i][:,j]))>((150/180)*np.pi))
                            wingkin_max[2] = (np.amax(np.abs(self.a_phi_L_mov[i][:,j]))>((120/180)*np.pi))
                            wingkin_max[3] = (np.amax(np.abs(self.a_xi_L_mov[i][:,j]))>((90/180)*np.pi))
                            if np.sum(wingkin_max)<1:
                                if j<30:
                                    # first 30 wingbeats are validation
                                    test_inds_list.append(wb_cntr)
                                else:
                                    # remaining wingbeats are training
                                    train_inds_list.append(wb_cntr)
                        wb_cntr = wb_cntr+1
                else:
                    outlier_ind = int(np.squeeze(np.argwhere(outliers[:,0]==i)))
                    X_i = np.zeros((N_i,self.N_window,13))
                    Y_i = np.zeros((N_i,80))
                    x_i = np.transpose(self.ca_traces[i])
                    x_i = self.Muscle_scale(x_i)
                    y_i = np.transpose(np.concatenate((self.a_theta_L_mov[i],self.a_eta_L_mov[i],self.a_phi_L_mov[i],self.a_xi_L_mov[i]),axis=0))
                    y_i = self.Wingkin_scale(y_i)
                    for j in range(N_i):
                        X_i[j,:,:] = x_i[j:(j+self.N_window),:]
                        Y_i[j,:] = y_i[j,:]
                        # check if frequency is within range
                        if j<outliers[outlier_ind,1] or j>outliers[outlier_ind,2]:
                            if self.freq_mov[i][j]>150 and self.freq_mov[i][j]<250:
                                # Check if wing kinematics are within range
                                wingkin_max = np.zeros(4)
                                wingkin_max[0] = (np.amax(np.abs(self.a_theta_L_mov[i][:,j]))>((60/180)*np.pi))
                                wingkin_max[1] = (np.amax(np.abs(self.a_eta_L_mov[i][:,j]))>((150/180)*np.pi))
                                wingkin_max[2] = (np.amax(np.abs(self.a_phi_L_mov[i][:,j]))>((120/180)*np.pi))
                                wingkin_max[3] = (np.amax(np.abs(self.a_xi_L_mov[i][:,j]))>((90/180)*np.pi))
                                if np.sum(wingkin_max)<1:
                                    if j<30:
                                        # first 30 wingbeats are validation
                                        test_inds_list.append(wb_cntr)
                                    else:
                                        # remaining wingbeats are training
                                        train_inds_list.append(wb_cntr)
                        wb_cntr = wb_cntr+1
                self.X_data_list.append(X_i)
                self.Y_data_list.append(Y_i)
                self.X_mean_list.append(np.mean(X_i,axis=1))
                self.X_gradient_list.append(np.mean(np.diff(X_i,axis=1),axis=1))
        self.X_data = np.concatenate(self.X_data_list,axis=0)
        self.Y_data = np.concatenate(self.Y_data_list,axis=0)
        self.X_mean = np.concatenate(self.X_mean_list,axis=0)
        self.X_gradient = np.concatenate(self.X_gradient_list,axis=0)
        print(self.X_data.shape)
        print(self.Y_data.shape)
        # Create training and testing set:
        self.N_wbs = self.Y_data.shape[0]
        self.unshuffled_inds = np.copy(np.array(train_inds_list))
        wb_ids_train = np.array(train_inds_list)
        np.random.shuffle(wb_ids_train)
        wb_ids_test = np.array(test_inds_list)
        np.random.shuffle(wb_ids_test)
        self.train_inds = wb_ids_train
        self.N_train     = self.train_inds.shape[0]
        print('N train: '+str(self.N_train))
        self.test_inds  = wb_ids_test
        self.N_test     = self.test_inds.shape[0]
        print('N test: '+str(self.N_test))
        self.N_test     = self.test_inds.shape[0]
        self.X_train     = self.X_data[self.train_inds,:,:]
        self.X_mean_train = self.X_mean[self.train_inds,:]
        self.Y_train     = self.Y_data[self.train_inds,:]
        self.X_test     = self.X_data[self.test_inds,:,:]
        self.X_mean_test = self.X_mean[self.test_inds,:]
        self.Y_test     = self.Y_data[self.test_inds,:]

    def Muscle_scale(self,X_in):
        X_out = X_in
        X_out[:,:12] = np.clip(X_in[:,:12],-0.5,1.5)
        X_out[:,12] = (np.clip(X_in[:,12],150.0,250.0)-150.0)/100.0
        return X_out

    def Muscle_scale_inverse(self,X_in):
        X_out = X_in
        X_out[:,:12] = X_in[:,:12]
        X_out[:,12] = X_in[:,12]*100.0+150.0
        return X_out

    def Wingkin_scale(self,X_in):
        X_out = (1.0/np.pi)*np.clip(X_in,-np.pi,np.pi)
        #X_out = np.clip(X_in,-np.pi,np.pi)
        return X_out

    def Wingkin_scale_inverse(self,X_in):
        X_out = X_in*np.pi
        return X_out

    def LegendreFit(self,trace_in,b1_in,b2_in,N_pol,N_const):
        N_pts = trace_in.shape[0]
        X_Legendre = self.LegendrePolynomials(N_pts,N_pol,N_const)
        A = X_Legendre[:,:,0]
        B = np.zeros((2*N_const,N_pol))
        # data points:
        b = np.transpose(trace_in)
        # restriction vector (add zeros to smooth the connection!!!!!)
        d = np.zeros(2*N_const)
        d_gradient_1 = b1_in
        d_gradient_2 = b2_in
        for j in range(N_const):
            d[j] = d_gradient_1[4-j]*np.power(N_pts/2.0,j)
            d[N_const+j] = d_gradient_2[4-j]*np.power(N_pts/2.0,j)
            d_gradient_1 = np.diff(d_gradient_1)
            d_gradient_2 = np.diff(d_gradient_2)
            B[j,:]             = np.transpose(X_Legendre[0,:,j])
            B[N_const+j,:]     = np.transpose(X_Legendre[-1,:,j])
        # Restricted least-squares fit:
        ATA = np.dot(np.transpose(A),A)
        ATA_inv = np.linalg.inv(ATA)
        AT = np.transpose(A)
        BT = np.transpose(B)
        BATABT     = np.dot(B,np.dot(ATA_inv,BT))
        c_ls     = np.linalg.solve(ATA,np.dot(AT,b))
        c_rls     = c_ls-np.dot(ATA_inv,np.dot(BT,np.linalg.solve(BATABT,np.dot(B,c_ls)-d)))
        return c_rls

    def LegendrePolynomials(self,N_pts,N_pol,n_deriv):
        L_basis = np.zeros((N_pts,N_pol,n_deriv))
        x_basis = np.linspace(-1.0,1.0,N_pts,endpoint=True)
        for i in range(n_deriv):
            if i==0:
                # Legendre basis:
                for n in range(N_pol):
                    if n==0:
                        L_basis[:,n,i] = 1.0
                    elif n==1:
                        L_basis[:,n,i] = x_basis
                    else:
                        for k in range(n+1):
                            L_basis[:,n,i] += (1.0/np.power(2.0,n))*np.power(scipy.special.binom(n,k),2)*np.multiply(np.power(x_basis-1.0,n-k),np.power(x_basis+1.0,k))
            else:
                # Derivatives:
                for n in range(N_pol):
                    if n>=i:
                        L_basis[:,n,i] = n*L_basis[:,n-1,i-1]+np.multiply(x_basis,L_basis[:,n-1,i])
        return L_basis

    def TemporalBC(self,a_c,N_pol,N_const):
        X_Legendre = self.LegendrePolynomials(30,N_pol,N_const)
        trace = np.dot(X_Legendre[:,:,0],a_c)
        b_L = np.zeros(9)
        b_L[0:4] = trace[-5:-1]
        b_L[4] = 0.5*(trace[0]+trace[-1])
        b_L[5:9] = trace[1:5]
        b_R = np.zeros(9)
        b_R[0:4] = trace[-5:-1]
        b_R[4] = 0.5*(trace[0]+trace[-1])
        b_R[5:9] = trace[1:5]
        c_per = self.LegendreFit(trace,b_L,b_R,N_pol,N_const)
        return c_per

    def predict(self,X_in):
        prediction = self.network.predict(X_in)
        return prediction

    def encode(self,X_in):
        encoded = self.encoder_network.predict(X_in)
        return encoded

    def decode(self,X_in):
        decoded = self.decoder_network.predict(X_in)
        return decoded

    def spline_fit(self,data_in,smoothing):
        n = data_in.shape[1]
        n_range = np.arange(n)
        m = data_in.shape[0]
        fit_out = np.zeros(data_in.shape)
        for j in range(m):
            tck = interpolate.splrep(n_range,data_in[j,:],s=smoothing)
            ynew = interpolate.splev(n_range, tck, der=0)
            fit_out[j,:] = ynew
        return fit_out

    def build_network(self,N_filters):
        l1_norm = 0.01
        l2_norm = 0.01

        input_enc = Input(shape=(self.N_window,13,1))
        enc = GaussianNoise(0.05)(input_enc)
        enc = Conv2D(filters=N_filters,kernel_size=(self.N_window,1),strides=(self.N_window,1),activation='selu')(enc)
        enc = Conv2D(filters=N_filters*4,kernel_size=(1,13),strides=(1,13),activation='selu')(enc)
        encoded = Flatten()(enc)
        input_dec = Input(shape=(N_filters*4,))
        dec = Dense(1024,activation='selu')(input_dec)
        decoded = Dense(80,activation='linear')(dec)
        encoder_model = Model(input_enc, encoded)
        decoder_model = Model(input_dec, decoded)
        auto_input = Input(shape=(self.N_window,13,1))
        encoded = encoder_model(auto_input)
        decoded = decoder_model(encoded)
        model = Model(auto_input, decoded)
        return model, encoder_model, decoder_model

    def load_network(self, N_filters, weights_fldr):
        weight_file = weights_fldr / 'muscle_wing_weights_new.h5'
        weight_file_2 = weights_fldr / 'muscle_encoder_weights_new.h5'
        weight_file_3 = weights_fldr / 'wingkin_decoder_weights_new.h5'
        self.network, self.encoder_network, self.decoder_network = self.build_network(N_filters)
        self.network.load_weights(weight_file)
        self.network.summary()
        self.encoder_network.load_weights(weight_file_2)
        self.encoder_network.summary()
        self.decoder_network.load_weights(weight_file_3)
        self.decoder_network.summary()

    def muscle_pca_plot(self):

        n_comp = 13
        X_plus     = self.X_data[:,2,:]

        pca = PCA(n_components=n_comp,svd_solver='full')
        Y_pca = pca.fit_transform(self.X_data[:,2,:])
        print(Y_pca)

        pca_mean = pca.mean_
        print(pca_mean)
        pca_comp = pca.components_
        print(pca_comp)
        pca_exp_var = pca.explained_variance_ratio_
        print(pca_exp_var)
        print(np.sum(pca_exp_var))

        fig, axs = plt.subplots(2,1)
        axs[0].scatter(Y_pca[:,1],Y_pca[:,0],s=0.5,c='k',marker='.')
        axs[0].set_xlim([-1.5,1.5])
        axs[0].set_ylim([-1.5,1.5])


    def figures_4(self):

        m_mean = np.array([0.5,0.1,0.5,0.35,0.35,0.1,0.6,0.5,0.35,0.35,0.4,0.4,0.5])

        m_names = ['b1','b2','b3','i1','i2','iii1','iii24','iii3','hg1','hg2','hg3','hg4','freq']
        c_muscle = ['darkred','red','orangered','dodgerblue','blue','darkgreen','lime','springgreen','indigo','fuchsia','mediumorchid','deeppink','yellow']

        plot_fldr = '/home/flythreads/Documents/publication_figures/pub_plots'

        lasso_models = []

        N_steps = 2

        n_deriv = 3

        m_space = np.zeros((13,N_steps,13))

        x = np.linspace(0.0,1.0,num=N_steps)

        t = np.linspace(0,1,num=100)
        X_theta = self.LegendrePolynomials(100,self.N_pol_theta,3)
        X_eta = self.LegendrePolynomials(100,self.N_pol_eta,3)
        X_phi = self.LegendrePolynomials(100,self.N_pol_phi,3)
        X_xi = self.LegendrePolynomials(100,self.N_pol_xi,3)
        line_thck = 1.2
        c_gray = (0.5,0.5,0.5)

        xgrid = np.linspace(0.0,1.0,50)
        ygrid = np.linspace(0.0,1.0,50)
        Xgrid, Ygrid = np.meshgrid(xgrid,ygrid)

        cmap_gray = copy(plt.cm.gist_yarg)

        fig, axs = plt.subplots(13,12)
        for i in range(12):
            pt_select = (self.X_gradient[:,i]>0.01)&(self.X_data[:,2,i]>0.0)&(self.X_data[:,2,i]<1.2)
            X_plus     = self.X_data[pt_select,:,:]
            n_pts = X_plus.shape[0]
            lasso     = linear_model.LinearRegression()
            #lasso     = linear_model.RANSACRegressor()
            reg     = lasso.fit(np.expand_dims(X_plus[:,2,i],axis=1),X_plus[:,2,:])
            lasso_models.append(reg)
            print('muscle '+m_names[i])
            print('n pts: '+str(n_pts))
            print(reg.coef_)
            print(reg.intercept_)

            data_b1     = np.vstack([X_plus[:,2,i],X_plus[:,2,0]])
            data_b2     = np.vstack([X_plus[:,2,i],X_plus[:,2,1]])
            data_b3     = np.vstack([X_plus[:,2,i],X_plus[:,2,2]])
            data_i1     = np.vstack([X_plus[:,2,i],X_plus[:,2,3]])
            data_i2     = np.vstack([X_plus[:,2,i],X_plus[:,2,4]])
            data_iii1     = np.vstack([X_plus[:,2,i],X_plus[:,2,5]])
            data_iii2     = np.vstack([X_plus[:,2,i],X_plus[:,2,6]])
            data_iii3     = np.vstack([X_plus[:,2,i],X_plus[:,2,7]])
            data_hg1     = np.vstack([X_plus[:,2,i],X_plus[:,2,8]])
            data_hg2     = np.vstack([X_plus[:,2,i],X_plus[:,2,9]])
            data_hg3     = np.vstack([X_plus[:,2,i],X_plus[:,2,10]])
            data_hg4     = np.vstack([X_plus[:,2,i],X_plus[:,2,11]])
            data_f         = np.vstack([X_plus[:,2,i],X_plus[:,2,12]])

            vmin = 0.0
            vmax = 0.7

            if i != 0:
                kde_b1     = gaussian_kde(data_b1)
                z_b1     = kde_b1.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_b1     = z_b1.reshape(Xgrid.shape)
                axs[0,i].imshow(Z_b1,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 1:
                kde_b2     = gaussian_kde(data_b2)
                z_b2     = kde_b2.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_b2     = z_b2.reshape(Xgrid.shape)
                axs[1,i].imshow(Z_b2,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 2:
                kde_b3     = gaussian_kde(data_b3)
                z_b3     = kde_b3.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_b3     = z_b3.reshape(Xgrid.shape)
                axs[2,i].imshow(Z_b3,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 3:
                kde_i1     = gaussian_kde(data_i1)
                z_i1     = kde_i1.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_i1     = z_i1.reshape(Xgrid.shape)
                axs[3,i].imshow(Z_i1,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 4:
                kde_i2     = gaussian_kde(data_i2)
                z_i2     = kde_i2.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_i2     = z_i2.reshape(Xgrid.shape)
                axs[4,i].imshow(Z_i2,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 5:
                kde_iii1= gaussian_kde(data_iii1)
                z_iii1     = kde_iii1.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_iii1     = z_iii1.reshape(Xgrid.shape)
                axs[5,i].imshow(Z_iii1,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 6:
                kde_iii2= gaussian_kde(data_iii2)
                z_iii2     = kde_iii2.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_iii2     = z_iii2.reshape(Xgrid.shape)
                axs[6,i].imshow(Z_iii2,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 7:
                kde_iii3= gaussian_kde(data_iii3)
                z_iii3     = kde_iii3.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_iii3     = z_iii3.reshape(Xgrid.shape)
                axs[7,i].imshow(Z_iii3,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 8:
                kde_hg1 = gaussian_kde(data_hg1)
                z_hg1     = kde_hg1.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_hg1     = z_hg1.reshape(Xgrid.shape)
                axs[8,i].imshow(Z_hg1,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 9:
                kde_hg2 = gaussian_kde(data_hg2)
                z_hg2     = kde_hg2.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_hg2     = z_hg2.reshape(Xgrid.shape)
                axs[9,i].imshow(Z_hg2,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 10:
                kde_hg3 = gaussian_kde(data_hg3)
                z_hg3     = kde_hg3.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_hg3     = z_hg3.reshape(Xgrid.shape)
                axs[10,i].imshow(Z_hg3,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 11:
                kde_hg4 = gaussian_kde(data_hg4)
                z_hg4     = kde_hg4.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_hg4     = z_hg4.reshape(Xgrid.shape)
                axs[11,i].imshow(Z_hg4,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)
            if i != 12:
                kde_f = gaussian_kde(data_f)
                z_f     = kde_f.evaluate(np.vstack([Xgrid.ravel(),Ygrid.ravel()]))
                Z_f     = z_f.reshape(Xgrid.shape)
                axs[12,i].imshow(Z_f,origin='lower',aspect='auto',extent=[0.0,1.2,0.0,1.2],cmap=cmap_gray,vmin=vmin,vmax=vmax)


            axs[0,i].set_xlim([0,1.2])
            axs[0,i].set_ylim([0,1.2])
            axs[1,i].set_xlim([0,1.2])
            axs[1,i].set_ylim([0,1.2])
            axs[2,i].set_xlim([0,1.2])
            axs[2,i].set_ylim([0,1.2])
            axs[3,i].set_xlim([0,1.2])
            axs[3,i].set_ylim([0,1.2])
            axs[4,i].set_xlim([0,1.2])
            axs[4,i].set_ylim([0,1.2])
            axs[5,i].set_xlim([0,1.2])
            axs[5,i].set_ylim([0,1.2])
            axs[6,i].set_xlim([0,1.2])
            axs[6,i].set_ylim([0,1.2])
            axs[7,i].set_xlim([0,1.2])
            axs[7,i].set_ylim([0,1.2])
            axs[8,i].set_xlim([0,1.2])
            axs[8,i].set_ylim([0,1.2])
            axs[9,i].set_xlim([0,1.2])
            axs[9,i].set_ylim([0,1.2])
            axs[10,i].set_xlim([0,1.2])
            axs[10,i].set_ylim([0,1.2])
            axs[11,i].set_xlim([0,1.2])
            axs[11,i].set_ylim([0,1.2])
            axs[12,i].set_xlim([0,1.2])
            axs[12,i].set_ylim([0,1.2])

            if i==0:
                adjust_spines(axs[0,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[1,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[2,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[3,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[4,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[5,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[6,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[7,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[8,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[9,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[10,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[11,i],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
                adjust_spines(axs[12,i],['left','bottom'],xticks=[0,1],yticks=[0,1],linewidth=0.8,spineColor='k')
            else:
                adjust_spines(axs[0,i],[])
                adjust_spines(axs[1,i],[])
                adjust_spines(axs[2,i],[])
                adjust_spines(axs[3,i],[])
                adjust_spines(axs[4,i],[])
                adjust_spines(axs[5,i],[])
                adjust_spines(axs[6,i],[])
                adjust_spines(axs[7,i],[])
                adjust_spines(axs[8,i],[])
                adjust_spines(axs[9,i],[])
                adjust_spines(axs[10,i],[])
                adjust_spines(axs[11,i],[])
                adjust_spines(axs[12,i],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
            
            axs[0,i].plot(x,reg.coef_[0]*x+reg.intercept_[0],color=c_muscle[i],linewidth=2)
            axs[1,i].plot(x,reg.coef_[1]*x+reg.intercept_[1],color=c_muscle[i],linewidth=2)
            axs[2,i].plot(x,reg.coef_[2]*x+reg.intercept_[2],color=c_muscle[i],linewidth=2)
            axs[3,i].plot(x,reg.coef_[3]*x+reg.intercept_[3],color=c_muscle[i],linewidth=2)
            axs[4,i].plot(x,reg.coef_[4]*x+reg.intercept_[4],color=c_muscle[i],linewidth=2)
            axs[5,i].plot(x,reg.coef_[5]*x+reg.intercept_[5],color=c_muscle[i],linewidth=2)
            axs[6,i].plot(x,reg.coef_[6]*x+reg.intercept_[6],color=c_muscle[i],linewidth=2)
            axs[7,i].plot(x,reg.coef_[7]*x+reg.intercept_[7],color=c_muscle[i],linewidth=2)
            axs[8,i].plot(x,reg.coef_[8]*x+reg.intercept_[8],color=c_muscle[i],linewidth=2)
            axs[9,i].plot(x,reg.coef_[9]*x+reg.intercept_[9],color=c_muscle[i],linewidth=2)
            axs[10,i].plot(x,reg.coef_[10]*x+reg.intercept_[10],color=c_muscle[i],linewidth=2)
            axs[11,i].plot(x,reg.coef_[11]*x+reg.intercept_[11],color=c_muscle[i],linewidth=2)
            axs[12,i].plot(x,reg.coef_[12]*x+reg.intercept_[12],color=c_muscle[i],linewidth=2)

            axs[0,i].scatter(m_mean[i],reg.coef_[0]*m_mean[i]+reg.intercept_[0],s=70,c=c_gray,marker='.',zorder=10)
            axs[1,i].scatter(m_mean[i],reg.coef_[1]*m_mean[i]+reg.intercept_[1],s=70,c=c_gray,marker='.',zorder=10)
            axs[2,i].scatter(m_mean[i],reg.coef_[2]*m_mean[i]+reg.intercept_[2],s=70,c=c_gray,marker='.',zorder=10)
            axs[3,i].scatter(m_mean[i],reg.coef_[3]*m_mean[i]+reg.intercept_[3],s=70,c=c_gray,marker='.',zorder=10)
            axs[4,i].scatter(m_mean[i],reg.coef_[4]*m_mean[i]+reg.intercept_[4],s=70,c=c_gray,marker='.',zorder=10)
            axs[5,i].scatter(m_mean[i],reg.coef_[5]*m_mean[i]+reg.intercept_[5],s=70,c=c_gray,marker='.',zorder=10)
            axs[6,i].scatter(m_mean[i],reg.coef_[6]*m_mean[i]+reg.intercept_[6],s=70,c=c_gray,marker='.',zorder=10)
            axs[7,i].scatter(m_mean[i],reg.coef_[7]*m_mean[i]+reg.intercept_[7],s=70,c=c_gray,marker='.',zorder=10)
            axs[8,i].scatter(m_mean[i],reg.coef_[8]*m_mean[i]+reg.intercept_[8],s=70,c=c_gray,marker='.',zorder=10)
            axs[9,i].scatter(m_mean[i],reg.coef_[9]*m_mean[i]+reg.intercept_[9],s=70,c=c_gray,marker='.',zorder=10)
            axs[10,i].scatter(m_mean[i],reg.coef_[10]*m_mean[i]+reg.intercept_[10],s=70,c=c_gray,marker='.',zorder=10)
            axs[11,i].scatter(m_mean[i],reg.coef_[11]*m_mean[i]+reg.intercept_[11],s=70,c=c_gray,marker='.',zorder=10)
            axs[12,i].scatter(m_mean[i],reg.coef_[12]*m_mean[i]+reg.intercept_[12],s=70,c=c_gray,marker='.',zorder=10)

            axs[0,i].scatter(1.0,reg.coef_[0]*1.0+reg.intercept_[0],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[1,i].scatter(1.0,reg.coef_[1]*1.0+reg.intercept_[1],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[2,i].scatter(1.0,reg.coef_[2]*1.0+reg.intercept_[2],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[3,i].scatter(1.0,reg.coef_[3]*1.0+reg.intercept_[3],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[4,i].scatter(1.0,reg.coef_[4]*1.0+reg.intercept_[4],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[5,i].scatter(1.0,reg.coef_[5]*1.0+reg.intercept_[5],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[6,i].scatter(1.0,reg.coef_[6]*1.0+reg.intercept_[6],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[7,i].scatter(1.0,reg.coef_[7]*1.0+reg.intercept_[7],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[8,i].scatter(1.0,reg.coef_[8]*1.0+reg.intercept_[8],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[9,i].scatter(1.0,reg.coef_[9]*1.0+reg.intercept_[9],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[10,i].scatter(1.0,reg.coef_[10]*1.0+reg.intercept_[10],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[11,i].scatter(1.0,reg.coef_[11]*1.0+reg.intercept_[11],s=70,c=c_muscle[i],marker='.',zorder=2)
            axs[12,i].scatter(1.0,reg.coef_[12]*1.0+reg.intercept_[12],s=70,c=c_muscle[i],marker='.',zorder=2)

        file_name = self.plot_loc / 'Extended_data_fig_4.svg'
        fig.savefig(file_name, dpi=200)
        
        n_bins = 100
        ca_wbs = np.concatenate(self.ca_traces,axis=1)
        
        fig, axs = plt.subplots(10,12)

        n_b1, bins_b1, patches_b1 = axs[0,0].hist(ca_wbs[0,:], bins=n_bins, color=c_muscle[0], alpha=1.0, label='b1')
        n_b2, bins_b2, patches_b2 = axs[0,1].hist(ca_wbs[1,:], bins=n_bins, color=c_muscle[1], alpha=1.0, label='b2')
        n_b3, bins_b3, patches_b3 = axs[0,2].hist(ca_wbs[2,:], bins=n_bins, color=c_muscle[2], alpha=1.0, label='b3')
        n_i1, bins_i1, patches_i1 = axs[0,3].hist(ca_wbs[3,:], bins=n_bins, color=c_muscle[3], alpha=1.0, label='i1')
        n_i2, bins_i2, patches_i2 = axs[0,4].hist(ca_wbs[4,:], bins=n_bins, color=c_muscle[4], alpha=1.0, label='i2')
        n_iii1, bins_iii1, patches_iii1 = axs[0,5].hist(ca_wbs[5,:], bins=n_bins, color=c_muscle[5], alpha=1.0, label='iii1')
        n_iii2, bins_iii2, patches_iii2 = axs[0,6].hist(ca_wbs[6,:], bins=n_bins, color=c_muscle[6], alpha=1.0, label='iii2')
        n_iii3, bins_iii3, patches_iii3 = axs[0,7].hist(ca_wbs[7,:], bins=n_bins, color=c_muscle[7], alpha=1.0, label='iii3')
        n_hg1, bins_hg1, patches_hg1 = axs[0,8].hist(ca_wbs[8,:], bins=n_bins, color=c_muscle[8], alpha=1.0, label='hg1')
        n_hg2, bins_hg2, patches_hg2 = axs[0,9].hist(ca_wbs[9,:], bins=n_bins, color=c_muscle[9], alpha=1.0, label='hg2')
        n_hg3, bins_hg3, patches_hg3 = axs[0,10].hist(ca_wbs[10,:], bins=n_bins, color=c_muscle[10], alpha=1.0, label='hg3')
        n_hg4, bins_hg4, patches_hg4 = axs[0,11].hist(ca_wbs[11,:], bins=n_bins, color=c_muscle[11], alpha=1.0, label='hg4')
        max_b1 = bins_b1[np.argmax(n_b1)]
        max_b2 = bins_b2[np.argmax(n_b2)]
        max_b3 = bins_b3[np.argmax(n_b3)]
        max_i1 = bins_i1[np.argmax(n_i1)]
        max_i2 = bins_i2[np.argmax(n_i2)]
        max_iii1 = bins_iii1[np.argmax(n_iii1)]
        max_iii2 = bins_iii2[np.argmax(n_iii2)]
        max_iii3 = bins_iii3[np.argmax(n_iii3)]
        max_hg1 = bins_hg1[np.argmax(n_hg1)]
        max_hg2 = bins_hg2[np.argmax(n_hg2)]
        max_hg3 = bins_hg3[np.argmax(n_hg3)]
        max_hg4 = bins_hg4[np.argmax(n_hg4)]

        axs[0,0].set_xlim([0,1.0])
        axs[0,1].set_xlim([0,1.0])
        axs[0,2].set_xlim([0,1.0])
        axs[0,3].set_xlim([0,1.0])
        axs[0,4].set_xlim([0,1.0])
        axs[0,5].set_xlim([0,1.0])
        axs[0,6].set_xlim([0,1.0])
        axs[0,7].set_xlim([0,1.0])
        axs[0,8].set_xlim([0,1.0])
        axs[0,9].set_xlim([0,1.0])
        axs[0,10].set_xlim([0,1.0])
        axs[0,11].set_xlim([0,1.0])

        axs[0,0].set_ylim([0,10000])
        axs[0,1].set_ylim([0,10000])
        axs[0,2].set_ylim([0,10000])
        axs[0,3].set_ylim([0,10000])
        axs[0,4].set_ylim([0,10000])
        axs[0,5].set_ylim([0,10000])
        axs[0,6].set_ylim([0,10000])
        axs[0,7].set_ylim([0,10000])
        axs[0,8].set_ylim([0,10000])
        axs[0,9].set_ylim([0,10000])
        axs[0,10].set_ylim([0,10000])
        axs[0,11].set_ylim([0,10000])

        adjust_spines(axs[0,0],['left'],yticks=[0,5000],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,2],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,3],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,4],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,5],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,6],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,7],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,8],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,9],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,10],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,11],[],linewidth=0.8,spineColor='k')

        m_base = m_mean
        print('baseline')
        print(m_base)

        coeff_b1     = np.squeeze(lasso_models[0].coef_)
        coeff_b2     = np.squeeze(lasso_models[1].coef_)
        coeff_b3     = np.squeeze(lasso_models[2].coef_)
        coeff_i1     = np.squeeze(lasso_models[3].coef_)
        coeff_i2     = np.squeeze(lasso_models[4].coef_)
        coeff_iii1     = np.squeeze(lasso_models[5].coef_)
        coeff_iii2  = np.squeeze(lasso_models[6].coef_)
        coeff_iii3     = np.squeeze(lasso_models[7].coef_)
        coeff_hg1     = np.squeeze(lasso_models[8].coef_)
        coeff_hg2     = np.squeeze(lasso_models[9].coef_)
        coeff_hg3     = np.squeeze(lasso_models[10].coef_)
        coeff_hg4     = np.squeeze(lasso_models[11].coef_)
        #coeff_freq     = np.squeeze(lasso_models[12].coef_)

        X_b1_plus     = np.zeros((N_steps,self.N_window,13))
        X_b1_min     = np.zeros((N_steps,self.N_window,13))
        X_b2_plus      = np.zeros((N_steps,self.N_window,13))
        X_b3_plus      = np.zeros((N_steps,self.N_window,13))
        X_b3_min      = np.zeros((N_steps,self.N_window,13))
        X_i1_plus      = np.zeros((N_steps,self.N_window,13))
        X_i2_plus      = np.zeros((N_steps,self.N_window,13))
        X_i2_min      = np.zeros((N_steps,self.N_window,13))
        X_iii1_plus    = np.zeros((N_steps,self.N_window,13))
        X_iii2_plus = np.zeros((N_steps,self.N_window,13))
        X_iii2_min     = np.zeros((N_steps,self.N_window,13))
        X_iii3_plus = np.zeros((N_steps,self.N_window,13))
        X_iii3_min  = np.zeros((N_steps,self.N_window,13))
        X_hg1_plus     = np.zeros((N_steps,self.N_window,13))
        X_hg2_plus     = np.zeros((N_steps,self.N_window,13))
        X_hg3_plus     = np.zeros((N_steps,self.N_window,13))
        X_hg3_min      = np.zeros((N_steps,self.N_window,13))
        X_hg4_plus     = np.zeros((N_steps,self.N_window,13))
        X_hg4_min     = np.zeros((N_steps,self.N_window,13))

        m_b1_plus     = np.linspace(0.0,1.0-m_base[0],num=N_steps)
        m_b1_min     = np.linspace(-m_base[0],0.0,num=N_steps)
        m_b2_plus      = np.linspace(0.0,1.0-m_base[1],num=N_steps)
        m_b3_plus     = np.linspace(0.0,1.0-m_base[2],num=N_steps)
        m_b3_min     = np.linspace(-m_base[2],0.0,num=N_steps)
        m_i1_plus     = np.linspace(0.0,1.0-m_base[3],num=N_steps)
        m_i2_min     = np.linspace(-m_base[4],0.0,num=N_steps)
        m_i2_plus     = np.linspace(0.0,1.0-m_base[4],num=N_steps)
        m_iii1_plus = np.linspace(0.0,1.0-m_base[5],num=N_steps)
        m_iii2_plus = np.linspace(0.0,1.0-m_base[6],num=N_steps)
        m_iii2_min     = np.linspace(-m_base[6],0.0,num=N_steps)
        m_iii3_plus = np.linspace(0.0,1.0-m_base[7],num=N_steps)
        m_iii3_min     = np.linspace(-m_base[7],0.0,num=N_steps)
        m_hg1_plus     = np.linspace(0.0,1.0-m_base[8],num=N_steps)
        m_hg2_plus     = np.linspace(0.0,1.0-m_base[9],num=N_steps)
        m_hg3_plus     = np.linspace(0.0,1.0-m_base[10],num=N_steps)
        m_hg3_min     = np.linspace(-m_base[10],0.0,num=N_steps)
        m_hg4_plus     = np.linspace(0.0,1.0-m_base[11],num=N_steps)
        m_hg4_min     = np.linspace(-m_base[11],0.0,num=N_steps)

        for j in range(self.N_window):
            for k in range(N_steps):
                X_b1_plus[k,j,:]     = m_base+coeff_b1*m_b1_plus[k]
                X_b1_min[k,j,:]     = m_base+coeff_b1*m_b1_min[k]
                X_b2_plus[k,j,:]      = m_base+coeff_b2*m_b2_plus[k]
                X_b3_plus[k,j,:]      = m_base+coeff_b3*m_b3_plus[k]
                X_b3_min[k,j,:]      = m_base+coeff_b3*m_b3_min[k]
                X_i1_plus[k,j,:]      = m_base+coeff_i1*m_i1_plus[k]
                X_i2_plus[k,j,:]      = m_base+coeff_i2*m_i2_plus[k]
                X_i2_min[k,j,:]      = m_base+coeff_i2*m_i2_min[k]
                X_iii1_plus[k,j,:]    = m_base+coeff_iii1*m_iii1_plus[k]
                X_iii2_plus[k,j,:]  = m_base+coeff_iii2*m_iii2_plus[k]
                X_iii2_min[k,j,:]   = m_base+coeff_iii2*m_iii2_min[k]
                X_iii3_plus[k,j,:]  = m_base+coeff_iii3*m_iii3_plus[k]
                X_iii3_min[k,j,:]   = m_base+coeff_iii3*m_iii3_min[k]
                X_hg1_plus[k,j,:]     = m_base+coeff_hg1*m_hg1_plus[k]
                X_hg2_plus[k,j,:]    = m_base+coeff_hg2*m_hg2_plus[k]
                X_hg3_plus[k,j,:]    = m_base+coeff_hg3*m_hg3_plus[k]
                X_hg3_min[k,j,:]    = m_base+coeff_hg3*m_hg3_min[k]
                X_hg4_plus[k,j,:]    = m_base+coeff_hg4*m_hg4_plus[k]
                X_hg4_min[k,j,:]    = m_base+coeff_hg4*m_hg4_min[k]
                # set frequency to 0.5
                X_b1_plus[k,j,12]     = 0.5
                X_b1_min[k,j,12]     = 0.5
                X_b2_plus[k,j,12]      = 0.5
                X_b3_plus[k,j,12]      = 0.5
                X_b3_min[k,j,12]      = 0.5
                X_i1_plus[k,j,12]      = 0.5
                X_i2_plus[k,j,12]      = 0.5
                X_i2_min[k,j,12]      = 0.5
                X_iii1_plus[k,j,12]    = 0.5
                X_iii2_plus[k,j,12]  = 0.5
                X_iii2_min[k,j,12]   = 0.5
                X_iii3_plus[k,j,12]  = 0.5
                X_iii3_min[k,j,12]   = 0.5
                X_hg1_plus[k,j,12]     = 0.5
                X_hg2_plus[k,j,12]    = 0.5
                X_hg3_plus[k,j,12]    = 0.5
                X_hg3_min[k,j,12]    = 0.5
                X_hg4_plus[k,j,12]    = 0.5
                X_hg4_min[k,j,12]    = 0.5

        y_range = np.arange(13)
            
        axs[1,0].barh(y_range,np.flipud(X_b1_plus[1,0,:]),height=0.8,color=c_muscle[0])
        axs[1,1].barh(y_range,np.flipud(X_b2_plus[1,0,:]),height=0.8,color=c_muscle[1])
        axs[1,2].barh(y_range,np.flipud(X_b3_plus[1,0,:]),height=0.8,color=c_muscle[2])
        axs[1,3].barh(y_range,np.flipud(X_i1_plus[1,0,:]),height=0.8,color=c_muscle[3])
        axs[1,4].barh(y_range,np.flipud(X_i2_plus[1,0,:]),height=0.8,color=c_muscle[4])
        axs[1,5].barh(y_range,np.flipud(X_iii1_plus[1,0,:]),height=0.8,color=c_muscle[5])
        axs[1,6].barh(y_range,np.flipud(X_iii2_plus[1,0,:]),height=0.8,color=c_muscle[6])
        axs[1,7].barh(y_range,np.flipud(X_iii3_plus[1,0,:]),height=0.8,color=c_muscle[7])
        axs[1,8].barh(y_range,np.flipud(X_hg1_plus[1,0,:]),height=0.8,color=c_muscle[8])
        axs[1,9].barh(y_range,np.flipud(X_hg2_plus[1,0,:]),height=0.8,color=c_muscle[9])
        axs[1,10].barh(y_range,np.flipud(X_hg3_plus[1,0,:]),height=0.8,color=c_muscle[10])
        axs[1,11].barh(y_range,np.flipud(X_hg4_plus[1,0,:]),height=0.8,color=c_muscle[11])

        axs[1,0].barh(y_range,np.flipud(X_b1_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,1].barh(y_range,np.flipud(X_b2_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,2].barh(y_range,np.flipud(X_b3_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,3].barh(y_range,np.flipud(X_i1_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,4].barh(y_range,np.flipud(X_i2_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,5].barh(y_range,np.flipud(X_iii1_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,6].barh(y_range,np.flipud(X_iii2_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,7].barh(y_range,np.flipud(X_iii3_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,8].barh(y_range,np.flipud(X_hg1_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,9].barh(y_range,np.flipud(X_hg2_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,10].barh(y_range,np.flipud(X_hg3_plus[0,0,:]),height=0.4,color=c_gray)
        axs[1,11].barh(y_range,np.flipud(X_hg4_plus[0,0,:]),height=0.4,color=c_gray)

        adjust_spines(axs[1,0],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,1],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,2],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,3],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,4],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,5],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,6],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,7],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,8],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,9],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,10],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,11],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')

        Y_b1_plus     = self.predict(X_b1_plus)
        Y_b1_min     = self.predict(X_b1_min)
        Y_b2_plus     = self.predict(X_b2_plus)
        Y_b3_plus    = self.predict(X_b3_plus)
        Y_b3_min    = self.predict(X_b3_min)
        Y_i1_plus   = self.predict(X_i1_plus)
        Y_i2_plus     = self.predict(X_i2_plus)
        Y_i2_min     = self.predict(X_i2_min)
        Y_iii1_plus = self.predict(X_iii1_plus)
        Y_iii2_plus = self.predict(X_iii2_plus)
        Y_iii2_min  = self.predict(X_iii2_min)
        Y_iii3_plus = self.predict(X_iii3_plus)
        Y_iii3_min  = self.predict(X_iii3_min)
        Y_hg1_plus  = self.predict(X_hg1_plus)
        Y_hg2_plus  = self.predict(X_hg2_plus)
        Y_hg3_plus  = self.predict(X_hg3_plus)
        Y_hg3_min   = self.predict(X_hg3_min)
        Y_hg4_plus  = self.predict(X_hg4_plus)
        Y_hg4_min   = self.predict(X_hg4_min)

        a_b1_plus     = self.Wingkin_scale_inverse(Y_b1_plus)
        a_b1_min     = self.Wingkin_scale_inverse(Y_b1_min)
        a_b2_plus     = self.Wingkin_scale_inverse(Y_b2_plus)
        a_b3_plus     = self.Wingkin_scale_inverse(Y_b3_plus)
        a_b3_min     = self.Wingkin_scale_inverse(Y_b3_min)
        a_i1_plus     = self.Wingkin_scale_inverse(Y_i1_plus)
        a_i2_plus     = self.Wingkin_scale_inverse(Y_i2_plus)
        a_i2_min     = self.Wingkin_scale_inverse(Y_i2_min)
        a_iii1_plus = self.Wingkin_scale_inverse(Y_iii1_plus)
        a_iii2_plus = self.Wingkin_scale_inverse(Y_iii2_plus)
        a_iii2_min     = self.Wingkin_scale_inverse(Y_iii2_min)
        a_iii3_plus = self.Wingkin_scale_inverse(Y_iii3_plus)
        a_iii3_min     = self.Wingkin_scale_inverse(Y_iii3_min)
        a_hg1_plus     = self.Wingkin_scale_inverse(Y_hg1_plus)
        a_hg2_plus     = self.Wingkin_scale_inverse(Y_hg2_plus)
        a_hg3_plus     = self.Wingkin_scale_inverse(Y_hg3_plus)
        a_hg3_min     = self.Wingkin_scale_inverse(Y_hg3_min)
        a_hg4_plus     = self.Wingkin_scale_inverse(Y_hg4_plus)
        a_hg4_min     = self.Wingkin_scale_inverse(Y_hg4_min)

        axs[2,0].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,0].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,0].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,0].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,0].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b1_plus[1,44:60]),color=c_muscle[0],linewidth=line_thck)
        axs[4,0].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b1_plus[1,0:20]),color=c_muscle[0],linewidth=line_thck)
        axs[6,0].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b1_plus[1,20:44]),color=c_muscle[0],linewidth=line_thck)
        axs[8,0].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b1_plus[1,60:80]),color=c_muscle[0],linewidth=line_thck)
        axs[3,0].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b1_plus[0,44:60]-a_b1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,0].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b1_plus[0,0:20]-a_b1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,0].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b1_plus[0,20:44]-a_b1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,0].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b1_plus[0,60:80]-a_b1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,0].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b1_plus[1,44:60]-a_b1_plus[0,44:60]),color=c_muscle[0],linewidth=line_thck)
        axs[5,0].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b1_plus[1,0:20]-a_b1_plus[0,0:20]),color=c_muscle[0],linewidth=line_thck)
        axs[7,0].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b1_plus[1,20:44]-a_b1_plus[0,20:44]),color=c_muscle[0],linewidth=line_thck)
        axs[9,0].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b1_plus[1,60:80]-a_b1_plus[0,60:80]),color=c_muscle[0],linewidth=line_thck)
        # b2
        axs[2,1].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,1].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,1].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,1].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,1].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b2_plus[1,44:60]),color=c_muscle[1],linewidth=line_thck)
        axs[4,1].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b2_plus[1,0:20]),color=c_muscle[1],linewidth=line_thck)
        axs[6,1].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b2_plus[1,20:44]),color=c_muscle[1],linewidth=line_thck)
        axs[8,1].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b2_plus[1,60:80]),color=c_muscle[1],linewidth=line_thck)
        axs[3,1].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b2_plus[0,44:60]-a_b2_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,1].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b2_plus[0,0:20]-a_b2_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,1].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b2_plus[0,20:44]-a_b2_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,1].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b2_plus[0,60:80]-a_b2_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,1].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b2_plus[1,44:60]-a_b2_plus[0,44:60]),color=c_muscle[1],linewidth=line_thck)
        axs[5,1].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b2_plus[1,0:20]-a_b2_plus[0,0:20]),color=c_muscle[1],linewidth=line_thck)
        axs[7,1].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b2_plus[1,20:44]-a_b2_plus[0,20:44]),color=c_muscle[1],linewidth=line_thck)
        axs[9,1].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b2_plus[1,60:80]-a_b2_plus[0,60:80]),color=c_muscle[1],linewidth=line_thck)
        # b3
        axs[2,2].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,2].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,2].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,2].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,2].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b3_plus[1,44:60]),color=c_muscle[2],linewidth=line_thck)
        axs[4,2].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b3_plus[1,0:20]),color=c_muscle[2],linewidth=line_thck)
        axs[6,2].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b3_plus[1,20:44]),color=c_muscle[2],linewidth=line_thck)
        axs[8,2].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b3_plus[1,60:80]),color=c_muscle[2],linewidth=line_thck)
        axs[3,2].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b3_plus[0,44:60]-a_b3_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,2].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b3_plus[0,0:20]-a_b3_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,2].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b3_plus[0,20:44]-a_b3_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,2].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b3_plus[0,60:80]-a_b3_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,2].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_b3_plus[1,44:60]-a_b3_plus[0,44:60]),color=c_muscle[2],linewidth=line_thck)
        axs[5,2].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_b3_plus[1,0:20]-a_b3_plus[0,0:20]),color=c_muscle[2],linewidth=line_thck)
        axs[7,2].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_b3_plus[1,20:44]-a_b3_plus[0,20:44]),color=c_muscle[2],linewidth=line_thck)
        axs[9,2].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_b3_plus[1,60:80]-a_b3_plus[0,60:80]),color=c_muscle[2],linewidth=line_thck)
        # i1
        axs[2,3].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_i1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,3].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_i1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,3].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_i1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,3].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_i1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,3].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_i1_plus[1,44:60]),color=c_muscle[3],linewidth=line_thck)
        axs[4,3].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_i1_plus[1,0:20]),color=c_muscle[3],linewidth=line_thck)
        axs[6,3].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_i1_plus[1,20:44]),color=c_muscle[3],linewidth=line_thck)
        axs[8,3].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_i1_plus[1,60:80]),color=c_muscle[3],linewidth=line_thck)
        axs[3,3].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_i1_plus[0,44:60]-a_i1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,3].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_i1_plus[0,0:20]-a_i1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,3].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_i1_plus[0,20:44]-a_i1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,3].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_i1_plus[0,60:80]-a_i1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,3].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_i1_plus[1,44:60]-a_i1_plus[0,44:60]),color=c_muscle[3],linewidth=line_thck)
        axs[5,3].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_i1_plus[1,0:20]-a_i1_plus[0,0:20]),color=c_muscle[3],linewidth=line_thck)
        axs[7,3].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_i1_plus[1,20:44]-a_i1_plus[0,20:44]),color=c_muscle[3],linewidth=line_thck)
        axs[9,3].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_i1_plus[1,60:80]-a_i1_plus[0,60:80]),color=c_muscle[3],linewidth=line_thck)
        # i2
        axs[2,4].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_i1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,4].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_i1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,4].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_i1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,4].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_i1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,4].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_i2_plus[1,44:60]),color=c_muscle[4],linewidth=line_thck)
        axs[4,4].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_i2_plus[1,0:20]),color=c_muscle[4],linewidth=line_thck)
        axs[6,4].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_i2_plus[1,20:44]),color=c_muscle[4],linewidth=line_thck)
        axs[8,4].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_i2_plus[1,60:80]),color=c_muscle[4],linewidth=line_thck)
        axs[3,4].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_i2_plus[0,44:60]-a_i2_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,4].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_i2_plus[0,0:20]-a_i2_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,4].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_i2_plus[0,20:44]-a_i2_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,4].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_i2_plus[0,60:80]-a_i2_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,4].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_i2_plus[1,44:60]-a_i2_plus[0,44:60]),color=c_muscle[4],linewidth=line_thck)
        axs[5,4].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_i2_plus[1,0:20]-a_i2_plus[0,0:20]),color=c_muscle[4],linewidth=line_thck)
        axs[7,4].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_i2_plus[1,20:44]-a_i2_plus[0,20:44]),color=c_muscle[4],linewidth=line_thck)
        axs[9,4].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_i2_plus[1,60:80]-a_i2_plus[0,60:80]),color=c_muscle[4],linewidth=line_thck)
        # iii1
        axs[2,5].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,5].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,5].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,5].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,5].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii1_plus[1,44:60]),color=c_muscle[5],linewidth=line_thck)
        axs[4,5].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii1_plus[1,0:20]),color=c_muscle[5],linewidth=line_thck)
        axs[6,5].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii1_plus[1,20:44]),color=c_muscle[5],linewidth=line_thck)
        axs[8,5].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii1_plus[1,60:80]),color=c_muscle[5],linewidth=line_thck)
        axs[3,5].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii1_plus[0,44:60]-a_iii1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,5].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii1_plus[0,0:20]-a_iii1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,5].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii1_plus[0,20:44]-a_iii1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,5].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii1_plus[0,60:80]-a_iii1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,5].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii1_plus[1,44:60]-a_iii1_plus[0,44:60]),color=c_muscle[5],linewidth=line_thck)
        axs[5,5].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii1_plus[1,0:20]-a_iii1_plus[0,0:20]),color=c_muscle[5],linewidth=line_thck)
        axs[7,5].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii1_plus[1,20:44]-a_iii1_plus[0,20:44]),color=c_muscle[5],linewidth=line_thck)
        axs[9,5].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii1_plus[1,60:80]-a_iii1_plus[0,60:80]),color=c_muscle[5],linewidth=line_thck)
        # iii2
        axs[2,6].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,6].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,6].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,6].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,6].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii2_plus[1,44:60]),color=c_muscle[6],linewidth=line_thck)
        axs[4,6].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii2_plus[1,0:20]),color=c_muscle[6],linewidth=line_thck)
        axs[6,6].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii2_plus[1,20:44]),color=c_muscle[6],linewidth=line_thck)
        axs[8,6].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii2_plus[1,60:80]),color=c_muscle[6],linewidth=line_thck)
        axs[3,6].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii2_plus[0,44:60]-a_iii2_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,6].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii2_plus[0,0:20]-a_iii2_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,6].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii2_plus[0,20:44]-a_iii2_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,6].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii2_plus[0,60:80]-a_iii2_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,6].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii2_plus[1,44:60]-a_iii2_plus[0,44:60]),color=c_muscle[6],linewidth=line_thck)
        axs[5,6].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii2_plus[1,0:20]-a_iii2_plus[0,0:20]),color=c_muscle[6],linewidth=line_thck)
        axs[7,6].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii2_plus[1,20:44]-a_iii2_plus[0,20:44]),color=c_muscle[6],linewidth=line_thck)
        axs[9,6].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii2_plus[1,60:80]-a_iii2_plus[0,60:80]),color=c_muscle[6],linewidth=line_thck)
        # iii3
        axs[2,7].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,7].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,7].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,7].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,7].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii3_plus[1,44:60]),color=c_muscle[7],linewidth=line_thck)
        axs[4,7].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii3_plus[1,0:20]),color=c_muscle[7],linewidth=line_thck)
        axs[6,7].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii3_plus[1,20:44]),color=c_muscle[7],linewidth=line_thck)
        axs[8,7].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii3_plus[1,60:80]),color=c_muscle[7],linewidth=line_thck)
        axs[3,7].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii3_plus[0,44:60]-a_iii3_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,7].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii3_plus[0,0:20]-a_iii3_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,7].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii3_plus[0,20:44]-a_iii3_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,7].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii3_plus[0,60:80]-a_iii3_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,7].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_iii3_plus[1,44:60]-a_iii3_plus[0,44:60]),color=c_muscle[7],linewidth=line_thck)
        axs[5,7].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_iii3_plus[1,0:20]-a_iii3_plus[0,0:20]),color=c_muscle[7],linewidth=line_thck)
        axs[7,7].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_iii3_plus[1,20:44]-a_iii3_plus[0,20:44]),color=c_muscle[7],linewidth=line_thck)
        axs[9,7].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_iii3_plus[1,60:80]-a_iii3_plus[0,60:80]),color=c_muscle[7],linewidth=line_thck)
        # hg1
        axs[2,8].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,8].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,8].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,8].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,8].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg1_plus[1,44:60]),color=c_muscle[8],linewidth=line_thck)
        axs[4,8].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg1_plus[1,0:20]),color=c_muscle[8],linewidth=line_thck)
        axs[6,8].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg1_plus[1,20:44]),color=c_muscle[8],linewidth=line_thck)
        axs[8,8].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg1_plus[1,60:80]),color=c_muscle[8],linewidth=line_thck)
        axs[3,8].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg1_plus[0,44:60]-a_hg1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,8].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg1_plus[0,0:20]-a_hg1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,8].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg1_plus[0,20:44]-a_hg1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,8].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg1_plus[0,60:80]-a_hg1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,8].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg1_plus[1,44:60]-a_hg1_plus[0,44:60]),color=c_muscle[8],linewidth=line_thck)
        axs[5,8].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg1_plus[1,0:20]-a_hg1_plus[0,0:20]),color=c_muscle[8],linewidth=line_thck)
        axs[7,8].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg1_plus[1,20:44]-a_hg1_plus[0,20:44]),color=c_muscle[8],linewidth=line_thck)
        axs[9,8].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg1_plus[1,60:80]-a_hg1_plus[0,60:80]),color=c_muscle[8],linewidth=line_thck)
        # hg2
        axs[2,9].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,9].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,9].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,9].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,9].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg2_plus[1,44:60]),color=c_muscle[9],linewidth=line_thck)
        axs[4,9].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg2_plus[1,0:20]),color=c_muscle[9],linewidth=line_thck)
        axs[6,9].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg2_plus[1,20:44]),color=c_muscle[9],linewidth=line_thck)
        axs[8,9].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg2_plus[1,60:80]),color=c_muscle[9],linewidth=line_thck)
        axs[3,9].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg2_plus[0,44:60]-a_hg2_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,9].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg2_plus[0,0:20]-a_hg2_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,9].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg2_plus[0,20:44]-a_hg2_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,9].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg2_plus[0,60:80]-a_hg2_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,9].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg2_plus[1,44:60]-a_hg2_plus[0,44:60]),color=c_muscle[9],linewidth=line_thck)
        axs[5,9].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg2_plus[1,0:20]-a_hg2_plus[0,0:20]),color=c_muscle[9],linewidth=line_thck)
        axs[7,9].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg2_plus[1,20:44]-a_hg2_plus[0,20:44]),color=c_muscle[9],linewidth=line_thck)
        axs[9,9].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg2_plus[1,60:80]-a_hg2_plus[0,60:80]),color=c_muscle[9],linewidth=line_thck)
        # hg3
        axs[2,10].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,10].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,10].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,10].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,10].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg3_plus[1,44:60]),color=c_muscle[10],linewidth=line_thck)
        axs[4,10].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg3_plus[1,0:20]),color=c_muscle[10],linewidth=line_thck)
        axs[6,10].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg3_plus[1,20:44]),color=c_muscle[10],linewidth=line_thck)
        axs[8,10].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg3_plus[1,60:80]),color=c_muscle[10],linewidth=line_thck)
        axs[3,10].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg3_plus[0,44:60]-a_hg3_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,10].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg3_plus[0,0:20]-a_hg3_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,10].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg3_plus[0,20:44]-a_hg3_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,10].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg3_plus[0,60:80]-a_hg3_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,10].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg3_plus[1,44:60]-a_hg3_plus[0,44:60]),color=c_muscle[10],linewidth=line_thck)
        axs[5,10].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg3_plus[1,0:20]-a_hg3_plus[0,0:20]),color=c_muscle[10],linewidth=line_thck)
        axs[7,10].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg3_plus[1,20:44]-a_hg3_plus[0,20:44]),color=c_muscle[10],linewidth=line_thck)
        axs[9,10].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg3_plus[1,60:80]-a_hg3_plus[0,60:80]),color=c_muscle[10],linewidth=line_thck)
        # hg4
        axs[2,11].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg1_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[4,11].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg1_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[6,11].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg1_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[8,11].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg1_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[2,11].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg4_plus[1,44:60]),color=c_muscle[11],linewidth=line_thck)
        axs[4,11].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg4_plus[1,0:20]),color=c_muscle[11],linewidth=line_thck)
        axs[6,11].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg4_plus[1,20:44]),color=c_muscle[11],linewidth=line_thck)
        axs[8,11].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg4_plus[1,60:80]),color=c_muscle[11],linewidth=line_thck)
        axs[3,11].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg4_plus[0,44:60]-a_hg4_plus[0,44:60]),color=c_gray,linewidth=line_thck)
        axs[5,11].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg4_plus[0,0:20]-a_hg4_plus[0,0:20]),color=c_gray,linewidth=line_thck)
        axs[7,11].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg4_plus[0,20:44]-a_hg4_plus[0,20:44]),color=c_gray,linewidth=line_thck)
        axs[9,11].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg4_plus[0,60:80]-a_hg4_plus[0,60:80]),color=c_gray,linewidth=line_thck)
        axs[3,11].plot(t,(180.0/np.pi)*np.dot(X_phi[:,:,0],a_hg4_plus[1,44:60]-a_hg4_plus[0,44:60]),color=c_muscle[11],linewidth=line_thck)
        axs[5,11].plot(t,(180.0/np.pi)*np.dot(X_theta[:,:,0],a_hg4_plus[1,0:20]-a_hg4_plus[0,0:20]),color=c_muscle[11],linewidth=line_thck)
        axs[7,11].plot(t,(180.0/np.pi)*np.dot(X_eta[:,:,0],a_hg4_plus[1,20:44]-a_hg4_plus[0,20:44]),color=c_muscle[11],linewidth=line_thck)
        axs[9,11].plot(t,(180.0/np.pi)*np.dot(X_xi[:,:,0],a_hg4_plus[1,60:80]-a_hg4_plus[0,60:80]),color=c_muscle[11],linewidth=line_thck)


        adjust_spines(axs[2,0],['left'],yticks=[-45,0,90],linewidth=0.8,spineColor='k')
        adjust_spines(axs[3,0],['left'],yticks=[-20,0,20],linewidth=0.8,spineColor='k')
        adjust_spines(axs[4,0],['left'],yticks=[-30,0,30],linewidth=0.8,spineColor='k')
        adjust_spines(axs[5,0],['left'],yticks=[-20,0,20],linewidth=0.8,spineColor='k')
        adjust_spines(axs[6,0],['left'],yticks=[-90,0,45],linewidth=0.8,spineColor='k')
        adjust_spines(axs[7,0],['left'],yticks=[-20,0,20],linewidth=0.8,spineColor='k')
        adjust_spines(axs[8,0],['left'],yticks=[-30,0,30],linewidth=0.8,spineColor='k')
        adjust_spines(axs[9,0],['left','bottom'],xticks=[0,1],yticks=[-20,0,20],linewidth=0.8,spineColor='k')

        for i in range(12):
            axs[2,i].set_xlim([0,1])
            axs[2,i].set_ylim([-90,120])
            axs[3,i].set_xlim([0,1])
            axs[3,i].set_ylim([-50,30])
            axs[4,i].set_xlim([0,1])
            axs[4,i].set_ylim([-30,45])
            axs[5,i].set_xlim([0,1])
            axs[5,i].set_ylim([-50,30])
            axs[6,i].set_xlim([0,1])
            axs[6,i].set_ylim([-140,70])
            axs[7,i].set_xlim([0,1])
            axs[7,i].set_ylim([-50,30])
            axs[8,i].set_xlim([0,1])
            axs[8,i].set_ylim([-45,45])
            axs[9,i].set_xlim([0,1])
            axs[9,i].set_ylim([-50,30])
            if i>0:
                adjust_spines(axs[2,i],[],linewidth=0.8,spineColor='k')
                adjust_spines(axs[3,i],[],linewidth=0.8,spineColor='k')
                adjust_spines(axs[4,i],[],linewidth=0.8,spineColor='k')
                adjust_spines(axs[5,i],[],linewidth=0.8,spineColor='k')
                adjust_spines(axs[6,i],[],linewidth=0.8,spineColor='k')
                adjust_spines(axs[7,i],[],linewidth=0.8,spineColor='k')
                adjust_spines(axs[8,i],[],linewidth=0.8,spineColor='k')
                adjust_spines(axs[9,i],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')

        file_name = self.plot_loc / 'figure_4.svg'
        fig.savefig(file_name, dpi=300)

    def figure_3(self):

        # mov 225
        X_225 = self.X_data_list[225]
        Y_225 = self.Y_data_list[225]

        t = np.linspace(0.0,1.0,num=100)

        X_theta = self.LegendrePolynomials(100,self.N_pol_theta,1)
        X_eta = self.LegendrePolynomials(100,self.N_pol_eta,1)
        X_phi = self.LegendrePolynomials(100,self.N_pol_phi,1)
        X_xi = self.LegendrePolynomials(100,self.N_pol_xi,1)
        line_thck = 1.0
        line_mscl = 1.5
        line_clr = 'k'

        Y_pred_225 = self.predict(X_225)

        N_samples = 30

        range_225 = [90,120]

        t_225       = 5*(np.arange(range_225[0],range_225[1])-range_225[0])
        t_wingkin_225 = (np.arange(range_225[0]*100,range_225[1]*100)-range_225[0]*100)*0.05

        wb_j = [15]

        c_muscle = ['darkred','red','orangered','dodgerblue','blue','darkgreen','lime','springgreen','indigo','mediumorchid','fuchsia','deeppink','k']
        
        widths = [1.5,2.0]
        heights = [0.25,0.25,0.25,0.25,0.25,0.5,0.5,0.5,0.5]

        #gs_kw = dict(width_ratios=widths, height_ratios=heights)
        #fig, axs = plt.subplots(ncols=2, nrows=5, constrained_layout=True,gridspec_kw=gs_kw)
        #fig.set_size_inches(8,4.5)

        fig = plt.figure()

        gs = GridSpec(9,2,width_ratios=widths,height_ratios=heights)

        ax1  = fig.add_subplot(gs[0,0])
        ax2  = fig.add_subplot(gs[1,0])
        ax3  = fig.add_subplot(gs[2,0])
        ax4  = fig.add_subplot(gs[3,0])
        ax5  = fig.add_subplot(gs[4,0])
        ax6  = fig.add_subplot(gs[5,:])
        ax7  = fig.add_subplot(gs[6,:])
        ax8  = fig.add_subplot(gs[7,:])
        ax9  = fig.add_subplot(gs[8,:])

        ax1.plot(t_225,X_225[range_225[0]:range_225[1],0,0],color=c_muscle[0],linewidth=line_mscl)
        ax1.plot(t_225,X_225[range_225[0]:range_225[1],0,1],color=c_muscle[1],linewidth=line_mscl)
        ax1.plot(t_225,X_225[range_225[0]:range_225[1],0,2],color=c_muscle[2],linewidth=line_mscl)
        ax1.set_ylim([0.0,1.0])
        ax1.add_patch(Rectangle((wb_j[0]*5,0),45,1,color=(0.5,0.5,0.5),alpha=0.5,edgecolor=None))
        #ax1.add_patch(Rectangle((wb_j[0]*5,0),5,1,color=(0.5,0.5,0.5),alpha=0.2,edgecolor=None))
        adjust_spines(ax1,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')

        ax2.plot(t_225,X_225[range_225[0]:range_225[1],0,3],color=c_muscle[3],linewidth=line_mscl)
        ax2.plot(t_225,X_225[range_225[0]:range_225[1],0,4],color=c_muscle[4],linewidth=line_mscl)
        ax2.set_ylim([0.0,1.0])
        ax2.add_patch(Rectangle((wb_j[0]*5,0),45,1,color=(0.5,0.5,0.5),alpha=0.5,edgecolor=None))
        #ax2.add_patch(Rectangle((wb_j[0]*5,0),5,1,color=(0.5,0.5,0.5),alpha=0.2,edgecolor=None))
        adjust_spines(ax2,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')

        ax3.plot(t_225,X_225[range_225[0]:range_225[1],0,5],color=c_muscle[5],linewidth=line_mscl)
        ax3.plot(t_225,X_225[range_225[0]:range_225[1],0,6],color=c_muscle[6],linewidth=line_mscl)
        ax3.plot(t_225,X_225[range_225[0]:range_225[1],0,7],color=c_muscle[7],linewidth=line_mscl)
        ax3.set_ylim([0.0,1.0])
        ax3.add_patch(Rectangle((wb_j[0]*5,0),45,1,color=(0.5,0.5,0.5),alpha=0.5,edgecolor=None))
        #ax3.add_patch(Rectangle((wb_j[0]*5,0),5,1,color=(0.5,0.5,0.5),alpha=0.2,edgecolor=None))
        adjust_spines(ax3,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')

        ax4.plot(t_225,X_225[range_225[0]:range_225[1],0,8],color=c_muscle[8],linewidth=line_mscl)
        ax4.plot(t_225,X_225[range_225[0]:range_225[1],0,9],color=c_muscle[9],linewidth=line_mscl)
        ax4.plot(t_225,X_225[range_225[0]:range_225[1],0,10],color=c_muscle[10],linewidth=line_mscl)
        ax4.plot(t_225,X_225[range_225[0]:range_225[1],0,11],color=c_muscle[11],linewidth=line_mscl)
        ax4.set_ylim([0.0,1.0])
        ax4.add_patch(Rectangle((wb_j[0]*5,0),45,1,color=(0.5,0.5,0.5),alpha=0.5,edgecolor=None))
        #ax4.add_patch(Rectangle((wb_j[0]*5,0),5,1,color=(0.5,0.5,0.5),alpha=0.2,edgecolor=None))
        adjust_spines(ax4,['left'],yticks=[0,1],linewidth=0.8,spineColor='k')

        ax5.plot(t_225,X_225[range_225[0]:range_225[1],0,12]*100+150,color=c_muscle[12],linewidth=line_mscl)
        ax5.set_ylim([150,250])        
        ax5.add_patch(Rectangle((wb_j[0]*5,150),45,100,color=(0.5,0.5,0.5),alpha=0.5,edgecolor=None))
        #ax5.add_patch(Rectangle((wb_j[0]*5,150),5,100,color=(0.5,0.5,0.5),alpha=0.2,edgecolor=None))
        adjust_spines(ax5,['left','bottom'],xticks=[0,50,100,150],yticks=[150,250],linewidth=0.8,spineColor='k')

        for i in range(N_samples):
            a_true_225 = Y_225[range_225[0]+i,:]
            a_pred_225 = Y_pred_225[range_225[0]+i,:]
            ax7.plot(t_wingkin_225[i*100:(i+1)*100],(180.0)*np.dot(X_theta[:,:,0],a_true_225[0:20]),color=line_clr,linewidth=line_thck)
            ax7.plot(t_wingkin_225[i*100:(i+1)*100],(180.0)*np.dot(X_theta[:,:,0],a_pred_225[0:20]),color='r',linewidth=line_thck)
            ax8.plot(t_wingkin_225[i*100:(i+1)*100],(180.0)*np.dot(X_eta[:,:,0],a_true_225[20:44]),color=line_clr,linewidth=line_thck)
            ax8.plot(t_wingkin_225[i*100:(i+1)*100],(180.0)*np.dot(X_eta[:,:,0],a_pred_225[20:44]),color='r',linewidth=line_thck)
            ax6.plot(t_wingkin_225[i*100:(i+1)*100],(180.0)*np.dot(X_phi[:,:,0],a_true_225[44:60]),color=line_clr,linewidth=line_thck)
            ax6.plot(t_wingkin_225[i*100:(i+1)*100],(180.0)*np.dot(X_phi[:,:,0],a_pred_225[44:60]),color='r',linewidth=line_thck)
            ax9.plot(t_wingkin_225[i*100:(i+1)*100],(180.0)*np.dot(X_xi[:,:,0],a_true_225[60:80]),color=line_clr,linewidth=line_thck)
            ax9.plot(t_wingkin_225[i*100:(i+1)*100],(180.0)*np.dot(X_xi[:,:,0],a_pred_225[60:80]),color='r',linewidth=line_thck)

        ax7.set_ylim([-30,45])
        ax7.add_patch(Rectangle((wb_j[0]*5,-30),5,75,color=(0.5,0.5,0.5),alpha=0.5,edgecolor=None))
        adjust_spines(ax7,['left'],yticks=[-30,0,30],linewidth=0.8,spineColor='k')

        ax8.set_ylim([-140,90])
        ax8.add_patch(Rectangle((wb_j[0]*5,-120),5,210,color=(0.5,0.5,0.5),alpha=0.5,edgecolor=None))
        adjust_spines(ax8,['left'],yticks=[-90,0,45],linewidth=0.8,spineColor='k')

        ax6.set_ylim([-90,100])
        ax6.add_patch(Rectangle((wb_j[0]*5,-90),5,210,color=(0.5,0.5,0.5),alpha=0.5,edgecolor=None))
        adjust_spines(ax6,['left'],yticks=[-45,0,90],linewidth=0.8,spineColor='k')

        ax9.set_ylim([-45,45])
        ax9.add_patch(Rectangle((wb_j[0]*5,-60),5,115,color=(0.5,0.5,0.5),alpha=0.5,edgecolor=None))
        adjust_spines(ax9,['left','bottom'],xticks=[0,50,100,150],yticks=[-30,0,30],linewidth=0.8,spineColor='k')
        
        file_name = self.plot_loc / 'figure_3a.svg'
        fig.savefig(file_name, dpi=200)

        rand_1 = np.random.randn(64,13)
        rand_2 = np.random.randn(256,1)
        rand_3 = np.random.randn(1024,1)

        fig, axs = plt.subplots(1,5)
        print(X_225.shape)
        axs[0].pcolor(np.transpose(X_225[3,:,:]),cmap='Greys',vmin=-0.5,vmax=1.5)
        axs[0].set_xticklabels([])
        axs[0].set_yticklabels([])
        axs[0].spines['left'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        [k.set_visible(False) for k in axs[0].get_xticklines()]
        [k.set_visible(False) for k in axs[0].get_yticklines()]
        axs[1].pcolor(rand_1,cmap='Greys',vmin=-1,vmax=1)
        axs[1].set_xticklabels([])
        axs[1].set_yticklabels([])
        axs[1].spines['left'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['bottom'].set_visible(False)
        [k.set_visible(False) for k in axs[1].get_xticklines()]
        [k.set_visible(False) for k in axs[1].get_yticklines()]
        axs[2].pcolor(rand_2,cmap='Greys',vmin=-1,vmax=1)
        axs[2].set_xticklabels([])
        axs[2].set_yticklabels([])
        axs[2].spines['left'].set_visible(False)
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['bottom'].set_visible(False)
        [k.set_visible(False) for k in axs[2].get_xticklines()]
        [k.set_visible(False) for k in axs[2].get_yticklines()]
        axs[3].pcolor(rand_3,cmap='Greys',vmin=-1,vmax=1)
        axs[3].set_xticklabels([])
        axs[3].set_yticklabels([])
        axs[3].spines['left'].set_visible(False)
        axs[3].spines['top'].set_visible(False)
        axs[3].spines['right'].set_visible(False)
        axs[3].spines['bottom'].set_visible(False)
        [k.set_visible(False) for k in axs[3].get_xticklines()]
        [k.set_visible(False) for k in axs[3].get_yticklines()]
        axs[4].pcolor(np.transpose(np.expand_dims(Y_pred_225[3,:],axis=0)),cmap='Greys',vmin=-1,vmax=1)
        axs[4].set_xticklabels([])
        axs[4].set_yticklabels([])
        axs[4].spines['left'].set_visible(False)
        axs[4].spines['top'].set_visible(False)
        axs[4].spines['right'].set_visible(False)
        axs[4].spines['bottom'].set_visible(False)
        [k.set_visible(False) for k in axs[4].get_xticklines()]
        [k.set_visible(False) for k in axs[4].get_yticklines()]

        file_name = self.plot_loc / 'figure_3b.svg'
        fig.savefig(file_name, dpi=200)

    def figure_5(self):

        m_mean = np.array([0.5,0.1,0.5,0.35,0.35,0.1,0.6,0.5,0.35,0.35,0.4,0.4,0.5])

        dFT_du_L = np.array([[-0.13798439,  0.25995924, -0.30252188, -0.24197954, -0.54039025, -0.26601864, 0.19141026, -0.54956226, -0.06172088, -0.03086602,  0.23442875, -0.34096134],
            [-0.10072399,  0.32773337, -0.02465143, -0.15886206, -0.14818341,  0.04306711,  0.4966922,   0.14304729, -0.14897121, -0.15939759,  0.46592549,  0.15202313],
            [ 0.18469051,  0.81547836, -0.29040922, -0.18003368, -0.37617801,  0.26171538,  1.23382869,  0.06462377, -0.03145677,  0.07926825,  1.23603524,  0.19249149],
            [ 0.07038646,  0.42623308, -0.24395861, -0.12747967, -0.25041172,  0.10125387,  0.53187341, -0.03862265, -0.0335136,  -0.00421778,  0.54513611,  0.11267449],
            [-0.02709364, -0.10260878, -0.02197765,  0.05901346,  0.05635526, -0.02760262, -0.06134992,  0.04134153, -0.00637308, -0.03595999,  0.11300017,  0.15850667],
            [ 0.05316496, -0.33060946,  0.22743509,  0.20070058,  0.37586804,  0.14049337, -0.09743827,  0.17975139,  0.10783808,  0.11540679, -0.05936869,  0.18306013]])

        M_body = np.array([[ 1.0, 0.0, 0.0, 0.0, -1.0176e-02, -1.7079e-04],
             [ 0.0, 1.0, 0.0, 1.0176e-02, 0.0, -2.2945e-01],
             [ 0.0, 0.0, 1.0, 1.7079e-04, 2.2945e-01, 0.0],
             [ 0.0, 1.0176e-02, 1.7079e-04, 1.2834e-01, 7.1496e-05, 2.4212e-02],
             [-1.0176e-02, 0.0, 2.2945e-01, 7.1496e-05, 7.2614e-01, -3.7714e-06],
             [-1.7079e-04, -2.2945e-01, 0.0, 2.4212e-02, -3.7714e-06, 7.1336e-01]])

        M_body_inv = np.linalg.inv(M_body)

        acc_scaling = np.array([[self.g, 0.0 ,0.0 ,0.0 ,0.0, 0.0],
            [0.0, self.g ,0.0 ,0.0 ,0.0, 0.0],
            [0.0, 0.0 ,self.g ,0.0 ,0.0, 0.0],
            [0.0, 0.0 ,0.0 ,self.g*self.R_fly*1e-6, 0.0, 0.0],
            [0.0, 0.0 ,0.0 ,0.0 ,self.g*self.R_fly*1e-6, 0.0],
            [0.0, 0.0 ,0.0 ,0.0 ,0.0, self.g*self.R_fly*1e-6]])
        acc_du_L = np.dot(M_body_inv,dFT_du_L)

        print('accelerations')
        print(acc_du_L)

        l_width = 1.1

        c_muscle = ['darkred','red','orangered','dodgerblue','blue','darkgreen','lime','springgreen','indigo','mediumorchid','fuchsia','deeppink','k']

        fig, axs = plt.subplots(6,5)

        ylim0 = [-0.5,0.5]
        ylim1 = [-0.5,0.5]
        ylim2 = [-0.25,0.75]
        ylim3 = [-0.5,0.5]
        ylim4 = [-0.5,0.5]
        ylim5 = [-0.5,0.5]

        axs[0,0].plot([0,1.0-m_mean[0]],[0,dFT_du_L[0,0]*(1.0-m_mean[0])],linewidth=l_width,color=c_muscle[0])
        axs[0,0].plot([0,1.0-m_mean[1]],[0,dFT_du_L[0,1]*(1.0-m_mean[1])],linewidth=l_width,color=c_muscle[1])
        axs[0,0].plot([0,1.0-m_mean[2]],[0,dFT_du_L[0,2]*(1.0-m_mean[2])],linewidth=l_width,color=c_muscle[2])
        axs[1,0].plot([0,1.0-m_mean[0]],[0,dFT_du_L[1,0]*(1.0-m_mean[0])],linewidth=l_width,color=c_muscle[0])
        axs[1,0].plot([0,1.0-m_mean[1]],[0,dFT_du_L[1,1]*(1.0-m_mean[1])],linewidth=l_width,color=c_muscle[1])
        axs[1,0].plot([0,1.0-m_mean[2]],[0,dFT_du_L[1,2]*(1.0-m_mean[2])],linewidth=l_width,color=c_muscle[2])
        axs[2,0].plot([0,1.0-m_mean[0]],[0,dFT_du_L[2,0]*(1.0-m_mean[0])],linewidth=l_width,color=c_muscle[0])
        axs[2,0].plot([0,1.0-m_mean[1]],[0,dFT_du_L[2,1]*(1.0-m_mean[1])],linewidth=l_width,color=c_muscle[1])
        axs[2,0].plot([0,1.0-m_mean[2]],[0,dFT_du_L[2,2]*(1.0-m_mean[2])],linewidth=l_width,color=c_muscle[2])
        axs[3,0].plot([0,1.0-m_mean[0]],[0,dFT_du_L[3,0]*(1.0-m_mean[0])],linewidth=l_width,color=c_muscle[0])
        axs[3,0].plot([0,1.0-m_mean[1]],[0,dFT_du_L[3,1]*(1.0-m_mean[1])],linewidth=l_width,color=c_muscle[1])
        axs[3,0].plot([0,1.0-m_mean[2]],[0,dFT_du_L[3,2]*(1.0-m_mean[2])],linewidth=l_width,color=c_muscle[2])
        axs[4,0].plot([0,1.0-m_mean[0]],[0,dFT_du_L[4,0]*(1.0-m_mean[0])],linewidth=l_width,color=c_muscle[0])
        axs[4,0].plot([0,1.0-m_mean[1]],[0,dFT_du_L[4,1]*(1.0-m_mean[1])],linewidth=l_width,color=c_muscle[1])
        axs[4,0].plot([0,1.0-m_mean[2]],[0,dFT_du_L[4,2]*(1.0-m_mean[2])],linewidth=l_width,color=c_muscle[2])
        axs[5,0].plot([0,1.0-m_mean[0]],[0,dFT_du_L[5,0]*(1.0-m_mean[0])],linewidth=l_width,color=c_muscle[0])
        axs[5,0].plot([0,1.0-m_mean[1]],[0,dFT_du_L[5,1]*(1.0-m_mean[1])],linewidth=l_width,color=c_muscle[1])
        axs[5,0].plot([0,1.0-m_mean[2]],[0,dFT_du_L[5,2]*(1.0-m_mean[2])],linewidth=l_width,color=c_muscle[2])

        axs[0,0].set_xlim([0,1])
        axs[0,0].set_ylim(ylim0)
        axs[1,0].set_xlim([0,1])
        axs[1,0].set_ylim(ylim1)
        axs[2,0].set_xlim([0,1])
        axs[2,0].set_ylim(ylim2)
        axs[3,0].set_xlim([0,1])
        axs[3,0].set_ylim(ylim3)
        axs[4,0].set_xlim([0,1])
        axs[4,0].set_ylim(ylim4)
        axs[5,0].set_xlim([0,1])
        axs[5,0].set_ylim(ylim5)
        adjust_spines(axs[0,0],['left'],yticks=[-0.25,0,0.25],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,0],['left'],yticks=[-0.25,0,0.25],linewidth=0.8,spineColor='k')
        adjust_spines(axs[2,0],['left'],yticks=[-0.25,0,0.75],linewidth=0.8,spineColor='k')
        adjust_spines(axs[3,0],['left'],yticks=[-0.25,0,0.25],linewidth=0.8,spineColor='k')
        adjust_spines(axs[4,0],['left'],yticks=[-0.25,0,0.25],linewidth=0.8,spineColor='k')
        adjust_spines(axs[5,0],['left','bottom'],xticks=[0,1],yticks=[-0.25,0,0.25],linewidth=0.8,spineColor='k')

        axs[0,1].plot([0,1.0-m_mean[3]],[0,dFT_du_L[0,3]*(1.0-m_mean[3])],linewidth=l_width,color=c_muscle[3])
        axs[0,1].plot([0,1.0-m_mean[4]],[0,dFT_du_L[0,4]*(1.0-m_mean[4])],linewidth=l_width,color=c_muscle[4])
        axs[1,1].plot([0,1.0-m_mean[3]],[0,dFT_du_L[1,3]*(1.0-m_mean[3])],linewidth=l_width,color=c_muscle[3])
        axs[1,1].plot([0,1.0-m_mean[4]],[0,dFT_du_L[1,4]*(1.0-m_mean[4])],linewidth=l_width,color=c_muscle[4])
        axs[2,1].plot([0,1.0-m_mean[3]],[0,dFT_du_L[2,3]*(1.0-m_mean[3])],linewidth=l_width,color=c_muscle[3])
        axs[2,1].plot([0,1.0-m_mean[4]],[0,dFT_du_L[2,4]*(1.0-m_mean[4])],linewidth=l_width,color=c_muscle[4])
        axs[3,1].plot([0,1.0-m_mean[3]],[0,dFT_du_L[3,3]*(1.0-m_mean[3])],linewidth=l_width,color=c_muscle[3])
        axs[3,1].plot([0,1.0-m_mean[4]],[0,dFT_du_L[3,4]*(1.0-m_mean[4])],linewidth=l_width,color=c_muscle[4])
        axs[4,1].plot([0,1.0-m_mean[3]],[0,dFT_du_L[4,3]*(1.0-m_mean[3])],linewidth=l_width,color=c_muscle[3])
        axs[4,1].plot([0,1.0-m_mean[4]],[0,dFT_du_L[4,4]*(1.0-m_mean[4])],linewidth=l_width,color=c_muscle[4])
        axs[5,1].plot([0,1.0-m_mean[3]],[0,dFT_du_L[5,3]*(1.0-m_mean[3])],linewidth=l_width,color=c_muscle[3])
        axs[5,1].plot([0,1.0-m_mean[4]],[0,dFT_du_L[5,4]*(1.0-m_mean[4])],linewidth=l_width,color=c_muscle[4])

        axs[0,1].set_xlim([0,1])
        axs[0,1].set_ylim(ylim0)
        axs[1,1].set_xlim([0,1])
        axs[1,1].set_ylim(ylim1)
        axs[2,1].set_xlim([0,1])
        axs[2,1].set_ylim(ylim2)
        axs[3,1].set_xlim([0,1])
        axs[3,1].set_ylim(ylim3)
        axs[4,1].set_xlim([0,1])
        axs[4,1].set_ylim(ylim4)
        axs[5,1].set_xlim([0,1])
        axs[5,1].set_ylim(ylim5)
        adjust_spines(axs[0,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[2,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[3,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[4,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[5,1],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')

        axs[0,2].plot([0,1.0-m_mean[5]],[0,dFT_du_L[0,5]*(1.0-m_mean[5])],linewidth=l_width,color=c_muscle[5])
        axs[0,2].plot([0,1.0-m_mean[6]],[0,dFT_du_L[0,6]*(1.0-m_mean[6])],linewidth=l_width,color=c_muscle[6])
        axs[0,2].plot([0,1.0-m_mean[7]],[0,dFT_du_L[0,7]*(1.0-m_mean[7])],linewidth=l_width,color=c_muscle[7])
        axs[1,2].plot([0,1.0-m_mean[5]],[0,dFT_du_L[1,5]*(1.0-m_mean[5])],linewidth=l_width,color=c_muscle[5])
        axs[1,2].plot([0,1.0-m_mean[6]],[0,dFT_du_L[1,6]*(1.0-m_mean[6])],linewidth=l_width,color=c_muscle[6])
        axs[1,2].plot([0,1.0-m_mean[7]],[0,dFT_du_L[1,7]*(1.0-m_mean[7])],linewidth=l_width,color=c_muscle[7])
        axs[2,2].plot([0,1.0-m_mean[5]],[0,dFT_du_L[2,5]*(1.0-m_mean[5])],linewidth=l_width,color=c_muscle[5])
        axs[2,2].plot([0,1.0-m_mean[6]],[0,dFT_du_L[2,6]*(1.0-m_mean[6])],linewidth=l_width,color=c_muscle[6])
        axs[2,2].plot([0,1.0-m_mean[7]],[0,dFT_du_L[2,7]*(1.0-m_mean[7])],linewidth=l_width,color=c_muscle[7])
        axs[3,2].plot([0,1.0-m_mean[5]],[0,dFT_du_L[3,5]*(1.0-m_mean[5])],linewidth=l_width,color=c_muscle[5])
        axs[3,2].plot([0,1.0-m_mean[6]],[0,dFT_du_L[3,6]*(1.0-m_mean[6])],linewidth=l_width,color=c_muscle[6])
        axs[3,2].plot([0,1.0-m_mean[7]],[0,dFT_du_L[3,7]*(1.0-m_mean[7])],linewidth=l_width,color=c_muscle[7])
        axs[4,2].plot([0,1.0-m_mean[5]],[0,dFT_du_L[4,5]*(1.0-m_mean[5])],linewidth=l_width,color=c_muscle[5])
        axs[4,2].plot([0,1.0-m_mean[6]],[0,dFT_du_L[4,6]*(1.0-m_mean[6])],linewidth=l_width,color=c_muscle[6])
        axs[4,2].plot([0,1.0-m_mean[7]],[0,dFT_du_L[4,7]*(1.0-m_mean[7])],linewidth=l_width,color=c_muscle[7])
        axs[5,2].plot([0,1.0-m_mean[5]],[0,dFT_du_L[5,5]*(1.0-m_mean[5])],linewidth=l_width,color=c_muscle[5])
        axs[5,2].plot([0,1.0-m_mean[6]],[0,dFT_du_L[5,6]*(1.0-m_mean[6])],linewidth=l_width,color=c_muscle[6])
        axs[5,2].plot([0,1.0-m_mean[7]],[0,dFT_du_L[5,7]*(1.0-m_mean[7])],linewidth=l_width,color=c_muscle[7])

        axs[0,2].set_xlim([0,1])
        axs[0,2].set_ylim(ylim0)
        axs[1,2].set_xlim([0,1])
        axs[1,2].set_ylim(ylim1)
        axs[2,2].set_xlim([0,1])
        axs[2,2].set_ylim(ylim2)
        axs[3,2].set_xlim([0,1])
        axs[3,2].set_ylim(ylim3)
        axs[4,2].set_xlim([0,1])
        axs[4,2].set_ylim(ylim4)
        axs[5,2].set_xlim([0,1])
        axs[5,2].set_ylim(ylim5)
        adjust_spines(axs[0,2],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,2],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[2,2],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[3,2],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[4,2],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[5,2],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')

        axs[0,3].plot([0,1.0-m_mean[8]],[0,dFT_du_L[0,8]*(1.0-m_mean[8])],linewidth=l_width,color=c_muscle[8])
        axs[0,3].plot([0,1.0-m_mean[9]],[0,dFT_du_L[0,9]*(1.0-m_mean[9])],linewidth=l_width,color=c_muscle[9])
        axs[1,3].plot([0,1.0-m_mean[8]],[0,dFT_du_L[1,8]*(1.0-m_mean[8])],linewidth=l_width,color=c_muscle[8])
        axs[1,3].plot([0,1.0-m_mean[9]],[0,dFT_du_L[1,9]*(1.0-m_mean[9])],linewidth=l_width,color=c_muscle[9])
        axs[2,3].plot([0,1.0-m_mean[8]],[0,dFT_du_L[2,8]*(1.0-m_mean[8])],linewidth=l_width,color=c_muscle[8])
        axs[2,3].plot([0,1.0-m_mean[9]],[0,dFT_du_L[2,9]*(1.0-m_mean[9])],linewidth=l_width,color=c_muscle[9])
        axs[3,3].plot([0,1.0-m_mean[8]],[0,dFT_du_L[3,8]*(1.0-m_mean[8])],linewidth=l_width,color=c_muscle[8])
        axs[3,3].plot([0,1.0-m_mean[9]],[0,dFT_du_L[3,9]*(1.0-m_mean[9])],linewidth=l_width,color=c_muscle[9])
        axs[4,3].plot([0,1.0-m_mean[8]],[0,dFT_du_L[4,8]*(1.0-m_mean[8])],linewidth=l_width,color=c_muscle[8])
        axs[4,3].plot([0,1.0-m_mean[9]],[0,dFT_du_L[4,9]*(1.0-m_mean[9])],linewidth=l_width,color=c_muscle[9])
        axs[5,3].plot([0,1.0-m_mean[8]],[0,dFT_du_L[5,8]*(1.0-m_mean[8])],linewidth=l_width,color=c_muscle[8])
        axs[5,3].plot([0,1.0-m_mean[9]],[0,dFT_du_L[5,9]*(1.0-m_mean[9])],linewidth=l_width,color=c_muscle[9])

        axs[0,3].set_xlim([0,1])
        axs[0,3].set_ylim(ylim0)
        axs[1,3].set_xlim([0,1])
        axs[1,3].set_ylim(ylim1)
        axs[2,3].set_xlim([0,1])
        axs[2,3].set_ylim(ylim2)
        axs[3,3].set_xlim([0,1])
        axs[3,3].set_ylim(ylim3)
        axs[4,3].set_xlim([0,1])
        axs[4,3].set_ylim(ylim4)
        axs[5,3].set_xlim([0,1])
        axs[5,3].set_ylim(ylim5)
        adjust_spines(axs[0,3],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,3],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[2,3],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[3,3],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[4,3],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[5,3],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')

        axs[0,4].plot([0,1.0-m_mean[10]],[0,dFT_du_L[0,10]*(1.0-m_mean[10])],linewidth=l_width,color=c_muscle[10])
        axs[0,4].plot([0,1.0-m_mean[11]],[0,dFT_du_L[0,11]*(1.0-m_mean[11])],linewidth=l_width,color=c_muscle[11])
        axs[1,4].plot([0,1.0-m_mean[10]],[0,dFT_du_L[1,10]*(1.0-m_mean[10])],linewidth=l_width,color=c_muscle[10])
        axs[1,4].plot([0,1.0-m_mean[11]],[0,dFT_du_L[1,11]*(1.0-m_mean[11])],linewidth=l_width,color=c_muscle[11])
        axs[2,4].plot([0,1.0-m_mean[10]],[0,dFT_du_L[2,10]*(1.0-m_mean[10])],linewidth=l_width,color=c_muscle[10])
        axs[2,4].plot([0,1.0-m_mean[11]],[0,dFT_du_L[2,11]*(1.0-m_mean[11])],linewidth=l_width,color=c_muscle[11])
        axs[3,4].plot([0,1.0-m_mean[10]],[0,dFT_du_L[3,10]*(1.0-m_mean[10])],linewidth=l_width,color=c_muscle[10])
        axs[3,4].plot([0,1.0-m_mean[11]],[0,dFT_du_L[3,11]*(1.0-m_mean[11])],linewidth=l_width,color=c_muscle[11])
        axs[4,4].plot([0,1.0-m_mean[10]],[0,dFT_du_L[4,10]*(1.0-m_mean[10])],linewidth=l_width,color=c_muscle[10])
        axs[4,4].plot([0,1.0-m_mean[11]],[0,dFT_du_L[4,11]*(1.0-m_mean[11])],linewidth=l_width,color=c_muscle[11])
        axs[5,4].plot([0,1.0-m_mean[10]],[0,dFT_du_L[5,10]*(1.0-m_mean[10])],linewidth=l_width,color=c_muscle[10])
        axs[5,4].plot([0,1.0-m_mean[11]],[0,dFT_du_L[5,11]*(1.0-m_mean[11])],linewidth=l_width,color=c_muscle[11])

        axs[0,4].set_xlim([0,1])
        axs[0,4].set_ylim(ylim0)
        axs[1,4].set_xlim([0,1])
        axs[1,4].set_ylim(ylim1)
        axs[2,4].set_xlim([0,1])
        axs[2,4].set_ylim(ylim2)
        axs[3,4].set_xlim([0,1])
        axs[3,4].set_ylim(ylim3)
        axs[4,4].set_xlim([0,1])
        axs[4,4].set_ylim(ylim4)
        axs[5,4].set_xlim([0,1])
        axs[5,4].set_ylim(ylim5)
        adjust_spines(axs[0,4],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,4],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[2,4],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[3,4],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[4,4],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[5,4],['bottom'],xticks=[0,1],linewidth=0.8,spineColor='k')

        plot_fldr = '/home/flythreads/Documents/publication_figures/pub_plots'
        os.chdir(plot_fldr)
        file_name = 'figure_4.svg'
        fig.savefig(file_name, dpi=200)

    def plot_mov_prediction(self,mov_ind):
        X_mov = self.X_data_list[mov_ind]
        Y_mov = self.Y_data_list[mov_ind]
        N_samples = X_mov.shape[0]
        t_muscle  = [0]
        X_theta = self.LegendrePolynomials(100,self.N_pol_theta,1)
        X_eta = self.LegendrePolynomials(100,self.N_pol_eta,1)
        X_phi = self.LegendrePolynomials(100,self.N_pol_phi,1)
        X_xi = self.LegendrePolynomials(100,self.N_pol_xi,1)
        line_thck=0.8
        line_clr = (0.5,0.5,0.5)
        a_pred = self.predict(X_mov)

        t_zoom_start = 0.4
        t_zoom_end      = 0.6

        c_muscle = ['darkred','red','orangered','dodgerblue','blue','darkgreen','lime','springgreen','indigo','mediumorchid','fuchsia','deeppink','k']

        fig, axs = plt.subplots(9,2)
        fig.suptitle('Mov '+str(mov_ind))
        fig.set_size_inches(24,16)
        plt.axis('tight')
        for i in range(N_samples):
            a_true = Y_mov[i,:]
            a_theta_pred = a_pred[i,0:20]
            a_eta_pred      = a_pred[i,20:44]
            a_phi_pred      = a_pred[i,44:60]
            a_xi_pred      = a_pred[i,60:80]
            a_theta_true = a_true[0:20]
            a_eta_true      = a_true[20:44]
            a_phi_true      = a_true[44:60]
            a_xi_true      = a_true[60:80]
            f_i = X_mov[i,0,12]*100.0+150.0
            t_i = np.linspace(t_muscle[i],t_muscle[i]+1.0/f_i,num=100)
            t_muscle.append(t_muscle[i]+1.0/f_i)
            axs[6,0].plot(t_i,(180.0)*np.dot(X_theta[:,:,0],a_theta_true),color='k',linewidth=line_thck)
            axs[7,0].plot(t_i,(180.0)*np.dot(X_eta[:,:,0],a_eta_true),color='k',linewidth=line_thck)
            axs[5,0].plot(t_i,(180.0)*np.dot(X_phi[:,:,0],a_phi_true),color='k',linewidth=line_thck)
            axs[8,0].plot(t_i,(180.0)*np.dot(X_xi[:,:,0],a_xi_true),color='k',linewidth=line_thck)
            axs[6,0].plot(t_i,(180.0)*np.dot(X_theta[:,:,0],a_theta_pred),color='r',linewidth=line_thck)
            axs[7,0].plot(t_i,(180.0)*np.dot(X_eta[:,:,0],a_eta_pred),color='r',linewidth=line_thck)
            axs[5,0].plot(t_i,(180.0)*np.dot(X_phi[:,:,0],a_phi_pred),color='r',linewidth=line_thck)
            axs[8,0].plot(t_i,(180.0)*np.dot(X_xi[:,:,0],a_xi_pred),color='r',linewidth=line_thck)
            axs[6,1].plot(t_i,(180.0)*np.dot(X_theta[:,:,0],a_theta_true),color='k',linewidth=line_thck)
            axs[7,1].plot(t_i,(180.0)*np.dot(X_eta[:,:,0],a_eta_true),color='k',linewidth=line_thck)
            axs[5,1].plot(t_i,(180.0)*np.dot(X_phi[:,:,0],a_phi_true),color='k',linewidth=line_thck)
            axs[8,1].plot(t_i,(180.0)*np.dot(X_xi[:,:,0],a_xi_true),color='k',linewidth=line_thck)
            axs[6,1].plot(t_i,(180.0)*np.dot(X_theta[:,:,0],a_theta_pred),color='r',linewidth=line_thck)
            axs[7,1].plot(t_i,(180.0)*np.dot(X_eta[:,:,0],a_eta_pred),color='r',linewidth=line_thck)
            axs[5,1].plot(t_i,(180.0)*np.dot(X_phi[:,:,0],a_phi_pred),color='r',linewidth=line_thck)
            axs[8,1].plot(t_i,(180.0)*np.dot(X_xi[:,:,0],a_xi_pred),color='r',linewidth=line_thck)
        axs[6,0].set_ylabel(r'$\theta$')
        axs[7,0].set_ylabel(r'$\eta$')
        axs[5,0].set_ylabel(r'$\phi$')
        axs[8,0].set_ylabel(r'$\xi$')
        axs[8,0].set_xlabel('t [s]')
        axs[5,0].legend(['true','pred'],loc=1)
        axs[6,0].set_ylim([-45,45])
        axs[7,0].set_ylim([-150,90])
        axs[5,0].set_ylim([-90,120])
        axs[8,0].set_ylim([-60,60])
        axs[6,0].set_xlim([0,1])
        axs[7,0].set_xlim([0,1])
        axs[5,0].set_xlim([0,1])
        axs[8,0].set_xlim([0,1])
        axs[5,1].set_xlim([t_zoom_start,t_zoom_end])
        axs[6,1].set_xlim([t_zoom_start,t_zoom_end])
        axs[7,1].set_xlim([t_zoom_start,t_zoom_end])
        axs[8,1].set_xlim([t_zoom_start,t_zoom_end])
        axs[6,1].set_ylim([-45,45])
        axs[7,1].set_ylim([-150,90])
        axs[5,1].set_ylim([-90,120])
        axs[8,1].set_ylim([-60,60])        
        axs[8,1].set_xlabel('t [s]')
        t_m = np.array(t_muscle[:-1])+0.0025
        axs[0,0].plot(t_m,X_mov[:,0,0],color=c_muscle[0],linewidth=line_thck)
        axs[0,0].plot(t_m,X_mov[:,0,1],color=c_muscle[1],linewidth=line_thck)
        axs[0,0].plot(t_m,X_mov[:,0,2],color=c_muscle[2],linewidth=line_thck)
        axs[1,0].plot(t_m,X_mov[:,0,3],color=c_muscle[3],linewidth=line_thck)
        axs[1,0].plot(t_m,X_mov[:,0,4],color=c_muscle[4],linewidth=line_thck)
        axs[2,0].plot(t_m,X_mov[:,0,5],color=c_muscle[5],linewidth=line_thck)
        axs[2,0].plot(t_m,X_mov[:,0,6],color=c_muscle[6],linewidth=line_thck)
        axs[2,0].plot(t_m,X_mov[:,0,7],color=c_muscle[7],linewidth=line_thck)
        axs[3,0].plot(t_m,X_mov[:,0,8],color=c_muscle[8],linewidth=line_thck)
        axs[3,0].plot(t_m,X_mov[:,0,9],color=c_muscle[9],linewidth=line_thck)
        axs[3,0].plot(t_m,X_mov[:,0,10],color=c_muscle[10],linewidth=line_thck)
        axs[3,0].plot(t_m,X_mov[:,0,11],color=c_muscle[11],linewidth=line_thck)
        axs[4,0].plot(t_m,X_mov[:,0,12]*100.0+150.0,color=c_muscle[12],linewidth=line_thck)
        axs[0,1].plot(t_m,X_mov[:,0,0],color=c_muscle[0],linewidth=line_thck)
        axs[0,1].plot(t_m,X_mov[:,0,1],color=c_muscle[1],linewidth=line_thck)
        axs[0,1].plot(t_m,X_mov[:,0,2],color=c_muscle[2],linewidth=line_thck)
        axs[1,1].plot(t_m,X_mov[:,0,3],color=c_muscle[3],linewidth=line_thck)
        axs[1,1].plot(t_m,X_mov[:,0,4],color=c_muscle[4],linewidth=line_thck)
        axs[2,1].plot(t_m,X_mov[:,0,5],color=c_muscle[5],linewidth=line_thck)
        axs[2,1].plot(t_m,X_mov[:,0,6],color=c_muscle[6],linewidth=line_thck)
        axs[2,1].plot(t_m,X_mov[:,0,7],color=c_muscle[7],linewidth=line_thck)
        axs[3,1].plot(t_m,X_mov[:,0,8],color=c_muscle[8],linewidth=line_thck)
        axs[3,1].plot(t_m,X_mov[:,0,9],color=c_muscle[9],linewidth=line_thck)
        axs[3,1].plot(t_m,X_mov[:,0,10],color=c_muscle[10],linewidth=line_thck)
        axs[3,1].plot(t_m,X_mov[:,0,11],color=c_muscle[11],linewidth=line_thck)
        axs[4,1].plot(t_m,X_mov[:,0,12]*100.0+150.0,color=c_muscle[12],linewidth=line_thck)
        axs[0,0].set_ylabel('b')
        axs[1,0].set_ylabel('i')
        axs[2,0].set_ylabel('iii')
        axs[3,0].set_ylabel('hg')
        axs[4,0].set_ylabel('f')
        axs[0,0].set_xlim([0,1])
        axs[1,0].set_xlim([0,1])
        axs[2,0].set_xlim([0,1])
        axs[3,0].set_xlim([0,1])
        axs[4,0].set_xlim([0,1])
        axs[0,0].set_ylim([-0.2,1.5])
        axs[1,0].set_ylim([-0.2,1.5])
        axs[2,0].set_ylim([-0.2,1.5])
        axs[3,0].set_ylim([-0.2,1.5])
        axs[4,0].set_ylim([150,250])
        axs[0,1].set_xlim([t_zoom_start,t_zoom_end])
        axs[1,1].set_xlim([t_zoom_start,t_zoom_end])
        axs[2,1].set_xlim([t_zoom_start,t_zoom_end])
        axs[3,1].set_xlim([t_zoom_start,t_zoom_end])
        axs[4,1].set_xlim([t_zoom_start,t_zoom_end])
        axs[0,1].set_ylim([-0.2,1.5])
        axs[1,1].set_ylim([-0.2,1.5])
        axs[2,1].set_ylim([-0.2,1.5])
        axs[3,1].set_ylim([-0.2,1.5])
        axs[4,1].set_ylim([150,250])
        #plt.tight_layout()
        adjust_spines(axs[0,0],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,0],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[2,0],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[3,0],['left'],yticks=[0,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[4,0],['left'],yticks=[150,200,250],linewidth=0.8,spineColor='k')
        adjust_spines(axs[5,0],['left'],yticks=[-90,0,90],linewidth=0.8,spineColor='k')
        adjust_spines(axs[6,0],['left'],yticks=[-45,0,45],linewidth=0.8,spineColor='k')
        adjust_spines(axs[7,0],['left'],yticks=[-90,0,90],linewidth=0.8,spineColor='k')
        adjust_spines(axs[8,0],['left','bottom'],yticks=[-45,0,45],xticks=[0,0.5,1],linewidth=0.8,spineColor='k')
        adjust_spines(axs[0,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[1,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[2,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[3,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[4,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[5,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[6,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[7,1],[],linewidth=0.8,spineColor='k')
        adjust_spines(axs[8,1],['bottom'],xticks=[t_zoom_start,0.5,t_zoom_end],linewidth=0.8,spineColor='k')
        #plt.tight_layout()
        plot_fldr = '/home/flythreads/Documents/publication_figures/reconstruction'
        os.chdir(plot_fldr)
        file_name = 'mov_'+str(mov_ind)+'.svg'
        fig.savefig(file_name, dpi=400)
        plt.close()



