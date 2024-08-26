# -*- coding: utf-8 -*-
"""
Python3 ICA_HCP.py
If you need install dependency libraries, please use pip3 rather than pip

Created on  May.3 2024
@author: Qiang Li
"""
from scipy.io import loadmat, savemat
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import FastICA
from scipy import signal
from image import *
import cv2
from scipy import sparse
from scipy.sparse import *
from sklearn.preprocessing import scale
import h5py

#%%
def is_symmetric(m):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    m : array or sparse matrix
    A square matrix.

    Returns
    -------
    check : bool
    The check result.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check

def estimate_k(corr_matrix, th=2, n_voxels=None):
    """
    Estimates k using z-score based method
    """
    z_matrix = scale(corr_matrix, axis=1)
    z_matrix[z_matrix < th] = 0
    k = np.count_nonzero(z_matrix) / n_voxels
   
def remove_poor_connected_voxels(corr_matrix, n_voxels):
    """
    Optional step to remove poorly conencted voxels. Voxels that are not
    well connected with other voxels are considered to be not meaningful/noise.
    """
    print('Removing poorly connected voxels ...')
    degree = np.sum(corr_matrix, axis=0)
    degree_ind = np.argsort(degree)
    mask = [degree_ind[i] for i in range(int(n_voxels/100),n_voxels)]
    corr_matrix = corr_matrix[mask, :]
    corr_matrix = corr_matrix[:, mask]
    n_mask_voxels = corr_matrix.shape[0]

def knn_corr_matrix(n_mask_voxels, corr_matrix, k):
    """
    Build a sparse k nearest neighbors correlation matrix from the full correlation matrix
    """
    print ('Computing k nearest neighbor correlation matrix ...')
    for i in range(n_mask_voxels):
        row = corr_matrix[i, :]
        row_sorted = np.sort(row)
        row[row < row_sorted[-k]] = 0
        corr_matrix[i, :] = row

    corr_matrix = sparse.csr_matrix(corr_matrix)

    corr_copy = corr_matrix.copy()
    corr_copy.data = np.ones(corr_copy.nnz)

    # either
    corr_copy += corr_copy.transpose()

    corr_copy.data = 1.0 / corr_copy.data

    corr_matrix += corr_matrix.transpose()
    corr_matrix = corr_matrix.multiply(corr_copy)

    corr_matrix.eliminate_zeros()

    assert is_symmetric(corr_matrix)

    
def GaussianMatrix(X,sigma):
    G = torch.mm(X, X.T)
    K = 2*G-(torch.diag(G).reshape([1,G.size()[0]]))
    K = 1/(2*sigma**2)*(K-(torch.diag(G).reshape([G.size()[0],1])))
    K = torch.exp(K)

    return K


def MMI8(variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8,sigma1,alpha):
    input1 = variable1
    K_x = GaussianMatrix(input1,sigma1)/(input1.size(dim=0))
    L_x,_ = torch.symeig(K_x,eigenvectors=True)
    lambda_x = torch.abs(L_x)
    
    #lambda_x = L_x
    H_x = (1/(1-alpha))*torch.log2((torch.sum(lambda_x ** alpha)))
    
    
    
    input2 = variable2
    K_y = GaussianMatrix(input2,sigma1)/(input2.size(dim=0))
    L_y,_ = torch.symeig(K_y,eigenvectors=True)
    lambda_y = torch.abs(L_y)
    #lambda_y = L_y
    H_y = (1/(1-alpha))*torch.log2((torch.sum(lambda_y ** alpha)))
    
    
    
    input3 = variable3
    K_z = GaussianMatrix(input3,sigma1)/(input3.size(dim=0))
    L_z,_ = torch.symeig(K_z,eigenvectors=True)
    lambda_z = torch.abs(L_z)
    #lambda_y = L_y
    H_z = (1/(1-alpha))*torch.log2((torch.sum(lambda_z ** alpha)))
    
    input4 = variable4
    K_4 = GaussianMatrix(input4,sigma1)/(input4.size(dim=0))
    L_4,_ = torch.symeig(K_4,eigenvectors=True)
    lambda_4 = torch.abs(L_4)
    #lambda_y = L_y
    H_4 = (1/(1-alpha))*torch.log2((torch.sum(lambda_4 ** alpha)))
    
    input5 = variable5
    K_5 = GaussianMatrix(input5,sigma1)/(input5.size(dim=0))
    L_5,_ = torch.symeig(K_5,eigenvectors=True)
    lambda_5 = torch.abs(L_5)
    #lambda_y = L_y
    H_5 = (1/(1-alpha))*torch.log2((torch.sum(lambda_5 ** alpha)))
    
    input6 = variable6
    K_6 = GaussianMatrix(input6,sigma1)/(input6.size(dim=0))
    L_6,_ = torch.symeig(K_6,eigenvectors=True)
    lambda_6 = torch.abs(L_6)
    #lambda_y = L_y
    H_6 = (1/(1-alpha))*torch.log2((torch.sum(lambda_6 ** alpha)))
    
    input7 = variable7
    K_7 = GaussianMatrix(input7,sigma1)/(input7.size(dim=0))
    L_7,_ = torch.symeig(K_7,eigenvectors=True)
    lambda_7 = torch.abs(L_7)
    #lambda_y = L_y
    H_7 = (1/(1-alpha))*torch.log2((torch.sum(lambda_7 ** alpha)))
    
    input8 = variable8
    K_8 = GaussianMatrix(input8,sigma1)/(input8.size(dim=0))
    L_8,_ = torch.symeig(K_8,eigenvectors=True)
    lambda_8 = torch.abs(L_8)
    #lambda_y = L_y
    H_8 = (1/(1-alpha))*torch.log2((torch.sum(lambda_8 ** alpha)))
    

    
    K_xyz = K_x*K_y*K_z*K_4*K_5*K_6*K_7*K_8*(input1.size(dim=0))
    K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
    L_xyz,_ = torch.symeig(K_xyz,eigenvectors=True)
    lambda_xyz = torch.abs(L_xyz)
    #lambda_xy = L_xy
    H_xyz =  (1/(1-alpha))*torch.log2((torch.sum(lambda_xyz ** alpha)))
    
    #mutual_information = H_x + H_y - H_xy
    return H_x+H_y+H_z+H_4+H_5+H_6+H_7+H_8 - H_xyz



def MMI5(variable1,variable2,variable3,variable4,variable5,sigma1,alpha):
    input1 = variable1
    K_x = GaussianMatrix(input1,sigma1)/(input1.size(dim=0))
    L_x,_ = torch.symeig(K_x,eigenvectors=True)
    lambda_x = torch.abs(L_x)
    
    #lambda_x = L_x
    H_x = (1/(1-alpha))*torch.log2((torch.sum(lambda_x ** alpha)))
    
    
    
    input2 = variable2
    K_y = GaussianMatrix(input2,sigma1)/(input2.size(dim=0))
    L_y,_ = torch.symeig(K_y,eigenvectors=True)
    lambda_y = torch.abs(L_y)
    #lambda_y = L_y
    H_y = (1/(1-alpha))*torch.log2((torch.sum(lambda_y ** alpha)))
    
    
    
    input3 = variable3
    K_z = GaussianMatrix(input3,sigma1)/(input3.size(dim=0))
    L_z,_ = torch.symeig(K_z,eigenvectors=True)
    lambda_z = torch.abs(L_z)
    #lambda_y = L_y
    H_z = (1/(1-alpha))*torch.log2((torch.sum(lambda_z ** alpha)))
    
    input4 = variable4
    K_4 = GaussianMatrix(input4,sigma1)/(input4.size(dim=0))
    L_4,_ = torch.symeig(K_4,eigenvectors=True)
    lambda_4 = torch.abs(L_4)
    #lambda_y = L_y
    H_4 = (1/(1-alpha))*torch.log2((torch.sum(lambda_4 ** alpha)))
    
    input5 = variable5
    K_5 = GaussianMatrix(input5,sigma1)/(input5.size(dim=0))
    L_5,_ = torch.symeig(K_5,eigenvectors=True)
    lambda_5 = torch.abs(L_5)
    #lambda_y = L_y
    H_5 = (1/(1-alpha))*torch.log2((torch.sum(lambda_5 ** alpha)))
    
    
    K_xyz = K_x*K_y*K_z*K_4*K_5*(input1.size(dim=0))
    K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
    L_xyz,_ = torch.symeig(K_xyz,eigenvectors=True)
    lambda_xyz = torch.abs(L_xyz)
    #lambda_xy = L_xy
    H_xyz =  (1/(1-alpha))*torch.log2((torch.sum(lambda_xyz ** alpha)))
    
    #mutual_information = H_x + H_y - H_xy
    return H_x+H_y+H_z+H_4+H_5 - H_xyz

# def MMI(variable1,variable2,variable3,variable4,variable5,variable6,variable7,variable8,sigma1,alpha):
#     input1 = variable1
#     K_1 = GaussianMatrix(input1,sigma)/(input1.size(dim=0))
#     L_1,_ = torch.symeig(K_1,eigenvectors=True)
#     lambda_1 = torch.abs(L_1)
#     H_1 = (1/(1-alpha))*torch.log((torch.sum(lambda_1 ** alpha)))
    
    
#     input2 = variable2
#     K_2 = GaussianMatrix(input2,sigma)/(input2.size(dim=0))
#     L_2,_ = torch.symeig(K_2,eigenvectors=True)
#     lambda_2 = torch.abs(L_2)
#     H_2 = (1/(1-alpha))*torch.log((torch.sum(lambda_2 ** alpha)))
    
    
#     input3 = variable3
#     K_3 = GaussianMatrix(input3,sigma)/(input3.size(dim=0))
#     L_3,_ = torch.symeig(K_3,eigenvectors=True)
#     lambda_3 = torch.abs(L_3)
#     H_3 = (1/(1-alpha))*torch.log((torch.sum(lambda_3 ** alpha)))
    
#     input4 = variable4
#     K_4 = GaussianMatrix(input4,sigma)/(input4.size(dim=0))
#     L_4,_ = torch.symeig(K_4,eigenvectors=True)
#     lambda_4 = torch.abs(L_4)
#     H_4 = (1/(1-alpha))*torch.log((torch.sum(lambda_4 ** alpha)))
    
#     input5 = variable5
#     K_5 = GaussianMatrix(input5,sigma)/(input5.size(dim=0))
#     L_5,_ = torch.symeig(K_5,eigenvectors=True)
#     lambda_5 = torch.abs(L_5)
#     H_5 = (1/(1-alpha))*torch.log((torch.sum(lambda_5 ** alpha)))
    
#     input6 = variable6
#     K_6 = GaussianMatrix(input6,sigma)/(input6.size(dim=0))
#     L_6,_ = torch.symeig(K_6,eigenvectors=True)
#     lambda_6 = torch.abs(L_6)
#     H_6 = (1/(1-alpha))*torch.log((torch.sum(lambda_6 ** alpha)))
    
    
#     input7 = variable7
#     K_7 = GaussianMatrix(input7,sigma)/(input7.size(dim=0))
#     L_7,_ = torch.symeig(K_7,eigenvectors=True)
#     lambda_7 = torch.abs(L_7)
#     H_7 = (1/(1-alpha))*torch.log((torch.sum(lambda_7 ** alpha)))
    
#     input8 = variable8
#     K_8 = GaussianMatrix(input8,sigma)/(input8.size(dim=0))
#     L_8,_ = torch.symeig(K_8,eigenvectors=True)
#     lambda_8 = torch.abs(L_8)
#     H_8 = (1/(1-alpha))*torch.log((torch.sum(lambda_8 ** alpha)))
    

    
#     K_xyz = K_1*K_2*K_3*K_4*K_5*K_6*K_7*K_8*(input1.size(dim=0))
#     K_xyz = K_xyz / torch.sum(torch.diag(K_xyz))
    
#     L_xyz,_ = torch.symeig(K_xyz,eigenvectors=True)
#     lambda_xyz = torch.abs(L_xyz)
#     #lambda_xy = L_xy
#     H_xyz =  (1/(1-alpha))*torch.log((torch.sum(lambda_xyz ** alpha)))
    
#     #mutual_information = H_x + H_y - H_xy
#     return H_1+H_2+H_3+H_4+H_5+H_6+H_7+H_8 - H_xyz

#source = loadmat('HCP_DeepICA_AverPC.mat')
#source_data = source['HCP_C']

with h5py.File('HCPPCA_DeepICA.mat', 'r') as f:
    source_data = f['SMs'][:]
print('Check Voxel-Correlation Size:')
print(source_data.shape)
#source_data[source_data<0.6]=0
#k=1742 #estimate_k(source_data, th=2, n_voxels=68235)
#n_mask_voxels=remove_poor_connected_voxels(source_data, n_voxels=68235)
#print(n_mask_voxels)
#SparseCp=knn_corr_matrix(n_mask_voxels, source_data, k)
X_all = source_data.reshape(source_data.shape[0]*source_data.shape[1],1)
print('Check Voxel-Correlation Size:')
print(X_all.shape)

w=X_all.shape[0]
h=X_all.shape[1]


input_dim = 1
output_dim = 8
output_dim_2 = 5
h_n = 4096

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(4, 8, kernel_size=6, stride=3)
#        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
#        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
#        #self.conv4 = nn.Conv2d(8, 4, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(input_dim, int(h_n/1))
        self.norm1 = nn.BatchNorm1d(h_n)
        self.fc2 = nn.Linear(int(h_n/1), int(h_n/1))
        self.norm2 = nn.BatchNorm1d(h_n)
        self.fc3 = nn.Linear(h_n, int(h_n/1))
        self.norm3 = nn.BatchNorm1d(h_n)
        self.fc4 = nn.Linear(int(h_n/1), int(h_n/1))
        self.norm4 = nn.BatchNorm1d(h_n)
        self.fc5 = nn.Linear(int(h_n/1), h_n)
        self.norm5 = nn.BatchNorm1d(h_n)
        self.fc6 = nn.Linear(h_n, int(h_n/1))
        self.norm6 = nn.BatchNorm1d(h_n)
        self.fc7 = nn.Linear(h_n, 128)
        self.norm7 = nn.BatchNorm1d(128)
        
        self.fc8 = nn.Linear(128, output_dim)
        self.fc9 = nn.Linear(128, output_dim_2)
        
    def single_power_step(self, A, x):
        x = torch.matmul(A, x)
        x = x/torch.norm(x)
        return x
    def alt_matrix_power(self, A, x, power):
        iter_count_tf = 0
        #condition  = lambda it, A, x: it< power
        #body = lambda it, A, x: (it+1, A, )
        #loop_vars = [iter_count_tf, A, x]
        it = 0
        while it<power:
            it+=1
            x = self.single_power_step(A, x)
        
        
        #output = tf.while_loop(condition, body, loop_vars)[2]
        e = torch.norm(torch.matmul(A, x))
        return x, e
    
    def alt_power_whitening(self, input_tensor, output_dim, n_iterations=250, **kwargs):
        R = torch.empty([output_dim,output_dim]).normal_(mean=0,std=1).cuda()
        W = torch.zeros([output_dim,output_dim]).cuda()
        input_tensor - input_tensor.mean(0)[None,:]
        C = torch.matmul(input_tensor.T, input_tensor)/input_tensor.shape[0]
        iter_count_tf = 0
        condition = lambda it, C, W, R: it<output_dim
        it = 0
        while it<output_dim:
            v, l = self.alt_matrix_power(C, R[:, it, None], n_iterations)
            it+=1
            C = C - l * torch.matmul(v, v.T)
            W = W + 1 / torch.sqrt(l) * torch.matmul(v, v.T)
        whitened_output = torch.matmul(input_tensor, W.T)
        return whitened_output, W, input_tensor.mean(0), C
        
        

    def forward(self, x):
        #x = self.alt_power_whitening(x, output_dim)[0]
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.relu(self.norm3(self.fc3(x)))
        x = F.relu(self.norm4(self.fc4(x)))
        x = F.relu(self.norm5(self.fc5(x)))
        x = F.relu(self.norm6(self.fc6(x)))
        x = F.relu(self.norm7(self.fc7(x)))
        
        #x = F.sigmoid(self.fc8(x))
        x1 = self.fc8(x)
        x1 = F.softmax(x1,-1)
        x1 = self.alt_power_whitening(x1, output_dim)[0]
        
        x2 = self.fc9(x)
        x2 = F.softmax(x2,-1)
        x2 = self.alt_power_whitening(x2, output_dim_2)[0]
        #x2 = F.softmax(x2,-1)
        return x1,x2
       
model = Net()
model = nn.DataParallel(model)
model.to(device)   


N =1000
crition = nn.MSELoss()
loss_list = []
current_loss_old =1000000

params = list(model.parameters())# + list(model2.parameters())
optimizer = optim.Adam(params, lr=0.0001)
k= 0 
show_id = 2
for ep in range(3):
    range_all = list(range(len(X_all)))
    permuted_all = list(np.random.permutation(range_all))
    X_all_shuffle = X_all[permuted_all]

    for it in range(int(len(X_all)/N)):
        xinput = X_all_shuffle[int(it*N):int(it*N)+N].astype(np.float32)
        #print('-------------------Xinput---------------------')
        #print(xinput.shape)
        optimizer.zero_grad()
        xinput = torch.tensor(xinput).cuda().to(device)
        latent_x, latent_x2= model(xinput) 
        savemat('/home/users/qli27/Downloads/DDICA-main/HCP_IC8/Latent.mat',{'latent_x':latent_x.cpu().detach().numpy(), 'latent_x2':latent_x2.cpu().detach().numpy()})
        sigma = 0.1#(N)**(-1/(4+(1)))
        alpha = 1.01#1.01
        
        if show_id%2 == 0: 
            loss = MMI8(latent_x[:,0].unsqueeze(-1), latent_x[:,1].unsqueeze(-1),latent_x[:,2].unsqueeze(-1),latent_x[:,3].unsqueeze(-1),latent_x[:,4].unsqueeze(-1)
                    ,latent_x[:,5].unsqueeze(-1),latent_x[:,6].unsqueeze(-1),latent_x[:,7].unsqueeze(-1),sigma,alpha)
        else:
            loss = MMI5(latent_x2[:,0].unsqueeze(-1), latent_x2[:,1].unsqueeze(-1),latent_x2[:,2].unsqueeze(-1),latent_x2[:,3].unsqueeze(-1),latent_x2[:,4].unsqueeze(-1)
                    ,sigma,alpha)
        loss.backward()
        optimizer.step()
        current_loss = loss.cpu().detach().numpy()
        print('--------------current_loss----------------')
        print(current_loss)
        loss_list.append(current_loss)
        k+=1
        print('-------------------k----------------------')
        print(k)
        if k%120 == 0:
            print(k)

            print([current_loss])
            if show_id%2 == 0:
                show = np.zeros([X_all.shape[0],output_dim])
            else:
                show = np.zeros([X_all.shape[0],output_dim_2])
                
            for iij in range(int(len(X_all)/N)):
                
                X_sample = X_all_shuffle[iij:-1:int(int(w*h)/N)].astype(np.float32)#X.astype(np.float32)#
                X_sample = torch.tensor(X_sample).cuda().to(device)
                if show_id%2 == 0:
                    show_si,_ = model(X_sample)
                    show_si = show_si.cpu().detach().numpy()
                else:
                    _,show_si = model(X_sample)#.detach().numpy()
                    show_si = show_si.cpu().detach().numpy()
        
                show[iij:-1:int(int(w*h)/N)] = show_si
            if show_id%2 == 0:
                show_real = np.zeros([X_all.shape[0],output_dim])
            else:
                show_real = np.zeros([X_all.shape[0],output_dim_2])
                
            for i in range(len(show)):
                show_real[permuted_all[i]] =  show[i]
            show = show_real
            
            show_list = []
            for j in range(len(show)):
                if 1:
                    show_list.append(show[j])
            show_si =np.array(show_list)     
            
            
            if show_id%2 == 0:
                end_num = output_dim
            else:
                end_num = output_dim_2
            for endm in range(end_num):
                 show_si[:,endm]= (show_si[:,endm]-show_si[:,endm].min())/(show_si[:,endm].max()-show_si[:,endm].min())#(show_si[:,0]-show_si[:,0].mean())/show_si[:,0].std()
           
            show = show_si
            
            
            if show_id%2 == 0:            
                asa = np.max(show,-1)
                aaa= show - np.repeat(np.expand_dims(asa,-1),8,-1)
                aaa[aaa ==0] =1
                aaa[aaa<=0] = 0
                show = aaa
                   
                if 1:#os_i == 0:
                  index_list = [0,1,2,3,4,5,6,7]
                
                save_img = np.concatenate((show[:,index_list[0]].reshape(w,h),show[:,index_list[1]].reshape(w,h),show[:,index_list[2]].reshape(w,h),show[:,index_list[3]].reshape(w,h),show[:,index_list[4]].reshape(w,h),show[:,index_list[5]].reshape(w,h),show[:,index_list[6]].reshape(w,h),show[:,index_list[7]].reshape(w,h)),-1)
              
                save_show = np.zeros(show.shape)
                for idm in range(8):
                    save_show[:,idm] = show[:,index_list[idm]]
                  
    
                savemat('/home/users/qli27/Downloads/DDICA-main/HCP_IC8/current_results.mat',{'current_results':save_show})
                savemat('/home/users/qli27/Downloads/DDICA-main/HCP_IC8/current_loss.mat',{'current_loss':current_loss})
                
              
            else:
                
                if 1:#os_i == 0:
                  index_list = [0,1,2,3,4]                
                
                save_img = np.concatenate((show[:,index_list[0]].reshape(w,h),show[:,index_list[1]].reshape(w,h),show[:,index_list[2]].reshape(w,h),show[:,index_list[3]].reshape(w,h),show[:,index_list[4]].reshape(w,h)),-1)
                
       
                save_show = np.zeros(show.shape)
                for idm in range(5):
                    save_show[:,idm] = show[:,index_list[idm]]
                
                savemat('/home/users/qli27/Downloads/DDICA-main/HCP_IC8/current_results.mat',{'current_results':save_show})
                savemat('/home/users/qli27/Downloads/DDICA-main/HCP_IC8/current_loss.mat',{'current_loss':current_loss})
                
                
        

    
