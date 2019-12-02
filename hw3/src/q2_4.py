import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from utils import *
import time

def q2_4(MovieID, UserID, I_input, movie_features_matrix,MovieID_max,UserID_max,num,option,test,eval):
    options=[]

    if option == 1:
        options.append(num)
    else:
        options = [2, 5, 10, 20, 50]

    for latent_num in options:
        start = time.time()
        R=Variable(torch.from_numpy(np.array(movie_features_matrix.todense()-1)/4).type('torch.FloatTensor')).cuda()
        I = Variable(torch.from_numpy(np.array(I_input)).type('torch.FloatTensor')).cuda()
        U = Variable(torch.normal(torch.zeros((latent_num,UserID_max)),torch.ones((latent_num,UserID_max))/(latent_num)), requires_grad = True ).cuda()
        V = Variable(torch.normal(torch.zeros((latent_num,MovieID_max)),torch.ones((latent_num,MovieID_max))/(latent_num)), requires_grad = True ).cuda()
        U = nn.Parameter(U)
        V = nn.Parameter(V)

        opt = optim.Adam([U,V],lr=3e-3,betas=(0.9,0.9999))
        print('fatorization-%d' % (latent_num))

        if test == 1 and eval == 0:
            f = open('test-predictions.txt','w')
        if test == 0 and eval == 1:
            f = open('dev-predictions.txt', 'w')
        if test == 0 and eval == 0:
            f = open('../eval/fatorization-%d.txt' % (latent_num), 'w')

        for i in range(2000):
            loss1 = (I*torch.pow((R-torch.matmul(U.transpose(0,1),V)),2)).sum()/2
            lambda_U = 1/torch.std(U).detach()
            lambda_V = 1/torch.std(V).detach()

            loss2 = lambda_U/2*torch.pow(U,2).sum()
            loss3 = lambda_V/2*torch.pow(V,2).sum()

            opt.zero_grad()
            loss=1*loss1+0.4*loss2+0.4*loss3
            loss.backward()
            opt.step()

        torch.save([U,V],'factorization-%d.pt'%(latent_num))
        Z=torch.matmul(U.transpose(0,1),V).data.cpu().numpy()
        Z=np.clip(Z*4+1,1,5)

        for i in range(len(UserID)):
            f.write('%f\n'%Z[UserID[i],MovieID[i]])
        f.close()
        print("time :", time.time() - start)