import torch 
from config import *
from dataset import train_dataset,tensor_to_pil
import matplotlib.pyplot as plt 

# 前向diffusion计算参数
betas=torch.linspace(0.0001,0.02,T) # (T,)
alphas=1-betas  # (T,)
# print('betas:',betas)

# 累乘#
alphas_cumprod=torch.cumprod(alphas,dim=-1) # alpha_t累乘 (T,)    [a1,a2,a3,....] ->  [a1,a1*a2,a1*a2*a3,.....]
# print('alphas_cumprod.shape:',alphas_cumprod.shape) #torch.Size([100])

#计算方差时会用 #
alphas_cumprod_prev=torch.cat((torch.tensor([1.0]),alphas_cumprod[:-1]),dim=-1) # alpha_t-1累乘 (T,),  [1,a1,a1*a2,a1*a2*a3,.....a1*a2*....*a(t-1)]
variance=(1-alphas)*(1-alphas_cumprod_prev)/(1-alphas_cumprod)  # denoise用的方差   (T,)

# 执行前向加噪 #前向部分 #
def forward_diffusion(batch_x,batch_t): # batch_x: (batch,channel,width,height), batch_t: (batch_size,)
    # print(batch_x.shape) #torch.Size([2, 1, 48, 48])
    # print("##batch_t.shape:", batch_t.shape) ###batch_t.shape: torch.Size([2])

    #为每张图片配了一个高斯分布的噪音图，维度和每张图片一样，是 (batch,channel,width,height) #
    batch_noise_t=torch.randn_like(batch_x)   # 为每张图片生成第t步的高斯噪音，噪音图的维度是： (batch,channel,width,height) 
    # print("##batch_noise_t.shape:", batch_noise_t.shape) #torch.Size([2, 1, 48, 48])

    #每个样本所依赖的累乘alphas_cumprod_i是不一样的 # batch_t：是每张图片对应的时刻，将其当作一个mask去取元素 #
    batch_alphas_cumprod=alphas_cumprod.to(DEVICE)[batch_t].view(batch_x.size(0),1,1,1) #alphas_cumprod作用到每一个像素上 #广播到其他维度 #
    # print(alphas_cumprod.to(DEVICE)[batch_t].shape) #torch.Size([2])
    # print(batch_alphas_cumprod.shape) #torch.Size([2, 1, 1, 1])

    batch_x_t=torch.sqrt(batch_alphas_cumprod)*batch_x+torch.sqrt(1-batch_alphas_cumprod)*batch_noise_t # 基于公式直接生成第t步加噪后图片
    # print("##batch_x_t.shape:", batch_x_t.shape) #torch.Size([2, 1, 48, 48])
    return batch_x_t,batch_noise_t #(加完噪的图像（时刻多样）, 生成的噪音图像) #(模型的输入, 监督模型输出的label) #不同时刻，加噪程度不一样 #

if __name__=='__main__':
    batch_x=torch.stack((train_dataset[0][0],train_dataset[1][0]),dim=0).to(DEVICE) # 2个图片拼batch, (2,1,48,48)

    # 加噪前的样子
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(tensor_to_pil(batch_x[0]))
    plt.subplot(1,2,2)
    plt.imshow(tensor_to_pil(batch_x[1]))
    plt.show()

    batch_x=batch_x*2-1 # [0,1]像素值调整到[-1,1]之间,以便与高斯噪音值范围匹配 #原本的像素值范围是[0,1] #
    batch_t=torch.randint(0,T,size=(batch_x.size(0),)).to(DEVICE)  # 每张图片随机生成diffusion步数
    # batch_t = torch.tensor([5, 99], dtype=torch.long) #手动构造2个时刻 #
    #batch_t=torch.tensor([5,100],dtype=torch.long)
    # print('batch_t:',batch_t) #batch_t: tensor([57, 11]) #batch_t: tensor([71, 75]) #
    
    batch_x_t,batch_noise_t=forward_diffusion(batch_x,batch_t) #每张图像及其对应的时刻t #返回：加噪后的图像，以及对应的t-1时刻到t时刻加的噪音图像是多少 #
    # print('batch_x_t:',batch_x_t.size()) #torch.Size([2, 1, 48, 48])
    # print('batch_noise_t:',batch_noise_t.size()) #torch.Size([2, 1, 48, 48]) #模型的预测和此噪音图像求损失 #

    # 加噪后的样子
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(tensor_to_pil((batch_x_t[0]+1)/2))   
    plt.subplot(1,2,2)
    plt.imshow(tensor_to_pil((batch_x_t[1]+1)/2))
    plt.show()