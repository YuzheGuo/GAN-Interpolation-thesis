# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
from torchvision import transforms
import torch
import os
import torch.optim as optim
import matplotlib.pyplot as plt
torch.cuda.empty_cache()
print("""import finished""")

O3_min, O3_max = 0.0, 0.2475029081106186
pm25_min, pm25_max = 0.0, 422.7541

def plot_distribution(data_array: np.array, label=None, save_folder=None):
    '''
    输入二维的array，输出相应的面积图片
    cal the max and min for the color bar
    '''
    shape = data_array.shape
    x = np.arange(0, shape[1])  # len = 11
    y = np.arange(0, shape[0])  # len = 7

    fig, ax = plt.subplots(dpi=100)
    pcm = ax.pcolormesh(x,
                        y,
                        data_array,
                        shading='auto',
                        vmax=data_array.max(),
                        vmin=data_array.min(),
                        cmap='Blues')
    fig.colorbar(pcm, ax=ax)
    print(label)
    if not save_folder:
        plt.show()
    else:
        path = "".join([save_folder, "/", "plot-{}.jpg".format(label)])
        plt.savefig(path)


def default_loader(path):
    '''given an input of path, return the tensor file'''
    arr = np.load(path)[np.newaxis, :]
    arr = (arr - pm25_min) / (pm25_max - pm25_min)
    img_tensor = torch.from_numpy(arr)
    return img_tensor


def default_loader(path):
    '''given an input of path, return the tensor file'''
    arr = np.load(path)[np.newaxis, :]
    img_tensor = torch.from_numpy(arr)
    return img_tensor

# path = 'O3_numpy_split/20181215-3.npy'
# array = default_loader(path)

class trainDataset(Dataset):
    '''
    abstract class for the data set
    '''
    def __init__(self, realBase, sampleBase, loader=default_loader):
        #定义好 image 的路径
        self.realList = [realBase+name for name in os.listdir(realBase)]
        self.sampleList = [sampleBase+name for name in os.listdir(sampleBase)]                
        self.loader = loader

    def __getitem__(self, index):
        real_img = self.loader(self.realList[index])
        sample_img = self.loader(self.sampleList[index])
        return real_img, sample_img.float()

    def __len__(self):
        return len(self.realList)

print("""function difination finished""")
# %%
batchSize = 64
realBase = 'data/PM25_hourly/'
sampleBase = 'data/PM25_hourly_sample_by_station/'
print(os.listdir())
#%%
DataSet = trainDataset(realBase, sampleBase)
trainLoader = DataLoader(DataSet, batch_size=batchSize,shuffle=True)
print("""dataloader finised""")
# %%
# for i, data in enumerate(trainLoader):
#     pass

# print(data.size())
# %%
# the G, nc is number of channel, ngf is number of generater features
class Generator(nn.Module):
    def __init__(self, nc, ngf):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nc,ngf,kernel_size=4,stride=2,padding=(1,2)),
                                 nn.BatchNorm2d(ngf),
                                 nn.LeakyReLU(0.2,inplace=True))
        
        self.layer2 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
      
        
        self.layer3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
      
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
      
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())
        
        # the activate function has been modified to Sigmoid()
        self.layer6 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=(1,2)),
                                 nn.Sigmoid())
        
    def forward(self,_cpLayer, show_process = False):
        out = self.layer1(_cpLayer)
        if show_process: print(out.size())
        out = self.layer2(out)
        if show_process: print(out.size())
        out = self.layer3(out)
        if show_process: print(out.size())
        out = self.layer4(out)
        if show_process: print(out.size())
        out = self.layer5(out)
        if show_process: print(out.size())
        out = self.layer6(out)
        if show_process: print(out.size())
        return out


# for i, data in enumerate(trainLoader):
#     break
# netG = Generator(1, 64)
# netG(data[1])
# netG(data[1]).size()

class Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super(Discriminator,self).__init__()
        self.layer1_image = nn.Sequential(nn.Conv2d(nc,ndf//2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf//2),
                                 nn.LeakyReLU(0.2,inplace=True))
        
        self.layer1_cp = nn.Sequential(nn.Conv2d(nc,ndf//2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf//2),
                                 nn.LeakyReLU(0.2,inplace=True))
        
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))

        
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))

        
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,ndf*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*8),
                                 nn.LeakyReLU(0.2,inplace=True))
        # the final layer is Linear!

        self.layer5 = nn.Sequential(nn.Conv2d(ndf*8,1,kernel_size=(13, 16),stride=2),
                                 nn.Sigmoid())
        
    def forward(self,real_img,sample_img, show_process=False):
        out_1 = self.layer1_image(real_img)
        out_2 = self.layer1_cp(sample_img)        
        out = self.layer2(torch.cat((out_1,out_2),1))
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        if show_process: print(out.size())
        out = self.layer5(out).view(-1)
        return out

# netD = Discriminator(1, 64)
# out = netD(data[0], data[1])
# out

# %% [markdown]
# ## training

# %%
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# %%
netG = Generator(nc=1, ngf=64)
netG = netG.apply(weights_init)
netD = Discriminator(nc=1, ndf=64)
netD = netD.apply(weights_init)
print("""model init finished""")

# %%
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

lr = 0.0001
beta1 = 0.5
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# %%

# plot_distribution(O3_con)


# %%
# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
epochNum = 5
displayStep = 10
save_folder = "saved/PM25_res_plot_001rate"
loss_save_path = "saved/PM25-loss-64batchsize-001rate.npy"

print("Starting Training Loop...")
# For each epoch
for epoch in range(epochNum):
    # For each batch in the dataloader
    for i, data in enumerate(trainLoader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_img, sample_img = data[0], data[1]
        b_size = real_img.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float)
        # Forward pass real batch through D
        output = netD(real_img, sample_img)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        
        # Generate fake image batch with G
        fake = netG(sample_img)
        label.fill_(0)
        # Classify all fake batch with D
        output = netD(fake.detach(), sample_img.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        
        netG.zero_grad()
        label.fill_(1)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake, sample_img)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        
#         Output training stats
        if i % displayStep == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochNum, i, len(trainLoader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            plot_distribution(netG(sample_img)[0][0].detach(),
                              label="fake-epoch-{}-batch-{}".format(epoch, i),
                              save_folder=save_folder)
            plot_distribution(real_img[0][0],
                              label="real-epoch-{}-batch-{}".format(epoch, i),
                              save_folder=save_folder)
            
        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
#         # Check how the generator is doing by saving G's output on fixed_noise
#         if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
#             with torch.no_grad():
#                 fake = netG(fixed_noise).detach().cpu()
#             img_list.append(vutils.make_grid(fake, padding=2, normalize=True)


# %%

np.save(loss_save_path, np.array([G_losses, D_losses]))



