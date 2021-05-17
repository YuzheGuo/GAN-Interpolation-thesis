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
import normalization as n
import datetime

torch.cuda.device_count()
print("""import finished""")
#%%
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print("the time is: ", time_str)
torch.cuda.empty_cache()

epochNum = 100
batchSize = 64
displayStep = 10
GTrainNumber = 1  # the num of G trained to 1 D changed
maxGap = 10  # gap is the loss change amount

suspect_data_list = []  # if succeed gap, then append list
G_losses = []
D_losses = []

real_label = 1.
fake_label = 0.

lr = 0.0002
beta1 = 0.5

data_type = "PM25"  # PM25

realBase = 'data/{}_hourly_32/'.format(data_type)
sampleBase = 'data/{}_hourly_32_sample_by_station/'.format(data_type)
save_folder = "saved/{}_res_plot_32-{}".format(data_type, time_str)
os.mkdir(save_folder)
loss_save_path = "saved/{}-loss-32size-{}.npy".format(data_type, time_str)
suspect_data_list_save_path = "saved/{}-suspect-data-32size-{}.npy".format(data_type, time_str)

num_of_gpu = torch.cuda.device_count()
device = 'cuda' if num_of_gpu > 0 else 'cpu'

print("pm25 min: {}, pm25 max: {}".format(n.PM25_min, n.PM25_max))
print("O3 min: {}, O3 max: {}".format(n.O3_min, n.O3_max))
print('the data is {}, avaliable gpu is {}, device is: {}'.format(
    data_type, num_of_gpu, device))


# %%
# preprocess = transforms.Compose([
#     #transforms.Scale(256),
#     #transforms.CenterCrop(224),
#     transforms.Normalize(0, 1)
# ])
def plot_distribution(data_array: np.array, label=None, save_folder=None):
    '''
    输入二维的array，输出相应的面积图片
    label: the label if this photo
    save_folder: the folder to save
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
    if not save_folder:
        plt.show()
    else:
        path = "".join([save_folder, "/", "plot-{}.jpg".format(label)])
        plt.savefig(path)


def default_loader(path):
    '''given an input of path, return the tensor file'''
    arr = np.load(path)[np.newaxis, :]
    arr = (arr - n.O3_min) / (n.O3_max - n.O3_min)
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
        self.realList = [realBase + name for name in os.listdir(realBase)]
        self.sampleList = [
            sampleBase + name for name in os.listdir(sampleBase)
        ]
        self.loader = loader

    def __getitem__(self, index):
        real_img = self.loader(self.realList[index])
        sample_img = self.loader(self.sampleList[index])
        return real_img.float(), sample_img.float()

    def __len__(self):
        return len(self.realList)


#%%
DataSet = trainDataset(realBase, sampleBase)
trainLoader = DataLoader(DataSet, batch_size=batchSize, shuffle=True)

print("""the dataLoader finished""")


# print(data.size())
# %%
# the G, nc is number of channel, ngf is number of generater features
class Generator(nn.Module):
    def __init__(self, nc, ngf):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf), nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16 x 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8 x 128

        self.layer3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4), nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4 x 256
        # 4 x 4 x 256
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4,
                               ngf * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1), nn.BatchNorm2d(ngf * 2), nn.ReLU())
        # 8 x 8 x 128
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2,
                               ngf,
                               kernel_size=4,
                               stride=2,
                               padding=1), nn.BatchNorm2d(ngf), nn.ReLU())
        # 16 x 16 x 64
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid())
        # 32 x 32 x 1
    def forward(self, _cpLayer):
        out = self.layer1(_cpLayer)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


# for i, data in enumerate(trainLoader):
#     break
# netG = Generator(1, 64)
# netG(data[1])
# netG(data[1]).size()


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.layer1_real = nn.Sequential(
            nn.Conv2d(nc, ndf // 2, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(ndf/2),
            nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.layer1_sample = nn.Sequential(
            nn.Conv2d(nc, ndf // 2, kernel_size=4, stride=2, padding=1),
            #nn.BatchNorm2d(ndf/2),
            nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8

        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4

        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid())
        # 1

    def forward(self, real, sample):

        out_1 = self.layer1_real(real)
        out_2 = self.layer1_sample(sample)
        out = self.layer2(torch.cat((out_1, out_2), 1))
        out = self.layer3(out)
        out = self.layer4(out)
        return out.view(-1, )


# netD = Discriminator(1, 64)
# out = netD(data[0], data[1])
# out
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# %%
netG = Generator(nc=1, ngf=64).to(device)
netG = netG.apply(weights_init)
netD = Discriminator(nc=1, ndf=64).to(device)
netD = netD.apply(weights_init)

# %%
# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator

# Establish convention for real and fake labels during training
# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# %%
print("Starting Training Loop...")
print("epoch {}, batchsize {}".format(epochNum, batchSize))
print("train {} time of G to 1 time of D".format(GTrainNumber))
# For each epoch
count = 0
mean_list = []
for epoch in range(epochNum):
    # For each batch in the dataloader
    for i, data in enumerate(trainLoader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        # Format batch
        
        mean_list.append([data[0].mean(), data[1].mean()])

        if count == 289:
            # print(data)
            print(epoch, i)
            break

        count+=1
        if count%10==0: 
            print(count)
            print(mean_list[-1])
        #         Output training stats
        
        # Save Losses for plotting later
        

        #         # Check how the generator is doing by saving G's output on fixed_noise
        #         if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
        #             with torch.no_grad():
        #                 fake = netG(fixed_noise).detach().cpu()
        #             img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        # np.save('loss-g3.npy', np.array([G_losses, D_losses]))
#%%
plot_distribution(data[0][0][0])
# %%
data[0].mean()
# %%
x = [i[0] for i in mean_list]
y = [i[1] for i in mean_list]
plt.plot(y)

# %%
