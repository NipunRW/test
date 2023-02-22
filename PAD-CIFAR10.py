# %%
### Import ###
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm

# %%
### Dataset (CIFAR10) ###

batch_size = 512

transform = transforms.Compose(
                               [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                               )

dataset = torchvision.datasets.ImageNet() #CIFAR10(root="./data",
                                       train=True, download=True,
                                       transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

### Neural Networks ###

## Paramators

# Number of channel of the image
nc = 3

# Size of the latent space
nz = 256

# Size of the feature map passing through the Generator
ngf = 64

# Size of the feature map passing through the Discriminator
ndf = 64

# Number of the available GPU
ngpu = 1


### Generator (+Decoder) ###

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf=ngf, img_channels=3):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.bias = True

        # input is Z, going into a convolution
        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*4, kernel_size=4, stride=1, padding=0, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ngf*4) x 4 x 4
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ngf*2) x 8 x 8
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # state size. (ngf) x 16 x 16
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(ngf, img_channels, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.Tanh()
        )

    def forward(self, input, reverse=True):
        fc1 = input.view(input.size(0), input.size(1), 1, 1)
        tconv1 = self.tconv1(fc1)
        tconv2 = self.tconv2(tconv1)
        tconv3 = self.tconv3(tconv2)
        output = self.tconv4(tconv3)
        if reverse:
            output = grad_reverse(output)
        return output


### Other NN ###

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -1)

def grad_reverse(x):
    return GradReverse.apply(x)

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


### Discriminator (+Encoder) ###

class Discriminator(nn.Module):
    def __init__(self, ngpu, nz, ndf=ndf, img_channels=3, p_drop=0.0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.bias = True


        # input is (3) x 32 x 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels, ndf, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # state size. (64) x 16 x 16
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # state size. (ndf*2) x 8 x 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, inplace=True),
        )


        # # state size. (ndf*4) x 4 x 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, nz, kernel_size=4,stride=2, padding=0, bias=self.bias),
            Flatten()
        )
        
        self.dis = nn.Sequential(
             nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=0, bias=self.bias),
             Flatten()
        )

        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        fc_dis = self.sigmoid(self.dis(conv3))
        fc_enc = self.conv4(conv3)
        realfake = fc_dis.view(-1, 1).squeeze(1)
        return fc_enc, realfake

### Setup NN ###


### nn initialization

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

### Settings before training

# setup networks
netG = Generator(ngpu, nz=nz, ngf=ngf, img_channels=nc)
netG.apply(weights_init)

netD = Discriminator(ngpu, nz=nz, ndf=ndf, img_channels=nc) # p_drop=opt.drop 一番後ろから除外
netD.apply(weights_init)

# send to GPU
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netD.to(device)
netG.to(device)

# hyper paramator (for this project, whic means witout 'opt' things)
lrD = 0.0002
lrG = 0.0002
batchSize = batch_size


# setup optimizer
beta1 = 0.5
optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

d_losses = []
g_losses = []
r_losses_real = []
r_losses_fake = []
kl_losses = []

# loss functions
dis_criterion = nn.BCELoss() # discriminator
rec_criterion = nn.MSELoss() # reconstruction

# tensor placeholders
dis_label = torch.zeros(batchSize, dtype=torch.float32, device=device)
real_label_value = 1.0
fake_label_value = 0

eval_noise = torch.randn(batch_size, nz, device=device)


### Things that are needed beforehand

def kl_loss(latent_output):
    m = torch.mean(latent_output, dim=0)
    s = torch.std(latent_output, dim=0)
    
    kl_loss = torch.mean((s ** 2 + m ** 2) / 2 - torch.log(s) - 1/2)
    return kl_loss

# Occlusion for NREM
class Occlude(object):
    def __init__(self, drop_rate=0.0, tile_size=7):
        self.drop_rate = drop_rate
        self.tile_size = tile_size

    def __call__(self, imgs, d=0):
        imgs_n = imgs.clone()
        if d==0:
            device='cpu'
        else:
            device = imgs.get_device()
            if device ==-1:
                device = 'cpu'
        mask = torch.ones((imgs_n.size(d), imgs_n.size(d+1), imgs_n.size(d+2)), device=device)  # only ones = no mask
        i = 0
        while i < imgs_n.size(d+1):
            j = 0
            while j < imgs_n.size(d+2):
                if np.random.rand() < self.drop_rate:
                    for k in range(mask.size(0)):
                        mask[k, i:i + self.tile_size, j:j + self.tile_size] = 0  # set to zero the whole tile
                j += self.tile_size
            i += self.tile_size
        
        imgs_n = imgs_n * mask  # apply the mask to each image
        return imgs_n

### Training ###

# hyper paramator
epochs = 50
epsilon = 0.0
W = 1.0
N = 1.0
R = 1.0
lmbd = 0.5  # ('--lmbd', type=float, default=0.5, help='convex combination factor for REM')

# store loss
store_loss_D = []
store_loss_G = []
store_loss_R_real = []
store_loss_R_fake = []
store_norm = []
store_kl = []


for epoch in tqdm(range(epochs)):


    for i, data in enumerate(dataloader, 0):

        ############################
        # Wake (W)
        ###########################
        # Discrimination wake
        optimizerD.zero_grad()
        optimizerG.zero_grad()
        real_image, label = data
        real_image, label = real_image.to(device), label.to(device)
        latent_output, dis_output = netD(real_image)
        latent_output_noise = latent_output #+ epsilon*torch.randn(batch_size, nz, device=device) # noise transformation
        dis_label[:] = real_label_value  # should be classified as real
        dis_errD_real = dis_criterion(dis_output, dis_label)
        if R > 0.0:  # if GAN learning occurs
            (dis_errD_real).backward(retain_graph=True)

        # KL divergence regularization
        kl = kl_loss(latent_output)
        (kl).backward(retain_graph=True)
        
        # reconstruction Real data space
        reconstructed_image = netG(latent_output_noise, reverse=False)
        rec_real = rec_criterion(reconstructed_image, real_image)
        if W > 0.0:
            (W*rec_real).backward()
        optimizerD.step()
        optimizerG.step()
        # compute the mean of the discriminator output (between 0 and 1)
        D_x = dis_output.cpu().mean()
        latent_norm = torch.mean(torch.norm(latent_output.squeeze(), dim=1)).item()
        
        
        
        ###########################
        # NREM perturbed dreaming (N)
        ##########################
        optimizerD.zero_grad()
        latent_z = latent_output.detach()
        
        with torch.no_grad():
            nrem_image = netG(latent_z)
            occlusion = Occlude(drop_rate=random.random(), tile_size=random.randint(1,8))
            occluded_nrem_image = occlusion(nrem_image, d=1)
        latent_recons_dream, _ = netD(occluded_nrem_image)
        rec_fake = rec_criterion(latent_recons_dream, latent_output.detach())
        if N > 0.0:
            (N * rec_fake).backward()
        optimizerD.step()

     


        ###########################
        # REM adversarial dreaming (R)
        ##########################

        optimizerD.zero_grad()
        optimizerG.zero_grad()
        lmbd = lmbd
        noise = torch.randn(batch_size, nz, device=device)
        if i==0:
            #latent_z = 0.5*latent_output.detach() + 0.5*noise
            latent_z = latent_output.detach()
        else:
            #latent_z = 0.25*latent_output.detach() + 0.25*old_latent_output + 0.5*noise
            latent_z = 0.5*latent_output.detach() + 0.5*old_latent_output
        
        dreamed_image_adv = netG(latent_z, reverse=True) # activate plasticity switch
        latent_recons_dream, dis_output = netD(dreamed_image_adv)
        dis_label[:] = fake_label_value # should be classified as fake
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        if R > 0.0: # if GAN learning occurs
            dis_errD_fake.backward(retain_graph=True)
            optimizerD.step()
            optimizerG.step()
        dis_errG = - dis_errD_fake

        D_G_z1 = dis_output.cpu().mean()

        old_latent_output = latent_output.detach()
        
        
        
        ###########################
        # Compute average losses
        ###########################
        store_loss_G.append(dis_errG.item())
        store_loss_D.append((dis_errD_fake + dis_errD_real).item())
        store_loss_R_real.append(rec_real.item())
        store_loss_R_fake.append(rec_fake.item()) ## NREM thing
        store_norm.append(latent_norm)
        store_kl.append(kl.item())
        
    print('[%d/%d] loss_G = %.4f , loss_D = %.4f , loss_R_real = %.4f , loss_R_fake = %.4f , norm = %.4f , kl = %.4f ' 
          % (epoch+1, epochs, dis_errG.item(), (dis_errD_fake + dis_errD_real).item(), rec_real.item(), rec_fake.item(), latent_norm, kl.item()))


### Plot Loss ###

plt.figure(figsize=(20,10))
plt.title("Loss During Training")
plt.plot(store_loss_G,label="G")
plt.plot(store_loss_D,label="D")
plt.plot(store_loss_R_real,label="R_real")
plt.plot(store_loss_R_fake,label="R_fake")
plt.plot(store_norm,label="norm")
plt.plot(store_kl,label="kl")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

### Plot Figures ###

fig = plt.figure(figsize=(30, 12))
for i in range(0, 5):
    # 第一段に訓練データを
    plt.subplot(4, 5, i+1)
    plt.title('Input Real Image No.%d' % (i+1), fontsize = 20, y=-0.15)
    plt.imshow(((real_image[i].permute(1,2,0))/2+0.5).cpu().detach().numpy())
    # 第二段に復元データを
    plt.subplot(4, 5, 5+i+1)
    plt.title('Wake Reconstructed Image No.%d' % (i+1) , fontsize = 20, y=-0.15)
    plt.imshow(((reconstructed_image[i].permute(1,2,0))/2+0.5).cpu().detach().numpy())
    # 第三段にNREMデータを表示する
    plt.subplot(4, 5, 10+i+1)
    plt.title('NREM Perturbed Dream No.%d' % (i+1), fontsize = 20, y=-0.15)
    plt.imshow(((occluded_nrem_image[i].permute(1,2,0))/2+0.5).cpu().detach().numpy())
    # 第四段に生成データを表示する
    plt.subplot(4, 5, 15+i+1)
    plt.title('REM Adversarial Dream No.%d' % (i+1), fontsize = 20, y=-0.15)
    plt.imshow(((dreamed_image_adv[i].permute(1,2,0))/2+0.5).cpu().detach().numpy())