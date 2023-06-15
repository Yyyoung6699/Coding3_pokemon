import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configuration Parameters
class Config():
    result_save_path = 'results_3/' # Path to save generated images
    d_net_path = 'snapshots_3/dnet.pth' # Path to save discriminator network weights
    g_net_path = 'snapshots_3/gnet.pth' # Path to save generator network weights
    img_path = 'dataset/pokemon' # Path to the source image files
    img_size = 96 # Size of the cropped images
    batch_size = 256 # Batch size
    max_epoch = 5000 # Maximum number of training epochs
    noise_dim = 100 # Dimension of the initial noise vector
    feats_channel = 64 # Dimension of the intermediate feature maps

opt = Config() # Instantiate the Config class

if not os.path.exists('results_3'):
    os.mkdir('results_3') # Create 'results_3' directory if it doesn't exist
if not os.path.exists('snapshots_3'):
    os.mkdir('snapshots_3') # Create 'snapshots_3' directory if it doesn't exist

# Generator Network Design
class Gnet(nn.Module):
    def __init__(self, opt):
        super(Gnet, self).__init__()
        self.feats = opt.feats_channel
        self.generate = nn.Sequential(
            # input = (n, c, h, w) = (256, 100, 1, 1)
            nn.ConvTranspose2d(in_channels=opt.noise_dim, out_channels=self.feats * 8, kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.feats * 8),
            nn.ReLU(inplace=True),
            # deconv = (input - 1) * stride + k - 2 * padding = (1-1)*1 + 4-0 = 4
            # output = (256, 800, 1, 1)

            nn.ConvTranspose2d(in_channels=self.feats * 8, out_channels=self.feats * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.ReLU(inplace=True),

            # deconv = (input - 1) * stride + k - 2 * padding = (4-1)*2 + 4-2 = 8

            nn.ConvTranspose2d(in_channels=self.feats * 4, out_channels=self.feats * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.ReLU(inplace=True),

            # deconv = (input - 1) * stride + k - 2 * padding = (8-1)*2 + 4-2 = 16

            nn.ConvTranspose2d(in_channels=self.feats * 2, out_channels=self.feats, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(self.feats),
            nn.ReLU(inplace=True),

            # deconv = (input - 1) * stride + k - 2 * padding = (16-1)*2 + 4-2 = 32

            nn.ConvTranspose2d(in_channels=self.feats, out_channels=3, kernel_size=5, stride=3, padding=1, bias=False),

            nn.Tanh(),
            # deconv = (input - 1) * stride + k - 2 * padding = (32-1)*3 + 5-2 = 96
            # output = (n, c, h, w) = (256, 3, 96, 96)
        )

    def forward(self, x):
        return self.generate(x)
#----------------------------------3. Discriminator Network Design -----------------------------#

class Dnet(nn.Module):
    def __init__(self, opt):
        super(Dnet, self).__init__()
        self.feats = opt.feats_channel
        self.discrim = nn.Sequential(
            # input = (n, c, h, w) = (256, 3, 96, 96)
            nn.Conv2d(in_channels=3, out_channels=self.feats, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # conv = (input - k + 2 * padding ) / stride + 1 = (256 - 5 + 2) / 3 + 1 = 128

            nn.Conv2d(in_channels=self.feats, out_channels=self.feats * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feats * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=self.feats * 2, out_channels=self.feats * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feats * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=self.feats * 4, out_channels=self.feats * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feats * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=self.feats * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=True),
            nn.Sigmoid()
            # output = (n, c, h, w) = (256, 1, 1, 1)
        )

    def forward(self, x):
        return self.discrim(x).view(-1)

g_net, d_net = Gnet(opt), Dnet(opt) # Instantiate the Gnet and Dnet classes

#---------------------------------4. Data Loading and Preparation---------------------------------#

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(opt.img_size), # Resize image
    torchvision.transforms.CenterCrop(opt.img_size), # Center crop image
    torchvision.transforms.ToTensor() # Convert to tensor and normalize to [0,1]
])

dataset = torchvision.datasets.ImageFolder(root=opt.img_path, transform=transforms)

dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    num_workers=0,
    drop_last=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use GPU if available, else use CPU

g_net.to(device)
d_net.to(device)

optimize_g = torch.optim.Adam(g_net.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimize_d = torch.optim.Adam(d_net.parameters(), lr=2e-4, betas=(0.5, 0.999))

criterions = nn.BCELoss().to(device) # Binary cross-entropy loss

true_labels = torch.ones(opt.batch_size).to(device) # Label for real images as 1
fake_labels = torch.zeros(opt.batch_size).to(device) # Label for fake images as 0

noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device) # Random noise for generator input

test_noises = torch.randn(opt.batch_size, opt.noise_dim, 1, 1).to(device) # Random noise for testing

#-----------------------------------5. Start Training----------------------------------#

try:
    g_net.load_state_dict(torch.load(opt.g_net_path)) # Load pre-trained weights
    d_net.load_state_dict(torch.load(opt.d_net_path))
    print('Weights loaded successfully, continuing training.')
except:
    print('Failed to load weights, starting training from scratch.')

for epoch in range(opt.max_epoch): # Total number of epochs

    for iteration, (img, _) in tqdm(enumerate(dataloader)): # Iterate over the dataset
        real_img = img.to(device)

        if iteration % 5 == 0: # Train discriminator 5 times, then train generator 1 time

            optimize_d.zero_grad() # Clear discriminator gradients

            output = d_net(real_img) # Real data input to discriminator
            d_real_loss = criterions(output, true_labels) # Discriminator loss for real images with label 1
            fake_image = g_net(noises.detach()).detach() # Generate fake images using the generator
            output = d_net(fake_image) # Fake data input to discriminator
            d_fake_loss = criterions(output, fake_labels) # Discriminator loss for fake images with label 0
            d_loss = (d_fake_loss + d_real_loss) / 2 # Combined discriminator loss

            d_loss.backward() # Backpropagate discriminator loss
            optimize_d.step() # Update discriminator optimizer

        if iteration % 1 == 0:
            optimize_g.zero_grad() # Clear generator gradients
            noises.data.copy_(torch.randn(opt.batch_size, opt.noise_dim, 1, 1)) # Copy noise data for generator input
            fake_image = g_net(noises) # Generate fake images
            output = d_net(fake_image) # Fake data input to discriminator
            g_loss = criterions(output, true_labels) # Generator loss, aiming to generate images classified as real

            g_loss.backward() # Backpropagate generator loss
            optimize_g.step() # Update generator optimizer

    vid_fake_image = g_net(test_noises) # Generate random noise for testing
    torchvision.utils.save_image(vid_fake_image.data[:16], "%s/%s.png" % (opt.result_save_path, epoch), normalize=True) # Save the first 16 images
    torch.save(d_net.state_dict(), opt.d_net_path) # Save discriminator weights
    torch.save(g_net.state_dict(), opt.g_net_path) # Save generator weights
    print('Epoch:', epoch, '---D-loss:---', d_loss.item(), '---G-loss:---', g_loss.item()) # Visualize loss

