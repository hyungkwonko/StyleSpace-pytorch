# https://github.com/xrenaa/StyleSpace-pytorch/blob/main/StyleSpace_FFHQ.ipynb

from model import Generator
import torch
import numpy as np
from PIL import Image

config = {"latent" : 512, "n_mlp" : 8, "channel_multiplier": 2}
generator = Generator(
        size= 1024,
        style_dim=config["latent"],
        n_mlp=config["n_mlp"],
        channel_multiplier=config["channel_multiplier"]
    )

generator.load_state_dict(torch.load("checkpoint/stylegan2-ffhq-config-f.pt")['g_ema'], strict=False)
generator.eval()
generator.cuda()

print(generator)

from torch.nn import functional as F

index = [0,1,1,2,2,3,4,4,5,6,6,7,8,8,9,10,10,11,12,12,13,14,14,15,16,16]

def conv_warper(layer, input, style, noise):
    # the conv should change
    conv = layer.conv
    batch, in_channel, height, width = input.shape

    style = style.view(batch, 1, in_channel, 1, 1)  # reshape (e.g., 512 --> 1,512,1,1)
    weight = conv.scale * conv.weight * style

    if conv.demodulate:
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(batch, conv.out_channel, 1, 1, 1)

    weight = weight.view(
        batch * conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
    )

    if conv.upsample:
        input = input.view(1, batch * in_channel, height, width)
        weight = weight.view(
            batch, conv.out_channel, in_channel, conv.kernel_size, conv.kernel_size
        )
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, conv.out_channel, conv.kernel_size, conv.kernel_size
        )
        out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        out = conv.blur(out)

    elif conv.downsample:
        input = conv.blur(input)
        _, _, height, width = input.shape
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)

    else:
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=conv.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, conv.out_channel, height, width)
        
    out = layer.noise(out, noise=noise)
    out = layer.activate(out)
    
    return out
    

def encoder(G, noise):
    # an encoder warper for G
    styles = [noise]  # (1, 512)
    style_space = []
    
    styles = [G.style(s) for s in styles]

    noise = [getattr(G.noises, 'noise_{}'.format(i)) for i in range(G.num_layers)]
    inject_index = G.n_latent
    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)  # (18, 512)
    style_space.append(G.conv1.conv.modulation(latent[:, 0]))   # ()

    i = 1

    # EqualLinear layers to fit the channel dimension (e.g., 512 --> 64)
    for conv1, conv2 in zip(G.convs[::2], G.convs[1::2]):
        style_space.append(conv1.conv.modulation(latent[:, i]))
        style_space.append(conv2.conv.modulation(latent[:, i+1]))
        # print(latent[:, i].shape)
        # print(conv1.conv.modulation(latent[:, i]).shape)
        i += 2
    return style_space, latent, noise


def decoder(G, style_space, latent, noise):
    # an decoder warper for G
    out = G.input(latent)
    out = conv_warper(G.conv1, out, style_space[0], noise[0])
    skip = G.to_rgb1(out, latent[:, 1])

    i = 1
    for conv1, conv2, noise1, noise2, to_rgb in zip(
        G.convs[::2], G.convs[1::2], noise[1::2], noise[2::2], G.to_rgbs
    ):
        out = conv_warper(conv1, out, style_space[i], noise=noise1)
        out = conv_warper(conv2, out, style_space[i+1], noise=noise2)
        skip = to_rgb(out, latent[:, i + 2], skip)

        i += 2

    image = skip

    return image


def save_fig(output, name):
    output = (output + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    output = output[0].detach().cpu().permute(1,2,0).numpy()
    output = (output*255).astype(np.uint8)
    im = Image.fromarray(output).resize((256,256), Image.ANTIALIAS)
    im.save(f"results/img_{name}.png")


# default image generation
test_input = torch.randn(1, 512).cuda()
output, _ = generator([test_input], False)
save_fig(output, 'default')

# eye
style_space, latent, noise = encoder(generator, test_input)
style_space[index[9]][:, 409] += 10
image = decoder(generator, style_space, latent, noise)
save_fig(image, 'eye')


# hair
style_space, latent, noise = encoder(generator, test_input)
style_space[index[12]][:, 330] -= 50
image = decoder(generator, style_space, latent, noise)
save_fig(image, 'hair')


# mouth
style_space, latent, noise = encoder(generator, test_input)
style_space[index[6]][:, 259] -= 20
image = decoder(generator, style_space, latent, noise)
save_fig(image, 'mouth')

# lip
style_space, latent, noise = encoder(generator, test_input)
style_space[index[15]][:, 45] -= 3
image = decoder(generator, style_space, latent, noise)
save_fig(image, 'lip')

print("generation complete...!")