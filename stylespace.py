import os
import argparse
import numpy as np
from PIL import Image
from model import Generator
from tqdm import tqdm

import torch
from torch.nn import functional as F


def conv_warper(layer, input, style, noise):
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
        i += 2
    return style_space, latent, noise


def decoder(G, style_space, latent, noise):
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


def generate_img(generator, input, layer_no, channel_no, degree=30):
    style_space, latent, noise = encoder(generator, input)  # len(style_space) = 17
    style_space[index[layer_no]][:, channel_no] += degree
    image = decoder(generator, style_space, latent, noise)
    return image


def save_fig(output, name, size=128):
    output = (output + 1)/2
    output = torch.clamp(output, 0, 1)
    if output.shape[1] == 1:
        output = torch.cat([output, output, output], 1)
    output = output[0].detach().cpu().permute(1,2,0).numpy()
    output = (output*255).astype(np.uint8)
    im = Image.fromarray(output).resize((size,size), Image.ANTIALIAS)
    im.save(name)


if __name__ =='__main__':

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default="checkpoint/stylegan2-ffhq-config-f.pt")
    parser.add_argument("--out_dir", type=str, default='sample')
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--save_all_attr", type=int, default=0)

    args = parser.parse_args()

    generator = Generator(size= 1024, style_dim=args.latent, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier)
    generator.load_state_dict(torch.load(args.ckpt)['g_ema'], strict=False)
    generator.eval()
    generator.cuda()

    print(generator)

    index = [0,1,1,2,2,3,4,4,5,6,6,7,8,8,9,10,10,11,12,12,13,14,14,15,16,16]
    s_channel = [
        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
        512, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64, 32, 32
        ]

    os.makedirs(args.out_dir, exist_ok=True)

    # default image generation
    torch.manual_seed(args.seed)
    input = torch.randn(1, args.latent).cuda()
    image, _ = generator([input], False)
    save_fig(image, os.path.join(args.out_dir, f'{str(args.seed).zfill(6)}_default.png'))

    if args.save_all_attr:
        # 1. SAVE_ALL ATTR MANIPUlATION RESULT: Let's find out
        # TAKES SOME TIME
        for ix in range(len(index)):
            os.makedirs(os.path.join(args.out_dir, ix), exist_ok=True)
            for i in tqdm(range(s_channel[ix])):
                image = generate_img(generator, input, layer_no=ix, channel_no=i, degree=30)
                save_fig(image, os.path.join(args.out_dir, ix, f'{str(args.seed).zfill(6)}_{ix}_{i}.png'))
    else:
        # 2. MANIPULATE SPECIFIC ATTRIBUTE
        # pose (?)
        for i in [-30, -10, 10, 30]:
            image = generate_img(generator, input, layer_no=3, channel_no=95, degree=i)
            save_fig(image, os.path.join(args.out_dir, f'{str(args.seed).zfill(6)}_pose_{i}.png'))

        # eye
        image = generate_img(generator, input, layer_no=9, channel_no=409, degree=10)
        save_fig(image, os.path.join(args.out_dir, f'{str(args.seed).zfill(6)}_eye.png'))

        # hair
        image = generate_img(generator, input, layer_no=12, channel_no=330, degree=-50)
        save_fig(image, os.path.join(args.out_dir, f'{str(args.seed).zfill(6)}_hair.png'))

        # mouth
        image = generate_img(generator, input, layer_no=6, channel_no=259, degree=-20)
        save_fig(image, os.path.join(args.out_dir, f'{str(args.seed).zfill(6)}_mouth.png'))

        # lip
        image = generate_img(generator, input, layer_no=15, channel_no=45, degree=-3)
        save_fig(image, os.path.join(args.out_dir, f'{str(args.seed).zfill(6)}_lip.png'))

    print("generation complete...!")