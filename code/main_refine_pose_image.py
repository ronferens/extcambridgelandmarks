import PIL
import numpy as np
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from os.path import join, splitext, basename, exists
from os import makedirs


model_id = 'timbrooks/instruct-pix2pix'
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
                                                              torch_dtype=torch.float16,
                                                              safety_checker=None)
pipe.enable_attention_slicing()
pipe.to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


# Creating the output folder if it not exists
output_folder = 'outputs'
if not exists(output_folder):
    makedirs(output_folder)

# Setting the input image
dataset_folder = '/media/dev/data/datasets/cambridge'
scene = 'StMarysChurch'
seq_num = 13
frames = [314]

# Setting the prompt and its parameters
output_file_postfix = 'summer'
prompt = 'make it looks like a sunny day in the summer'# with bright skies'
guidance_scales = 2.0

for frame_name in frames:
    filename = join(dataset_folder, f'{scene}/seq{seq_num}/frame{frame_name:05d}.png')

    # Generating the output image
    image = PIL.Image.open(filename)
    org_h, org_w = image.size
    image = image.resize((org_h // 2, org_w // 2))
    images = pipe(prompt, image=image, num_inference_steps=25, image_guidance_scale=guidance_scales).images
    out_img = images[0].resize((org_h, org_w))

    # out_img.show()

    # Saving the generated image
    output_filename = join(output_folder, splitext(basename(filename))[0] + '_' + output_file_postfix + splitext(basename(filename))[1])
    out_img.save(output_filename)