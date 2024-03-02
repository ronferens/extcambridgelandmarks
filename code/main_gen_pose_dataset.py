import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from os import makedirs
from os.path import join, splitext, exists, basename, dirname
import pandas as pd


# Setting the instructionPix2Pix pipeline
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id,
                                                              torch_dtype=torch.float16,
                                                              safety_checker=None)
pipe.enable_attention_slicing()
pipe.to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

scenes = ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']

for scene in scenes:
    # Setting the input files to process
    labels_files = f'/home/dev/git/hyperpose/datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{scene}_test.csv'
    dataset_folder = '/media/dev/data/datasets/cambridge'

    output_postfix = ['winter', 'evening', 'summer']
    prompts = ['make it looks like winter',
               'make it looks like evening',
               'make it looks like summer with very sunny sky']
    guidance_scales = [1.2, 1.2, 1.5]
20
    for i, prompt in enumerate(prompts):
        output_file_postfix = output_postfix[i]
        output_folder = f'/media/dev/data/datasets/cambridge_gen/{output_file_postfix}'

        df = pd.read_csv(labels_files)
        df_gen = df.copy()

        # Main loop
        for index, filename in enumerate(df['img_path']):
            image = PIL.Image.open(join(dataset_folder, filename))
            org_h, org_w = image.size
            image = image.resize((org_h // 2, org_w // 2), PIL.Image.BICUBIC)

            images = pipe(prompt, image=image, num_inference_steps=25,
                          image_guidance_scale=guidance_scales[i]).images

            out_img = images[0].resize((org_h, org_w))

            output_filename = join(dirname(filename), splitext(basename(filename))[0] + '_' + output_file_postfix + splitext(basename(filename))[1])
            df_gen.loc[index, 'img_path'] = output_filename

            # Creating the output folder if it doesn't exist
            output_path = join(output_folder, dirname(filename))
            if not exists(output_path):
                makedirs(output_path)
            out_img.save(join(output_folder, output_filename))

        labels_file_output = splitext(basename(labels_files))[0] + '_' + output_file_postfix + splitext(basename(labels_files))[1]
        df_gen.to_csv(labels_file_output)
