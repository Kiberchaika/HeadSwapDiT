# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import random
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers.models import FluxControlNetModel
from facexlib.recognition import init_recognition_model
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image

from .pipeline_flux_infusenet import FluxInfuseNetPipeline
from .resampler import Resampler


def seed_everything(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# modified from https://github.com/instantX-research/InstantID/blob/main/pipeline_stable_diffusion_xl_instantid.py
def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def extract_arcface_bgr_embedding(in_image, landmark, arcface_model=None, in_settings=None):
    kps = landmark
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
    arc_face_image = 2 * arc_face_image - 1
    arc_face_image = arc_face_image.cuda().contiguous()
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda')
    face_emb = arcface_model(arc_face_image)[0] # [512], normalized
    return face_emb


def resize_and_pad_image(source_img, target_img_size):
    # Get original and target sizes
    source_img_size = source_img.size
    target_width, target_height = target_img_size
    
    # Determine the new size based on the shorter side of target_img
    if target_width <= target_height:
        new_width = target_width
        new_height = int(target_width * (source_img_size[1] / source_img_size[0]))
    else:
        new_height = target_height
        new_width = int(target_height * (source_img_size[0] / source_img_size[1]))
    
    # Resize the source image using LANCZOS interpolation for high quality
    resized_source_img = source_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Compute padding to center resized image
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    
    # Create a new image with white background
    padded_img = Image.new("RGB", target_img_size, (255, 255, 255))
    padded_img.paste(resized_source_img, (pad_left, pad_top))
    
    return padded_img


class InfUFluxPipeline:
    def __init__(
            self, 
            base_model_path, 
            infu_model_path, 
            insightface_root_path = './',
            image_proj_num_tokens=8,
            infu_flux_version='v1.0',
            model_version='aes_stage2',
        ):

        self.infu_flux_version = infu_flux_version
        self.model_version = model_version
        
        # Load pipeline
        try:
            infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
            self.infusenet = FluxControlNetModel.from_pretrained(infusenet_path, torch_dtype=torch.bfloat16)
        except:
            print("No InfiniteYou model found. Downloading from HuggingFace `ByteDance/InfiniteYou` to `./models/InfiniteYou` ...")
            snapshot_download(repo_id='ByteDance/InfiniteYou', local_dir='./models/InfiniteYou', local_dir_use_symlinks=False)
            infu_model_path = os.path.join('./models/InfiniteYou', f'infu_flux_{infu_flux_version}', model_version)
            infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
            self.infusenet = FluxControlNetModel.from_pretrained(infusenet_path, torch_dtype=torch.bfloat16)
            insightface_root_path = './models/InfiniteYou/supports/insightface'
        try:
            pipe = FluxInfuseNetPipeline.from_pretrained(
                base_model_path,
                controlnet=self.infusenet,
                torch_dtype=torch.bfloat16,
            )
        except:
            try:
                pipe = FluxInfuseNetPipeline.from_single_file(
                    base_model_path,
                    controlnet=self.infusenet,
                    torch_dtype=torch.bfloat16,
                )
            except Exception as e:
                print(e)
                print('\nIf you are using `black-forest-labs/FLUX.1-dev` and have not downloaded it into a local directory, '
                      'please accept the agreement and obtain access at https://huggingface.co/black-forest-labs/FLUX.1-dev. '
                      'Then, use `huggingface-cli login` and your access tokens at https://huggingface.co/settings/tokens to authenticate. '
                      'After that, run the code again. If you have downloaded it, please use `base_model_path` to specify the correct path.')
                print('\nIf you are using other models, please download them to a local directory and use `base_model_path` to specify the correct path.')
                exit()
        pipe.to('cuda', torch.bfloat16)
        self.pipe = pipe

        # Load image proj model
        num_tokens = image_proj_num_tokens
        image_emb_dim = 512
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=4096,
            ff_mult=4,
        )
        image_proj_model_path = os.path.join(infu_model_path, 'image_proj_model.bin')
        ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu")
        image_proj_model.load_state_dict(ipm_state_dict['image_proj'])
        del ipm_state_dict
        image_proj_model.to('cuda', torch.bfloat16)
        image_proj_model.eval()

        self.image_proj_model = image_proj_model

        # Load face encoder
        self.app_640 = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))

        self.app_320 = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))

        self.app_160 = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app_160.prepare(ctx_id=0, det_size=(160, 160))

        self.arcface_model = init_recognition_model('arcface', device='cuda')

    def load_loras(self, loras):
        names, scales = [],[]
        for lora_path, lora_name, lora_scale in loras:
            if lora_path != "":
                print(f"loading lora {lora_path}")
                self.pipe.load_lora_weights(lora_path, adapter_name = lora_name)
                names.append(lora_name)
                scales.append(lora_scale)

        if len(names) > 0:
            self.pipe.set_adapters(names, adapter_weights=scales)

    def _detect_face(self, id_image_cv2):
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        
        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info

        face_info = self.app_160.get(id_image_cv2)
        return face_info

    def _prepare_mask_and_control(self, target_image_pil, target_face_info, target_size, blur_kernel_size=9, expand_ratio=0.3):
        """
        Prepares the inpainting mask and control hint (keypoints on black) for the target image.
        """
        width, height = target_size
        bbox = target_face_info['bbox']
        kps = target_face_info['kps']

        # Create mask
        mask = np.zeros((height, width), dtype=np.float32)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w_bbox = x2 - x1
        h_bbox = y2 - y1

        # Calculate expanded bounding box
        # Use expand_ratio + 0.1 for vertical expansion as per plan
        expand_w = w_bbox * expand_ratio
        expand_h = h_bbox * (expand_ratio + 0.1)

        new_x1 = np.clip(cx - w_bbox/2 - expand_w, 0, width - 1).astype(int)
        new_y1 = np.clip(cy - h_bbox/2 - expand_h, 0, height - 1).astype(int)
        new_x2 = np.clip(cx + w_bbox/2 + expand_w, 0, width - 1).astype(int)
        new_y2 = np.clip(cy + h_bbox/2 + expand_h, 0, height - 1).astype(int)

        # Fill the expanded rectangle in the mask
        mask[new_y1:new_y2, new_x1:new_x2] = 1.0

        # Ensure blur kernel size is odd and >= 1
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        blur_kernel_size = max(1, blur_kernel_size)

        # Apply Gaussian blur
        mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)

        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0) # Shape [1, 1, H, W]

        # Create control hint image (keypoints on black)
        generated_control_hint_image = Image.new('RGB', target_size, (0, 0, 0))
        generated_control_hint_image = draw_kps(generated_control_hint_image, kps)

        return mask_tensor, generated_control_hint_image

    def __call__(
        self,
        id_image: Image.Image,  # CONCEPT: Source Identity Image (PIL.Image.Image RGB)
        prompt: str,
        control_image: Optional[Image.Image] = None,  # CONCEPT: Target Context/Pose Image (PIL.Image.Image RGB) - Now Required
        width = 864,
        height = 1152,
        seed = 42,
        guidance_scale = 3.5,
        num_steps = 30,
        infusenet_conditioning_scale = 1.0,
        infusenet_guidance_start = 0.0,
        infusenet_guidance_end = 1.0,
        # --- NEW Explicit KWArgs ---
        blur_kernel_size=9,
        mask_expand_ratio=0.3,
        # **kwargs # Keep if originally present
    ):
        # --- Check for Target Image ---
        if control_image is None:
            raise ValueError("Target image (passed as 'control_image') is required for face swapping/inpainting.")

        # Ensure generator is created on the correct device
        device = self.pipe.device
        dtype = self.pipe.dtype # Usually torch.bfloat16
        generator = torch.Generator(device=device).manual_seed(seed)

        # --- Process id_image (Source) ---
        print('Preparing Source ID embeddings')
        id_image_cv2 = cv2.cvtColor(np.array(id_image), cv2.COLOR_RGB2BGR)
        face_info = self._detect_face(id_image_cv2)
        if len(face_info) == 0:
            raise ValueError("No face detected in source image (id_image)")

        # Select largest face based on bounding box area
        source_face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        landmark = source_face_info['kps']

        face_emb = extract_arcface_bgr_embedding(id_image_cv2, landmark, self.arcface_model) # [512]
        # Project ID embedding
        id_embed = self.image_proj_model(face_emb.unsqueeze(0).unsqueeze(1).to(device=device, dtype=dtype)) # [1, 1, 512] -> [1, num_tokens, 4096]

        # --- Process control_image (Target) ---
        print('Preparing Target context, mask, and control hint')
        target_image_pil = control_image.convert("RGB")
        target_image_cv2 = cv2.cvtColor(np.array(target_image_pil), cv2.COLOR_RGB2BGR)

        face_info_target = self._detect_face(target_image_cv2)
        if len(face_info_target) == 0:
            raise ValueError("No face detected in target image (control_image)")

        # Select largest face
        target_face_info = sorted(face_info_target, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]

        # Prepare mask and control hint image (kps on black)
        mask_tensor, generated_control_hint_image = self._prepare_mask_and_control(
            target_image_pil, target_face_info, target_size=(width, height),
            blur_kernel_size=blur_kernel_size, expand_ratio=mask_expand_ratio
        )
        mask_tensor = mask_tensor.to(device=device, dtype=dtype) # Move mask to device

        # Resize/Pad original control_image for VAE encoding
        # Using direct resize as specified in plan for target_image_processed
        target_image_processed = target_image_pil.resize((width, height), Image.LANCZOS)

        # Preprocess for VAE
        target_image_tensor = self.pipe.image_processor.preprocess(target_image_processed).to(device=device, dtype=self.pipe.vae.dtype) # Use VAE dtype

        # VAE Encode control_image to get initial latents
        with torch.no_grad():
            target_latent_dist = self.pipe.vae.encode(target_image_tensor).latent_dist
            target_image_latents = target_latent_dist.sample(generator=generator) # Sample latents
            # Keep latents UNPACKED here. Shift and scale.
            target_image_latents = (target_image_latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor


        # --- Call Underlying Pipeline (FluxInfuseNetPipeline) ---
        print("Calling underlying FluxInfuseNetPipeline for generation...")
        # Note: FluxInfuseNetPipeline __call__ needs modification in Phase 2 to accept target_image_latents and inpainting_mask
        image = self.pipe(
            prompt=prompt,
            controlnet_prompt_embeds=id_embed,           # Source ID embedding projected by Resampler
            control_image=generated_control_hint_image,  # Target kps drawn on black (ControlNet input)
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            controlnet_guidance_scale=1.0,               # ControlNet scale for kps hint (kept from original intention likely)
            controlnet_conditioning_scale=infusenet_conditioning_scale, # Scale for InfuseNet branch (ID embedding)
            control_guidance_start=infusenet_guidance_start,
            control_guidance_end=infusenet_guidance_end,
            generator=generator,
            # --- NEW Args Passed for Inpainting ---
            target_image_latents=target_image_latents, # Unpacked initial target latents [B, C, H/8, W/8]
            inpainting_mask=mask_tensor,               # Mask [B, 1, H, W]
            # Pass other relevant args like negative_prompt if applicable (assuming not needed based on original __call__)
        ).images[0]

        print("Generation complete.")
        return image
