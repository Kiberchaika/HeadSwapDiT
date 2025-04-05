import sys
sys.path.append('KV-Edit')

import os
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from PIL import ExifTags, Image
import torch
import gradio as gr
import numpy as np
from flux.sampling import prepare
from flux.util import (configs, load_ae, load_clip, load_t5)
from models.kv_edit import Flux_kv_edit

# --- Helper function to update model paths ---
def update_model_paths(model_name, base_path="./models/KVEdit", flux_specific_path="./models/FLUX.1-dev"):
    """Updates paths in the global configs dictionary for the specified model."""
    if model_name not in configs:
        print(f"Warning: Model name '{model_name}' not found in configs.")
        return

    # config is likely a ModelSpec object, not a dict
    config_spec = configs[model_name]
    os.makedirs(base_path, exist_ok=True) # Ensure the base directory exists

    # Define expected components and their potential ATTRIBUTE names in the ModelSpec
    # NOTE: These attribute names are assumptions. Adjust if the actual ModelSpec structure differs.
    component_map = {
        'transformer': ['transformer', 'model'], # Main model component
        'ae': ['ae', 'autoencoder'],           # Autoencoder
        'clip': ['text_encoder', 'clip'],       # CLIP text encoder
        't5': ['text_encoder_t5', 't5']         # T5 text encoder
    }

    print(f"Attempting to update model paths for '{model_name}' (ModelSpec) using base_path='{base_path}'...")

    # Update path for each component
    for component_type, potential_attribute_names in component_map.items():
        found_attr_name = None
        component_config = None # This should hold the config dict/object FOR the component
        for attr_name in potential_attribute_names:
            if hasattr(config_spec, attr_name):
                found_attr_name = attr_name
                component_ref = getattr(config_spec, attr_name) # Get the component config object/dict

                # We still assume the component's config itself might be a dictionary
                # or an object where we can set a 'path' attribute/key.
                # If component_ref IS the path string directly, this needs adjustment.
                if isinstance(component_ref, dict):
                     component_config = component_ref
                elif hasattr(component_ref, 'path'): # Check if it's an object with a path attribute
                    # In this case, component_config will represent the object itself
                    # and we'll try setting its 'path' attribute later.
                    component_config = component_ref
                else:
                    # Handle cases where the attribute might hold the path directly.
                    print(f"Info: Attribute '{attr_name}' on ModelSpec for '{model_name}' is neither a dict nor has a 'path' attribute. Assuming direct path/ID? Type: {type(component_ref)}")
                    # If it's a string, we might need to update it via setattr on config_spec
                    # setattr(config_spec, found_attr_name, new_path)
                    # Deferring this unless necessary and structure is known.
                    pass
                break # Found the attribute for this component type

        if found_attr_name is None:
            # print(f"Warning: Could not find attribute for component type '{component_type}' in ModelSpec for '{model_name}'.")
            continue # Component might not exist for this model

        # Determine the correct path
        # Use component's 'name' field if available, otherwise use the attribute name found
        component_name_for_path = found_attr_name # Default name
        if isinstance(component_config, dict) and 'name' in component_config:
            component_name_for_path = component_config.get('name', found_attr_name)
        elif hasattr(component_config, 'name'):
             component_name_for_path = getattr(component_config, 'name', found_attr_name)

        default_component_path = os.path.join(base_path, component_name_for_path)

        # Use specific path for the main flux-dev model's transformer component
        if component_type == 'transformer' and model_name == 'flux-dev':
             final_path = flux_specific_path
             # Ensure the specific directory exists if different from base_path
             os.makedirs(os.path.dirname(final_path), exist_ok=True)
             print(f"  Using specific path for flux-dev transformer: {final_path}")
        else:
             final_path = default_component_path
             print(f"  Using default path for {component_type} ({component_name_for_path}): {final_path}")


        # Update the path in the component's config
        if isinstance(component_config, dict):
            if 'path' in component_config:
                component_config['path'] = final_path
                # print(f"  Updated {model_name}.{found_attr_name}['path'] to: {final_path}")
            else:
                print(f"  Warning: Could not find 'path' key in {model_name}.{found_attr_name} dict. Path may not be updated.")
        elif hasattr(component_config, 'path'):
             try:
                 setattr(component_config, 'path', final_path)
                 # print(f"  Updated {model_name}.{found_attr_name}.path to: {final_path}")
             except AttributeError:
                 print(f"  Warning: Attribute 'path' on {model_name}.{found_attr_name} is not writable. Path may not be updated.")
        else:
             # This case corresponds to the 'Info' message above - the attribute didn't seem
             # like a config object/dict. We might need setattr(config_spec, found_attr_name, final_path) here
             # if the attribute itself IS the path.
              print(f"  Skipping path update for {model_name}.{found_attr_name} as its structure is unclear or lacks a 'path' field/attribute.")


# --- End Helper function ---

@dataclass
class SamplingOptions:
    source_prompt: str = ''
    target_prompt: str = ''
    width: int = 1366
    height: int = 768
    inversion_num_steps: int = 0
    denoise_num_steps: int = 0
    skip_step: int = 0
    inversion_guidance: float = 1.0
    denoise_guidance: float = 1.0
    seed: int = 42
    re_init: bool = False
    attn_mask: bool = False
    attn_scale: float = 1.0

class FluxEditor_kv_demo:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.offload = args.offload

        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"

        self.output_dir = 'regress_result'

        self.t5 = load_t5(self.device, max_length=256 if self.name == "flux-schnell" else 512)
        self.clip = load_clip(self.device)
        self.model = Flux_kv_edit(device="cpu" if self.offload else self.device, name=self.name)
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.device)

        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()
        self.info = {}
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)
        
    @torch.inference_mode()
    def inverse(self, brush_canvas,
             source_prompt, target_prompt, 
             inversion_num_steps, denoise_num_steps, 
             skip_step, 
             inversion_guidance, denoise_guidance,seed,
             re_init, attn_mask
             ):
        self.z0 = None
        self.zt = None
        # self.info = {}
        # gc.collect()
        if 'feature' in self.info:
            key_list = list(self.info['feature'].keys())
            for key in key_list:
                del self.info['feature'][key]
        self.info = {}
        
        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:,:,:3]
        shape = init_image.shape        
        height = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        width = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:height, :width, :]
        rgba_init_image = rgba_init_image[:height, :width, :]

        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=0,# no skip step in inverse leads chance to adjest skip_step in edit
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask
        )
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        torch.cuda.empty_cache()
        
        if opts.attn_mask:
            rgba_mask = brush_canvas["layers"][0][:height, :width, :]
            mask = rgba_mask[:,:,3]/255
            mask = mask.astype(int)
        
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device)
        else:
            mask = None
        
        self.init_image = self.encode(init_image, self.device).to(self.device)

        t0 = time.perf_counter()

        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        with torch.no_grad():
            inp = prepare(self.t5, self.clip,self.init_image, prompt=opts.source_prompt)
        
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)
        self.z0,self.zt,self.info = self.model.inverse(inp,mask,opts)
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            
        t1 = time.perf_counter()
        print(f"inversion Done in {t1 - t0:.1f}s.")
        return None

        
        
    @torch.inference_mode()
    def edit(self, brush_canvas,
             source_prompt, target_prompt, 
             inversion_num_steps, denoise_num_steps, 
             skip_step, 
             inversion_guidance, denoise_guidance,seed,
             re_init, attn_mask,attn_scale
             ):
        
        torch.cuda.empty_cache()
        
        rgba_init_image = brush_canvas["background"]
        init_image = rgba_init_image[:,:,:3]
        shape = init_image.shape        
        height = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        width = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16
        init_image = init_image[:height, :width, :]
        rgba_init_image = rgba_init_image[:height, :width, :]

        rgba_mask = brush_canvas["layers"][0][:height, :width, :]
        mask = rgba_mask[:,:,3]/255
        mask = mask.astype(int)
        
        rgba_mask[:,:,3] = rgba_mask[:,:,3]//2
        masked_image = Image.alpha_composite(Image.fromarray(rgba_init_image, 'RGBA'), Image.fromarray(rgba_mask, 'RGBA'))
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16).to(self.device)
        
        seed = int(seed)
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            inversion_num_steps=inversion_num_steps,
            denoise_num_steps=denoise_num_steps,
            skip_step=skip_step,
            inversion_guidance=inversion_guidance,
            denoise_guidance=denoise_guidance,
            seed=seed,
            re_init=re_init,
            attn_mask=attn_mask,
            attn_scale=attn_scale
        )
        if self.offload:
            
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            
        torch.manual_seed(opts.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)

        t0 = time.perf_counter()

        with torch.no_grad():
            inp_target = prepare(self.t5, self.clip, self.init_image, prompt=opts.target_prompt)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)
            
        x = self.model.denoise(self.z0,self.zt,inp_target,mask,opts,self.info)
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)
            
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x.to(self.device))
        
        x = x.clamp(-1, 1)
        x = x.float().cpu()
        x = rearrange(x[0], "c h w -> h w c")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        output_name = os.path.join(self.output_dir, "img_{idx}.jpg")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 0
        else:
            fns = [fn for fn in iglob(output_name.format(idx="*")) if re.search(r"img_[0-9]+\.jpg$", fn)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
            else:
                idx = 0
        
        fn = output_name.format(idx=idx)
    
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
    
        exif_data[ExifTags.Base.ImageDescription] = target_prompt
        img.save(fn, exif=exif_data, quality=95, subsampling=0)
        masked_image.save(fn.replace(".jpg", "_mask.png"),  format='PNG')
        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        
        print("End Edit")
        return img

    
    @torch.inference_mode()
    def encode(self,init_image, torch_device):
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
        init_image = init_image.unsqueeze(0) 
        init_image = init_image.to(torch_device)
        self.ae.encoder.to(torch_device)
        
        init_image = self.ae.encode(init_image).to(torch.bfloat16)
        return init_image
    
def create_demo(model_name: str):
    editor = FluxEditor_kv_demo(args)
    is_schnell = model_name == "flux-schnell"
    
    title = r"""
        <h1 align="center">🎨 KV-Edit: Training-Free Image Editing for Precise Background Preservation</h1>
        """
        
    description = r"""
        <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/Xilluill/KV-Edit' target='_blank'><b>KV-Edit: Training-Free Image Editing for Precise Background Preservation</b></a>.<br>
    
        💫💫 <b>Here is editing steps:</b> <br>
        1️⃣ Upload your image that needs to be edited. <br>
        2️⃣ Fill in your source prompt and click the "Inverse" button to perform image inversion. <br>
        3️⃣ Use the brush tool to draw your mask area. <br>
        4️⃣ Fill in your target prompt, then adjust the hyperparameters. <br>
        5️⃣ Click the "Edit" button to generate your edited image! <br>
        
        🔔🔔 [<b>Important</b>] Less skip steps, "re_init" and "attn_mask"  will enhance the editing performance, making the results more aligned with your text but may lead to discontinuous images.  <br>
        If you fail because of these three, we recommend trying to increase "attn_scale" to increase attention between mask and background.<br>
        """
    article = r"""
    If our work is helpful, please help to ⭐ the <a href='https://github.com/Xilluill/KV-Edit' target='_blank'>Github Repo</a>. Thanks! 
    """

    badge = r"""
    [![GitHub Stars](https://img.shields.io/github/stars/Xilluill/KV-Edit)](https://github.com/Xilluill/KV-Edit)
    """
    
    with gr.Blocks() as demo:
        gr.HTML(title)
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column():
                source_prompt = gr.Textbox(label="Source Prompt", value='' )
                inversion_num_steps = gr.Slider(1, 50, 28, step=1, label="Number of inversion steps")
                target_prompt = gr.Textbox(label="Target Prompt", value='' )
                denoise_num_steps = gr.Slider(1, 50, 28, step=1, label="Number of denoise steps")
                brush_canvas = gr.ImageEditor(label="Brush Canvas",
                                                sources=('upload'), 
                                                brush=gr.Brush(colors=["#ff0000"],color_mode='fixed'),
                                                interactive=True,
                                                transforms=[],
                                                container=True,
                                                format='png',scale=1)
                
                inv_btn = gr.Button("inverse")
                edit_btn = gr.Button("edit")
                
                
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):

                    skip_step = gr.Slider(0, 30, 4, step=1, label="Number of skip steps")
                    inversion_guidance = gr.Slider(1.0, 10.0, 1.5, step=0.1, label="inversion Guidance", interactive=not is_schnell)
                    denoise_guidance = gr.Slider(1.0, 10.0, 5.5, step=0.1, label="denoise Guidance", interactive=not is_schnell)
                    attn_scale = gr.Slider(0.0, 5.0, 1, step=0.1, label="attn_scale")
                    seed = gr.Textbox('0', label="Seed (-1 for random)", visible=True)
                    with gr.Row():
                        re_init = gr.Checkbox(label="re_init", value=False)
                        attn_mask = gr.Checkbox(label="attn_mask", value=False)

                
                output_image = gr.Image(label="Generated Image")
                gr.Markdown(article)
        inv_btn.click(
            fn=editor.inverse,
            inputs=[brush_canvas,
                    source_prompt, target_prompt, 
                    inversion_num_steps, denoise_num_steps, 
                    skip_step, 
                    inversion_guidance,
                    denoise_guidance,seed,
                    re_init, attn_mask
                    ],
            outputs=[output_image]
        )
        edit_btn.click(
            fn=editor.edit,
            inputs=[brush_canvas,
                    source_prompt, target_prompt, 
                    inversion_num_steps, denoise_num_steps, 
                    skip_step, 
                    inversion_guidance,
                    denoise_guidance,seed,
                    re_init, attn_mask,attn_scale
                    ],
            outputs=[output_image]
        )
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--port", type=int, default=41032)
    args = parser.parse_args()

    # --- MODIFICATION ---
    # Update model paths based on parsed arguments before creating the editor
    update_model_paths(args.name, base_path="./models/KVEdit", flux_specific_path="./models/FLUX.1-dev")
    # --- END MODIFICATION ---

    demo = create_demo(args.name)
    
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)