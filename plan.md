**Project Goal:** Modify the existing InfiniteYou-FLUX system to perform face swapping. The identity will be taken from an input image (referred to internally as `id_image`), and this identity will be inpainted onto the largest face detected in a second input image (referred to internally as `control_image`), preserving the second image's background, context, and the pose of the detected face.

**Core Constraint:** No existing variable names or function names can be changed. New functions and parameters can be added. Existing parameters will be repurposed conceptually.

---

**Phase 1: Modify `pipelines/pipeline_infu_flux.py` (High-Level Pipeline)**

1.  **`__init__(...)`:**
    *   **Action:** No changes required. The existing initialization correctly loads `FaceAnalysis` models (`app_640`, `app_320`, `app_160`) and the `arcface_model`, which are needed for both source and target face processing.

2.  **`_detect_face(...)`:**
    *   **Action:** No changes required. This function will be used to detect faces in both the image passed as the `id_image` parameter (Source) and the image passed as the `control_image` parameter (Target).

3.  **Add New Helper Function: `_prepare_mask_and_control(...)`:**
    *   **Action:** Define a *new* private method within the `InfUFluxPipeline` class.
    *   **Signature:**
        ```python
        def _prepare_mask_and_control(self, target_image_pil, target_face_info, target_size, blur_kernel_size=9, expand_ratio=0.3):
        ```
    *   **Inputs:**
        *   `target_image_pil`: The PIL image passed as the `control_image` parameter to `__call__`.
        *   `target_face_info`: The dictionary (`{'bbox': ..., 'kps': ...}`) for the largest face detected in `target_image_pil`.
        *   `target_size`: Tuple `(width, height)` of the desired output/processing dimensions.
        *   `blur_kernel_size`: Integer controlling the Gaussian blur strength for the mask edges.
        *   `expand_ratio`: Float controlling how much the mask expands beyond the detected bounding box.
    *   **Implementation:**
        1.  Extract `bbox = target_face_info['bbox']` and `kps = target_face_info['kps']`.
        2.  Create a NumPy array `mask = np.zeros((target_size[1], target_size[0]), dtype=np.float32)`.
        3.  Calculate expanded bounding box coordinates (`new_x1`, `new_y1`, `new_x2`, `new_y2`) based on `bbox`, `expand_ratio` (using `expand_ratio + 0.1` for vertical expansion), and clamping coordinates within `0` and `target_size[0/1] - 1`.
        4.  Fill the expanded rectangle in the `mask` array with `1.0`.
        5.  Ensure `blur_kernel_size` is odd (`if blur_kernel_size % 2 == 0: blur_kernel_size += 1`) and at least 1.
        6.  Apply Gaussian blur: `mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), 0)`.
        7.  Convert the NumPy mask to a PyTorch tensor: `mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)`.
        8.  Create a black background image: `generated_control_hint_image = Image.new('RGB', target_size, (0, 0, 0))`.
        9.  Use the existing `draw_kps` function to draw the target keypoints (`kps`) onto the black background: `generated_control_hint_image = draw_kps(generated_control_hint_image, kps)`.
    *   **Return:** `mask_tensor`, `generated_control_hint_image`

4.  **Modify `__call__(...)` Method:**
    *   **Action:** Keep the existing signature but add new optional keyword arguments for mask control. Adapt the internal logic to handle the conceptual shift of parameters.
    *   **Revised Signature (Conceptual):**
        ```python
        def __call__(
            self,
            id_image: Image.Image,  # CONCEPT: Source Identity Image
            prompt: str,
            control_image: Optional[Image.Image] = None,  # CONCEPT: Target Context/Pose Image (Now Required for Inpainting)
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
            # --- Existing ---
            # **kwargs # Keep if originally present
        ):
        ```
    *   **Implementation Changes:**
        1.  **Parameter Interpretation:** Clearly understand that `id_image` provides the source identity and `control_image` provides the target context/pose.
        2.  **Require `control_image`:** Add a check: `if control_image is None: raise ValueError("Target image (passed as 'control_image') is required for face swapping/inpainting.")`
        3.  **Process `id_image` (Source):**
            *   Convert `id_image` to `id_image_cv2` (BGR NumPy).
            *   Call `face_info = self._detect_face(id_image_cv2)`. Handle `len(face_info) == 0` error ("No face detected in source image (id_image)").
            *   Select largest face: `source_face_info = sorted(face_info, ...)[-1]`.
            *   Extract `landmark = source_face_info['kps']`.
            *   Call `id_embed = extract_arcface_bgr_embedding(id_image_cv2, landmark, self.arcface_model)`.
            *   Reshape and project `id_embed` using `self.image_proj_model` (existing logic remains). The variable `id_embed` now holds the *source* identity embedding.
        4.  **Process `control_image` (Target):**
            *   Ensure `control_image` is RGB PIL.
            *   Convert `control_image` to `target_image_cv2` (BGR NumPy).
            *   Call `face_info = self._detect_face(target_image_cv2)`. Handle `len(face_info) == 0` error ("No face detected in target image (control_image)").
            *   Select largest face: `target_face_info = sorted(face_info, ...)[-1]`.
            *   Call the new helper: `mask_tensor, generated_control_hint_image = self._prepare_mask_and_control(control_image, target_face_info, target_size=(width, height), blur_kernel_size=blur_kernel_size, expand_ratio=mask_expand_ratio)`.
            *   Resize/Pad original `control_image` to `(width, height)`: `target_image_processed = control_image.resize((width, height), Image.LANCZOS)` (or use `resize_and_pad_image`).
            *   Preprocess for VAE: `target_image_tensor = self.pipe.image_processor.preprocess(target_image_processed).to(device=self.pipe.device, dtype=self.pipe.vae.dtype)`.
            *   VAE Encode `control_image`:
                ```python
                with torch.no_grad():
                    target_image_latents = self.pipe.vae.encode(target_image_tensor).latent_dist.sample()
                    # Important: Keep latents UNPACKED here. Packing happens in Phase 2 if needed.
                    target_image_latents = (target_image_latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
                ```
        5.  **Call Underlying Pipeline (`self.pipe`):** Pass the correctly prepared arguments.
            ```python
            # Ensure generator is created on the correct device for add_noise compatibility
            generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

            # Call the modified FluxInfuseNetPipeline
            image = self.pipe(
                prompt=prompt,
                controlnet_prompt_embeds=id_embed,          # Source ID embedding
                control_image=generated_control_hint_image, # Target kps drawn on black
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                controlnet_guidance_scale=1.0, # Keep original name/usage if needed by InfuseNet
                controlnet_conditioning_scale=infusenet_conditioning_scale, # Applies to InfuseNet branch
                control_guidance_start=infusenet_guidance_start,
                control_guidance_end=infusenet_guidance_end,
                generator=generator,
                # --- NEW Args Passed ---
                target_image_latents=target_image_latents, # Unpacked initial target latents
                inpainting_mask=mask_tensor.to(device=self.pipe.device, dtype=torch.bfloat16), # Mask
                # Pass other relevant args like negative_prompt if applicable
            ).images[0]
            ```
        6.  **Return:** Return the final `image`.

---

**Phase 2: Modify `pipelines/pipeline_flux_infusenet.py` (Low-Level Pipeline with Denoising Loop)**

1.  **Modify `__call__(...)` Signature:**
    *   **Action:** Add new optional parameters to accept the target latents and mask.
    *   **Additions to Signature:**
        ```python
        # ... existing parameters ...,
        target_image_latents: Optional[torch.FloatTensor] = None,
        inpainting_mask: Optional[torch.FloatTensor] = None
        # Ensure `generator` is properly handled (passed in or created)
        ```

2.  **Prepare for Inpainting (Before Denoising Loop):**
    *   **Action:** Inside `__call__`, after standard setup (device, dtype, prompt encoding) but before the `for` loop over timesteps.
    *   **Implementation:**
        ```python
        target_image_latents_packed = None
        packed_latent_mask = None
        is_inpainting_mode = target_image_latents is not None and inpainting_mask is not None

        if is_inpainting_mode:
            # Pack the initial target latents (assuming they arrive unpacked)
            # Use the same logic as self.prepare_latents uses internally or self._pack_latents
            # Example structure (adapt based on actual _pack_latents implementation):
            packed_h = height // self.vae_scale_factor // 2
            packed_w = width // self.vae_scale_factor // 2
            num_channels_latents = self.transformer.config.in_channels // 4 # Should be 16 for FLUX
            target_image_latents_packed = target_image_latents.view(
                target_image_latents.shape[0], num_channels_latents // 4, 4, packed_h, packed_w
            )
            target_image_latents_packed = target_image_latents_packed.permute(0, 1, 3, 4, 2).reshape(
                target_image_latents.shape[0], num_channels_latents // 4, packed_h, packed_w * 4
            )
            target_image_latents_packed = target_image_latents_packed.permute(0, 2, 1, 3).reshape(
                target_image_latents.shape[0], packed_h, num_channels_latents // 4 * packed_w * 4
            )
            target_image_latents_packed = target_image_latents_packed.permute(0, 2, 1).contiguous() # Shape [B, C_packed, H_packed] -> Need [B, L, C_packed]? Check prepare_latents format

            # --> Revisit Packing: Flux packing might be different. Assume we get the correct shape for target_image_latents_packed matching `latents`.
            # If _pack_latents exists and is callable:
            # target_image_latents_packed = self._pack_latents(target_image_latents, batch_size * num_images_per_prompt, num_channels_latents, ...)
            # *** Safest: Ensure target_image_latents is encoded and *then packed* exactly like initial random `latents` are prepared ***

            # Resize inpainting mask to the *packed* latent dimensions
            # Get packed shape from the initial `latents` variable prepared by `self.prepare_latents`
            packed_latent_shape = latents.shape[2:] # Assuming shape is [B, L, C] or [B, C, H, W] -> need correct H, W
            # If latents are [B, L, C], mask needs interpolation across L? Unlikely. Flux uses 2D latents internally?
            # Assume latents are effectively [B, C_packed, H_packed, W_packed] before potential flattening.
            # Get H_packed, W_packed correctly based on input height/width and model specifics.
            packed_h = height // self.vae_scale_factor // 2 # Adjust based on Flux architecture if needed
            packed_w = width // self.vae_scale_factor // 2  # Adjust based on Flux architecture if needed
            packed_latent_mask = torch.nn.functional.interpolate(
                inpainting_mask, size=(packed_h, packed_w), mode='bilinear', align_corners=False
            )
            # Ensure mask has the right shape to broadcast with packed latents (e.g., [B, 1, H_packed, W_packed])
            packed_latent_mask = packed_latent_mask.to(device=device, dtype=dtype)
        ```
    *   **Crucial Note:** The exact packing logic (`_pack_latents`) and the effective 2D shape `(packed_h, packed_w)` for masking in the latent space *must* match the FLUX architecture precisely. This might require inspecting `self.prepare_latents` and the `transformer` forward pass.

3.  **Modify Denoising Loop:**
    *   **Action:** Inside the `for i, t in enumerate(timesteps):` loop, modify how the next `latents` state is computed if inpainting.
    *   **Location:** After calculating `noise_pred` (and potentially applying CFG/True CFG) but *before* the final `latents = self.scheduler.step(...)` assignment.
    *   **Implementation:**
        ```python
        # ... inside loop, after noise_pred is calculated ...

        # Calculate the predicted previous sample (x_{t-1}) using the current state (x_t = latents)
        # Ensure generator is passed if needed by scheduler step implementation
        latents_next_step = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

        # --- Inpainting Logic ---
        if is_inpainting_mode:
            # 1. Get the original target latents (packed) noised to the current timestep 't'
            noise_t = torch.randn_like(latents, device=device, generator=generator) # Generate noise matching current latents shape
            # Ensure target_image_latents_packed has same shape/device as latents for add_noise
            noisy_target_latents_packed_t = self.scheduler.add_noise(target_image_latents_packed.to(device), noise_t, t)

            # 2. Combine: Use predicted step inside mask, noised original outside mask
            # Need to reshape mask if latents are not [B, C, H, W] format
            # Assuming packed_latent_mask is [B, 1, H_packed, W_packed] and latents_next_step/noisy_target are compatible
            latents = latents_next_step * packed_latent_mask + noisy_target_latents_packed_t * (1.0 - packed_latent_mask)

        else:
            # --- Original Logic ---
            latents = latents_next_step # Standard denoising step

        # ... rest of the loop (callbacks, progress bar update) ...
        ```
    *   **Key Points:**
        *   The `generator` instance must be passed correctly to `scheduler.step` and `torch.randn_like` for reproducibility.
        *   `target_image_latents_packed` must be on the same `device` as `latents`.
        *   The shapes of `latents_next_step`, `noisy_target_latents_packed_t`, and `packed_latent_mask` must be compatible for element-wise multiplication. This depends heavily on the internal representation of `latents` in FLUX (e.g., `[B, L, C]` or `[B, C, H, W]`). The mask interpolation and broadcasting need to match this.

---

**Phase 3: Update `app.py` (Gradio UI)**

1.  **Update Input Components:**
    *   `ui_id_image`: Change `gr.Image(..., label="Identity Image", ...)` to `gr.Image(..., label="Source Identity Image", ...)`. **Keep `variable=ui_id_image`**.
    *   `ui_control_image`: Change `gr.Image(..., label="Control Image [Optional]", ...)` to `gr.Image(..., label="Target Image (Context/Pose)", ...)`. **Keep `variable=ui_control_image`**. Consider making it non-optional in the UI logic if `None` isn't handled gracefully by the backend now.

2.  **Add New Input Components:**
    *   Add `ui_blur_kernel_size = gr.Slider(minimum=1, maximum=51, step=2, value=9, label="Mask Blur Kernel Size")` (or `gr.Number`). Place within an Accordion or alongside other parameters.
    *   Add `ui_mask_expand_ratio = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3, label="Mask Expand Ratio")`. Place nearby.

3.  **Update `generate_image` Function:**
    *   **Action:** Modify the function called by the Generate button to accept and pass the new parameters.
    *   **Signature Change:** Add `blur_kernel_size`, `mask_expand_ratio` to the function definition:
        ```python
        def generate_image(
            # Keep existing parameters:
            input_image,  # Corresponds to ui_id_image
            control_image, # Corresponds to ui_control_image
            prompt, seed, width, height, guidance_scale, num_steps,
            infusenet_conditioning_scale, infusenet_guidance_start, infusenet_guidance_end,
            enable_realism, enable_anti_blur, model_version,
            # Add new parameters:
            blur_kernel_size,
            mask_expand_ratio
        ):
            # ... inside function ...
            # Prepare pipeline call
            try:
                image = pipeline( # Assuming 'pipeline' is the InfUFluxPipeline instance
                    id_image=input_image,       # Pass source
                    prompt=prompt,
                    control_image=control_image, # Pass target
                    # ... pass all other existing params like seed, width etc...
                    blur_kernel_size=blur_kernel_size, # Pass new param
                    mask_expand_ratio=mask_expand_ratio, # Pass new param
                )
            # ... error handling ...
            return gr.update(...)
        ```

4.  **Update `ui_btn_generate.click` Call:**
    *   **Action:** Add the new UI components to the `inputs` list for the button's click event.
    *   **Modification:**
        ```python
        ui_btn_generate.click(
            generate_image,
            inputs=[
                ui_id_image,
                ui_control_image,
                ui_prompt_text,
                ui_seed,
                ui_width,
                ui_height,
                ui_guidance_scale,
                ui_num_steps,
                ui_infusenet_conditioning_scale,
                ui_infusenet_guidance_start,
                ui_infusenet_guidance_end,
                ui_enable_realism,
                ui_enable_anti_blur,
                ui_model_version,
                # --- Add New Inputs ---
                ui_blur_kernel_size,
                ui_mask_expand_ratio,
            ],
            outputs=[image_output],
            concurrency_id="gpu"
        )
        ```

5.  **Update `gr.Examples`:**
    *   **Action:** Modify the `sample_list` and the example loading logic to accommodate two input images and potentially the new parameters.
    *   **`sample_list` Structure:** Each entry needs to represent the inputs needed by `generate_examples`. It should now conceptually be `[source_img_path, target_img_path, prompt, seed, realism, anti_blur, model_version]`. Update existing examples, e.g.:
        ```python
        # Old: ['./assets/examples/man.jpg', None, 'Prompt...', 666, False, False, 'aes_stage2']
        # New: ['./assets/examples/man.jpg', './assets/examples/man_pose.jpg', 'Prompt...', 666, False, False, 'aes_stage2'] # Provide a target image path
        ```
    *   **`gr.Examples` Inputs:** The `inputs` list must match the order/components expected by `generate_examples`.
        ```python
        gr.Examples(
            sample_list,
            inputs=[ # Match generate_examples signature
                ui_id_image,        # Represents source
                ui_control_image,   # Represents target
                ui_prompt_text,
                ui_seed,
                ui_enable_realism,
                ui_enable_anti_blur,
                ui_model_version,
                # Potentially add ui_blur_kernel_size, ui_mask_expand_ratio if examples should set them
            ],
            outputs=[image_output],
            fn=generate_examples, # Ensure this function handles the inputs
            cache_examples=True,
        )
        ```
    *   **Update `generate_examples` function:**
        *   Modify its signature to accept parameters matching the `inputs` list above (e.g., `def generate_examples(id_image_path, control_image_path, ...)`).
        *   Inside `generate_examples`, load the images from paths.
        *   Call the main `generate_image` function (or the pipeline directly), passing the loaded images and other parameters. Provide default values for `blur_kernel_size` and `mask_expand_ratio` if they aren't part of the example inputs.

6.  **Update UI Text:**
    *   Modify Markdown/HTML descriptions to refer to "Source Identity Image" and "Target Image (Context/Pose)". Explain the purpose of the new "Mask Blur Kernel Size" and "Mask Expand Ratio" sliders.

---

**Phase 4: Update `test.py` (Command-Line Script)**

1.  **Update `argparse.ArgumentParser`:**
    *   Modify `--id_image`: Update `help="Input source identity image"`. **Keep `dest='id_image'`**.
    *   Modify `--control_image`: Update `help="Input target context/pose image"`. Remove `default=None`. Add `required=True` if applicable. **Keep `dest='control_image'`**.
    *   Add `--blur_kernel_size`: `parser.add_argument('--blur_kernel_size', type=int, default=9, help="Gaussian blur kernel size for inpainting mask edges (odd number)")`.
    *   Add `--mask_expand_ratio`: `parser.add_argument('--mask_expand_ratio', type=float, default=0.3, help="Expansion ratio for inpainting mask beyond detected face bbox")`.

2.  **Update `pipe(...)` Call in `main()`:**
    *   **Action:** Pass the new arguments to the pipeline call using the keyword arguments defined in Phase 1.
    *   **Implementation:**
        ```python
        # ... inside main() after loading pipe and LoRAs ...
        image = pipe(
            id_image=Image.open(args.id_image).convert('RGB'), # Use existing arg name
            prompt=args.prompt,
            control_image=Image.open(args.control_image).convert('RGB'), # Use existing arg name
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            infusenet_conditioning_scale=args.infusenet_conditioning_scale,
            infusenet_guidance_start=args.infusenet_guidance_start,
            infusenet_guidance_end=args.infusenet_guidance_end,
            # --- Pass New Args ---
            blur_kernel_size=args.blur_kernel_size,
            mask_expand_ratio=args.mask_expand_ratio,
        )
        ```

3.  **Update Output Filename Generation:**
    *   **Action:** Modify the `out_name` f-string to include identifiers from both source (`args.id_image`) and target (`args.control_image`) images for clarity.
    *   **Example:**
        ```python
        id_name = os.path.splitext(os.path.basename(args.id_image))[0]
        target_name = os.path.splitext(os.path.basename(args.control_image))[0] # Get target name
        prompt_name = args.prompt[:100].replace('/', '|') + ('*' if len(args.prompt) > 100 else '') # Shorter prompt
        out_name = f'{index:05d}_src-{id_name}_tgt-{target_name}_{prompt_name}_seed{args.seed}.png' # Include both
        ```

---

**Phase 5: Testing and Refinement**

1.  **Execute `test.py`:** Run with various source/target image pairs and different mask parameters (`--blur_kernel_size`, `--mask_expand_ratio`). Verify output images visually for:
    *   Correct identity transfer.
    *   Correct pose matching the target face.
    *   Seamless blending at mask edges.
    *   Preservation of background/context from the target image.
2.  **Use Gradio App (`app.py`):** Test interactively. Ensure UI elements function correctly and parameter changes have the expected effect. Test example loading.
3.  **Debug Denoising Loop:** If inpainting fails or produces artifacts, insert debugging prints or use a debugger within the `pipeline_flux_infusenet.py` loop. Pay close attention to:
    *   Shapes and dtypes of `latents`, `target_image_latents_packed`, `packed_latent_mask`, `latents_next_step`, `noisy_target_latents_packed_t`.
    *   Device placement of all tensors involved in the combination step.
    *   Correctness of the `scheduler.add_noise` result.
    *   Correctness of the mask application (multiplication and addition).
4.  **Refine Defaults:** Adjust default values for `blur_kernel_size` and `mask_expand_ratio` in `pipeline_infu_flux.py`, `app.py`, and `test.py` based on testing results for optimal default performance.

---
This comprehensive plan outlines the necessary modifications across all relevant files to implement the face swapping feature with inpainting, while rigorously adhering to the constraint of not renaming any existing variables or functions.