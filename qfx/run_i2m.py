# -----------------------------------------------------------------------------
# Created by: Quentin Lengele (16/10/2025)
# -----------------------------------------------------------------------------

import sys
import os
import pathlib
import bpy
import cv2
import time
import json
import math
from mathutils import Matrix, Euler
import shutil
import warnings
import numpy as np
import torch
import pygltflib
from pygltflib.utils import ImageFormat
from PIL import Image, ImageFilter

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

import utils.imgops as ops
import utils.architecture.architecture as arch

warnings.filterwarnings('ignore')

# ====================================================================================================================

# Cap VRAM
total_vram = torch.cuda.get_device_properties(0).total_memory  # total VRAM in bytes
target_vram = 14 * 1024**3  # 14GB in bytes
fraction_vram = target_vram / total_vram
torch.cuda.set_per_process_memory_fraction(fraction_vram, device=0)

# ====================================================================================================================

NORMAL_MAP_MODEL = 'matgen/utils/models/1x_NormalMapGenerator-CX-Lite_200000_G.pth'
OTHER_MAP_MODEL = 'matgen/utils/models/1x_FrankenMapGenerator-CX-Lite_215000_G.pth'

DEVICE = torch.device('cuda')

MATGEN_MODELS = []

# ====================================================================================================================

def process_matgen(img, model):
    global DEVICE
    img = img * 1. / np.iinfo(img.dtype).max
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(DEVICE)

    output = model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    output = output[[2, 1, 0], :, :]
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.).round()
    return output


def load_matgen_model(model_path):
    global DEVICE
    state_dict = torch.load(model_path)
    model = arch.RRDB_Net(3, 3, 32, 12, gc=32, upscale=1, norm_type=None, act_type='leakyrelu',
                            mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model.to(DEVICE)


def fix_mesh(input_filepath, mesh_rotation, mesh_scale=1.0, smooth_normals=True, smooth_normals_angle=30.0):

    input_dir = os.path.dirname(input_filepath)

    glb_file_name = pathlib.Path(input_filepath).stem

    # Texture paths
    albedo_texture_filepath = os.path.join(input_dir, "textures", f"{glb_file_name}_basecolor.png")
    normal_texture_filepath = os.path.join(input_dir, "textures", f"{glb_file_name}_normal.png")
    roughness_texture_filepath = os.path.join(input_dir, "textures", f"{glb_file_name}_roughness.png")
    metallic_texture_filepath = os.path.join(input_dir, "textures", f"{glb_file_name}_metallic.png")

    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import GLB
    input_filepath = os.path.abspath(input_filepath)
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"GLB file not found: {input_filepath}")

    bpy.ops.import_scene.gltf(filepath=input_filepath)

    # Remove 'world' dummy parent if it exists
    for obj in bpy.data.objects:
        if obj.type == 'EMPTY' and obj.name.lower() == 'world':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Process all mesh objects
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.parent = None
            obj.name = glb_file_name
            obj.data.name = glb_file_name
            bpy.context.view_layer.objects.active = obj

            # Apply smoothing
            if smooth_normals:
                mesh = obj.data
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.shade_smooth()  # ← This actually smooths the faces visually
                mesh.use_auto_smooth = True
                mesh.auto_smooth_angle = smooth_normals_angle * (3.14159 / 180)
                mesh.update()

            # Apply scale
            obj.scale = (mesh_scale, mesh_scale, mesh_scale)
            bpy.context.view_layer.update()

            obj.select_set(True)

            # Apply rotation from mesh_rotation dict
            if mesh_rotation:                
                rot_x = math.radians(mesh_rotation.get('X', 0))
                rot_y = math.radians(mesh_rotation.get('Y', 0))
                rot_z = math.radians(mesh_rotation.get('Z', 0))

                rotation_matrix = Euler((rot_x, rot_y, rot_z), 'XYZ').to_matrix().to_4x4()

                # Apply rotation by multiplying the object's matrix
                obj.matrix_world = rotation_matrix @ obj.matrix_world

            # Update the scene to register the rotation
            bpy.context.view_layer.update()

            # Apply transformations
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Deselect object afterwards if needed
            obj.select_set(False)

            # Assign textures
            assign_textures(
                obj,
                albedo_path=albedo_texture_filepath,
                normal_path=normal_texture_filepath,
                roughness_path=roughness_texture_filepath,
                metallic_path=metallic_texture_filepath,
            )

    textures_folder = os.path.join(input_dir, "textures")
    redirect_textures_to_folder(textures_folder)

    # Resave GLB

    bpy.ops.export_scene.gltf(
        filepath=os.path.splitext(input_filepath)[0] + ".glb",
        export_format='GLB',
        export_texture_dir="textures"
    )

    # Export to FBX

    bpy.ops.export_scene.fbx(
        filepath=os.path.splitext(input_filepath)[0] + ".fbx",
        use_selection=False,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z',
        axis_up='Y',
        object_types={'MESH'},
        # mesh_smooth_type='FACE',
        mesh_smooth_type='EDGE',
        path_mode='COPY',
        embed_textures=False
    )

    # Remove unwanted .fbm folder if Blender created it
    fbm_folder = os.path.splitext(input_filepath)[0] + ".fbm"
    if os.path.exists(fbm_folder):
        shutil.rmtree(fbm_folder, ignore_errors=True)


def redirect_textures_to_folder(textures_folder):
    for image in bpy.data.images:
        if image and image.filepath:
            tex_name = os.path.basename(image.filepath)
            new_path = os.path.join(textures_folder, tex_name)
            image.filepath = bpy.path.abspath(new_path)
            image.reload()


def assign_textures(obj, albedo_path=None, normal_path=None, roughness_path=None, metallic_path=None):

    """Assign PBR textures to the object's material(s), supporting combined or separate metallic/roughness."""

    for mat_slot in obj.material_slots:
        mat = mat_slot.material
        if not mat or not mat.node_tree:
            continue

        mat.name = obj.name

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in nodes:
            if node.type in {'TEX_IMAGE', 'NORMAL_MAP', 'SEPARATE_RGB'}:
                nodes.remove(node)

        # Find Principled BSDF
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf:
            continue

        # --- Albedo ---
        if albedo_path and os.path.exists(albedo_path):
            img_node = nodes.new('ShaderNodeTexImage')
            img_node.image = bpy.data.images.load(albedo_path)
            img_node.image.colorspace_settings.is_data = False  # sRGB
            img_node.location = (-600, 300)
            links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])

        # --- Normal Map ---
        if normal_path and os.path.exists(normal_path):
            normal_img_node = nodes.new('ShaderNodeTexImage')
            normal_img_node.image = bpy.data.images.load(normal_path)
            normal_img_node.image.colorspace_settings.is_data = True  # Non-color
            normal_img_node.location = (-600, 100)

            normal_map_node = nodes.new('ShaderNodeNormalMap')
            normal_map_node.location = (-400, 100)

            links.new(normal_img_node.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])

        # --- Roughness ---
        if roughness_path and os.path.exists(roughness_path):
            rough_node = nodes.new('ShaderNodeTexImage')
            rough_node.image = bpy.data.images.load(roughness_path)
            rough_node.image.colorspace_settings.is_data = True  # Non-color
            rough_node.location = (-600, -100)
            links.new(rough_node.outputs['Color'], bsdf.inputs['Roughness'])

        # --- Metallic ---
        if metallic_path and os.path.exists(metallic_path):
            metal_node = nodes.new('ShaderNodeTexImage')
            metal_node.image = bpy.data.images.load(metallic_path)
            metal_node.image.colorspace_settings.is_data = True  # Non-color
            metal_node.location = (-600, -300)
            links.new(metal_node.outputs['Color'], bsdf.inputs['Metallic'])


def generate_pbr(albedo_filepath,
                 override=False,
                 tile_size=512,
                 seamless=False,
                 mirror=False,
                 replicate=False):

    global MATGEN_MODELS

    basename = os.path.splitext(os.path.basename(albedo_filepath))[0]
    basename = basename.replace("_BaseColor", "")

    output_folder = os.path.dirname(albedo_filepath)

    # read image
    try:
        img = cv2.imread(albedo_filepath, cv2.cv2.IMREAD_COLOR)
    except:
        img = cv2.imread(albedo_filepath, cv2.IMREAD_COLOR)

    # Seamless modes
    if seamless:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
    elif mirror:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
    elif replicate:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)

    img_height, img_width = img.shape[:2]

    # Whether to perform the split/merge action
    do_split = img_height > tile_size or img_width > tile_size

    if do_split:
        rlts = ops.esrgan_launcher_split_merge(img, process_matgen, MATGEN_MODELS, scale_factor=1, tile_size=tile_size)
    else:
        rlts = [process_matgen(img, model) for model in MATGEN_MODELS]

    if seamless or mirror or replicate:
        rlts = [ops.crop_seamless(rlt) for rlt in rlts]

    normal_map = rlts[0]
    roughness = rlts[1][:, :, 1]
    displacement = rlts[1][:, :, 0]

    normal_name = '{:s}_Normal.png'.format(basename)
    cv2.imwrite(os.path.join(output_folder, normal_name), normal_map)

    rough_name = '{:s}_Roughness.png'.format(basename)
    rough_img = roughness
    cv2.imwrite(os.path.join(output_folder, rough_name), rough_img)

    displace_name = '{:s}_Displacement.png'.format(basename)
    cv2.imwrite(os.path.join(output_folder, displace_name), displacement)

    metallic_file = albedo_filepath.replace("_BaseColor", "_Metallic")
    generate_metallic_map(albedo_filepath, metallic_file)

    displace_file = albedo_filepath.replace("_BaseColor", "_Displacement")
    ao_file = albedo_filepath.replace("_BaseColor", "_AO")
    generate_ao_map(displace_file, ao_file)

    roughness_file = albedo_filepath.replace("_BaseColor", "_Roughness")
    orm_file = albedo_filepath.replace("_BaseColor", "_ORM")

    generate_orm_map(ao_file, roughness_file, metallic_file, orm_file)

    emissive_file = albedo_filepath.replace("_BaseColor", "_Emissive")

    generate_emissive_map(albedo_filepath, emissive_file)
    

def generate_metallic_map(albedo_path: str,
                          output_path: str = "metallic.png",
                          grayness_thresh: float = 0.08,
                          saturation_thresh: float = 0.25,
                          smooth: bool = True) -> Image.Image:
    """
    Generate a metallic map from an albedo texture using color heuristics.

    Parameters:
        albedo_path (str): Path to the albedo (base color) image.
        output_path (str): Path to save the resulting metallic map.
        grayness_thresh (float): Max color std deviation for 'gray' detection (0–1).
        saturation_thresh (float): Max saturation value for metal classification (0–1).
        smooth (bool): If True, apply Gaussian-like blur for smoother mask.

    Returns:
        PIL.Image.Image: The generated metallic map as a grayscale image.
    """
    # Load image
    albedo = Image.open(albedo_path).convert("RGB")
    arr = np.array(albedo, dtype=np.float32) / 255.0

    # Compute color statistics
    mean_rgb = np.mean(arr, axis=2)
    std_rgb = np.std(arr, axis=2)
    max_rgb = np.max(arr, axis=2)
    min_rgb = np.min(arr, axis=2)

    # Compute saturation (approx from RGB)
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    grayness = std_rgb  # measure of how neutral the color is

    # Core metallic heuristic
    metallic_mask = np.where(
        (grayness < grayness_thresh) & (saturation < saturation_thresh) & (mean_rgb > 0.1),
        1.0,
        0.0
    )

    # Optional soft blending
    if smooth:
        from scipy.ndimage import gaussian_filter
        metallic_mask = gaussian_filter(metallic_mask, sigma=1.2)
        metallic_mask = np.clip(metallic_mask, 0, 1)

    # Save output
    metallic_img = Image.fromarray((metallic_mask * 255).astype(np.uint8))
    metallic_img.save(output_path)
    return metallic_img


def generate_ao_map(displacement_path: str, output_path: str = None, blur_radius: int = 2):
    """
    Generates a rough Ambient Occlusion (AO) map from a displacement (height) map.

    Args:
        displacement_path (str): Path to the displacement (height) map PNG.
        output_path (str, optional): Path to save the AO map.
                                     If None, saves alongside input with '_AO' suffix.
        blur_radius (int): Radius for Gaussian blur to simulate soft occlusion.

    Returns:
        str: Path of the saved AO map.
    """
    # Load displacement map as grayscale
    disp_img = Image.open(displacement_path).convert("L")
    disp_arr = np.array(disp_img, dtype=np.float32) / 255.0  # normalize 0-1

    # Invert height: high areas = less occluded
    inv_height = 1.0 - disp_arr

    # Convert back to image
    inv_height_img = Image.fromarray((inv_height * 255).astype(np.uint8))

    # Apply Gaussian blur to simulate ambient occlusion
    ao_img = inv_height_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(displacement_path)
        output_path = f"{base}_AO.png"

    # Save AO map
    ao_img.save(output_path)
    print(f"AO map saved to: {output_path}")
    return output_path


def generate_orm_map(ao_path=None, roughness_path=None, metallic_path=None, output_path=None):
    """
    Combines AO, Roughness, and Metallic textures into a single ORM map (R=AO, G=Roughness, B=Metallic).

    Args:
        ao_path (str): Path to AO texture (grayscale)
        roughness_path (str): Path to Roughness texture (grayscale)
        metallic_path (str): Path to Metallic texture (grayscale)
        output_path (str): Path to save the ORM texture. If None, saves as 'ORM.png' in the first input folder.

    Returns:
        str: Path of the saved ORM map.
    """

    # Load textures or create default gray if missing
    def load_gray(path, size=None):
        if path and os.path.exists(path):
            img = Image.open(path).convert("L")
        else:
            img = Image.new("L", size if size else (1024, 1024), 255)
        return img

    # Determine base size from first available texture
    base_size = None
    for path in [ao_path, roughness_path, metallic_path]:
        if path and os.path.exists(path):
            base_size = Image.open(path).size
            break
    if base_size is None:
        base_size = (1024, 1024)  # default

    ao_img = load_gray(ao_path, base_size)
    rough_img = load_gray(roughness_path, base_size)
    metal_img = load_gray(metallic_path, base_size)

    # Merge into RGB
    orm_img = Image.merge("RGB", (ao_img, rough_img, metal_img))

    # Determine output path
    if not output_path:
        first_path = next((p for p in [ao_path, roughness_path, metallic_path] if p), None)
        folder = os.path.dirname(first_path) if first_path else os.getcwd()
        output_path = os.path.join(folder, "ORM.png")

    orm_img.save(output_path)
    print(f"ORM map saved to: {output_path}")
    return output_path


def generate_emissive_map(input_path, output_path,
                          sat_threshold=0.4,
                          val_threshold=0.6,
                          white_exclude_threshold=0.25,
                          blur_radius=2,
                          excluded_hue_ranges=None):
    """
    Generate an emissive texture from an albedo image by detecting bright, colorful regions.
    Excludes white and light-blue (cyan/sky tones) by default.
    """
    if excluded_hue_ranges is None:
        excluded_hue_ranges = []
        
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img) / 255.0
    r, g, b = img_np[..., 0], img_np[..., 1], img_np[..., 2]

    cmax = np.max(img_np, axis=-1)
    cmin = np.min(img_np, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    mask = delta != 0
    hue[mask & (cmax == r)] = (60 * ((g - b) / delta) % 360)[mask & (cmax == r)]
    hue[mask & (cmax == g)] = (60 * ((b - r) / delta) + 120)[mask & (cmax == g)]
    hue[mask & (cmax == b)] = (60 * ((r - g) / delta) + 240)[mask & (cmax == b)]

    sat = np.zeros_like(cmax)
    sat[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]
    val = cmax

    # Basic emissive mask based on saturation and brightness
    emissive_mask = (sat > sat_threshold) & (val > val_threshold) & (sat > white_exclude_threshold)

    # Exclude unwanted hue ranges (like light blues)
    for low, high in excluded_hue_ranges:
        emissive_mask &= ~((hue >= low) & (hue <= high))

    # Apply mask
    emissive_img = np.zeros_like(img_np)
    emissive_img[emissive_mask] = img_np[emissive_mask]

    emissive_img = (emissive_img * 255).astype(np.uint8)

    emissive_img = Image.fromarray(emissive_img)
    emissive_img = emissive_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    emissive_img.save(output_path)
    print(f"Emissive texture saved to: {output_path}")
    

def extract_albedo_texture(input_filepath):

    texture_filename = pathlib.Path(input_filepath).stem
    input_folder = os.path.dirname(input_filepath)

    texture_folder = f"{input_folder}/textures"

    if not os.path.exists(texture_folder):
        os.mkdir(texture_folder)

    dest_file = f'{texture_folder}\\{texture_filename}_BaseColor.png'

    # export texture

    gltf = pygltflib.GLTF2()
    gltf.draco_compression = False  # skip Draco

    glb = gltf.load(input_filepath)
    if len(glb.images) > 0:

        glb.convert_images(ImageFormat.FILE, texture_folder, override=True)

        for i in range(0, len(glb.images)):

            if os.path.exists(dest_file):
                os.remove(dest_file)

            os.rename(f'{texture_folder}\\{glb.images[i].uri}', dest_file)


def cleanup_torch_lock(lock_path, wait_time=0.5, retries=3):
    
    """Safely removes a stale PyTorch extension build lock if it exists."""
    if not os.path.exists(lock_path):
        return  # No lock, nothing to do

    print(f"[TorchExtensions] Found lock file at: {lock_path}")

    for i in range(retries):
        try:
            os.remove(lock_path)
            print("[TorchExtensions] Stale lock removed successfully")
            return
        except PermissionError:
            print(f"[TorchExtensions] Lock file in use, waiting... ({i+1}/{retries})")
            time.sleep(wait_time)

    # If still locked after retries
    print("[TorchExtensions] Warning: lock file could not be removed. It may still be in use.")


def generate(input_filelist):

    global MATGEN_MODELS

    # Load JSON
    with open(input_filelist, "r", encoding="utf-8") as f:
        data = json.load(f)

    inputs = data["inputs"]
    parameters = data["parameters"]

    output_dir = parameters["output_folder"]
    seed = parameters["seed"]
    simplify_ratio = parameters["simplify_ratio"]
    texture_size = parameters["texture_size"]
    sparse_structure_steps = parameters["sparse_sampling_steps"]
    sparse_structure_cfg = parameters["sparse_sampling_cfg"]
    slat_steps = parameters["slat_sampling_steps"]
    slat_cfg = parameters["slat_sampling_cfg"]
    mesh_scale = parameters["mesh_scale"]
    mesh_rotation = parameters["mesh_rotation"]
    smooth_normals = parameters["smooth_normals"]
    smooth_normals_angle = parameters["smooth_normals_angle"]
    # model_type = parameters["model_type"]
    override = parameters["override"]

    # Path to your torch extension lock file
    lock_path = os.path.expandvars(
        r"%LOCALAPPDATA%\torch_extensions\torch_extensions\Cache\py310_cu128\nvdiffrast_plugin\lock"
    )

    # Run cleanup before any nvdiffrast or torch extension import
    cleanup_torch_lock(lock_path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load MatGen Models

    MATGEN_MODELS = [
        load_matgen_model(NORMAL_MAP_MODEL), # NORMAL MAP
        load_matgen_model(OTHER_MAP_MODEL)   # ROUGHNESS/DISPLACEMENT MAPS
    ]

    # Load a pipeline from a model folder
    pipeline = TrellisImageTo3DPipeline.from_pretrained("models/TRELLIS-image-large")
    pipeline.cuda()    

    # ---

    count = 0
    total = len(inputs)

    for file_input in inputs:

        # Empty PyTorch’s CUDA cache
        torch.cuda.empty_cache()

        input_name = file_input["name"]
        input_file_path = file_input["filepath"]

        print(f"TrellisProgress: {count}/{total}")

        output_folder = f"{output_dir}/{input_name}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        output_file_path = f"{output_folder}/{input_name}.glb"

        if os.path.exists(output_file_path) and not override:
            continue

        # Load an image
        image = Image.open(input_file_path)

        # Run the pipeline
        outputs = pipeline.run(
            image,
            seed=seed,

            # Optional parameters
            sparse_structure_sampler_params={
                "steps": sparse_structure_steps,
                "cfg_strength": sparse_structure_cfg,
            },
            slat_sampler_params={
                "steps": slat_steps,
                "cfg_strength": slat_cfg,
            },
            formats=['mesh', 'gaussian']
        )

        # GLB files can be extracted from the outputs
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=simplify_ratio,  # Ratio of triangles to remove in the simplification process
            texture_size=texture_size,  # Size of the texture used for the GLB
        )
        glb.export(output_file_path)

        print("Generating PBR Textures: 100%")

        extract_albedo_texture(output_file_path)

        # Generate PBR from Albedo
        albedo_filepath = f"{output_folder}/textures/{input_name}_BaseColor.png"
        generate_pbr(albedo_filepath, override)

        print("Finalizing Mesh: 100%")

        fix_mesh(output_file_path, mesh_rotation, mesh_scale, smooth_normals, smooth_normals_angle)

        # Optionally, synchronize to make sure the GPU finishes pending work
        torch.cuda.synchronize()

        count += 1


if __name__ == "__main__":

    if len(sys.argv) < 1:
        sys.exit(1)

    generate(sys.argv[-1])

