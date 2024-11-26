# Generating New Camera View

### Environment
* scenes, utils: SparseGS folder customized
* torch, CUDA setting is aligned with MVSplat setting:
```bash
git clone https://github.com/Three-Shots-Are-Enough/Gen-new-cam
cd Gen-new-cam
conda create -n mvsplat python=3.10
conda activate mvsplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
conda install -c conda-forge libgl
```


### Preparation
- The `scenes_sparse` folder is the one uploaded to Google Drive, but it doesn't contain an `images` folder, so it needs to be created manually.  
- The working script is `generate_cams_main.py`, and initially, the `scene_path` and `output_path` must be properly adjusted.

## Overall Pipeline
```
python generate_cams_main.py --cam1_id 8686 --cam2_id 8867 --num_interpolations 88
```

#### Workflow Overview
1. Read `cameras.txt` and `images.txt` from the specified `scene_sparse` path.
2. Extract intrinsic and extrinsic parameters from the `.txt` files to create camera info and instantiate Camera objects.  
3. Construct the `src_cams` dictionary in the format `{colmap_id1: Cam1, colmap_id2: Cam2}`.  
4. Use the cameras at indices `[0]` and `[1]` from `src_cams` along with `num_interpolation` as inputs to run the `gen_new_cam` function from `scene/generate_new_cam` for the specified number of interpolations.  
5. Collect only the newly generated cameras into a dictionary with the same structure as `src_cams`.  
6. Create an `images` dictionary containing only extrinsic parameters, formatted to match the input of `write_images_txt`.  
7. Use the `write_cameras_txt` and `write_images_txt` functions from COLMAP's `read_write_model` script to save the outputs in the `output/scene_name` directory.  

## Contribute: Please submit [Issues](https://github.com/Three-Shots-Are-Enough/Gen-new-cam/issues)
 
