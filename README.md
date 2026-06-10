# FAWA

This repository is source code for ECML/PKDD paper "FAWA: Fast Adversarial Watermark Attack on OCR Systems".



## Setup

1. Set up a vitual python environment:

   `conda env create --file=environment.yml`

2. Enter this environment we create:

   `source activate fawa`

3. 

### Dependencies





## Data

Related data can be downloaded [here](https://drive.google.com/drive/folders/1LAu-tcWs4iE0nqlHyROnYtW2ea81KKUw?usp=sharing)

The data include:
- attack_pair
- img_data
- ocr_model


## Additional utility scripts

- `preprocess_png_to_pkl.py`
  - Convert a folder of PNG images plus ground-truth and target text files into the repository pickle format.
  - Produces `img_data/{font_name}.pkl` and `attack_pair/{font_name}-{case}.pkl`.
  - Example:
    ```bash
    python preprocess_png_to_pkl.py \
      --png_dir ./my_pngs \
      --gt_txt ./gt.txt \
      --target_txt ./target.txt \
      --font_name Courier \
      --case easy
    ```

- `export_wm_result_images.py`
  - Extract images from `wm_result/*.pkl` files.
  - Can export `adv_img`, `wm0_img`, and/or `rgb_img` as PNG files.
  - Example:
    ```bash
    python export_wm_result_images.py --input wm_result --save_adv --save_rgb
    ```


## Reference

1 [https://github.com/Calamari-OCR/calamari](https://github.com/Calamari-OCR/calamari) 

2 [https://github.com/Belval/TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) 

3 [https://github.com/strongman1995/Fast-Adversarial-Watermark-Attack-on-OCR](https://github.com/strongman1995/Fast-Adversarial-Watermark-Attack-on-OCR) 

4 [https://github.com/tensorﬂow/cleverhan](