# FAWA 工具说明（中文）

本说明针对两个新增脚本：

- `preprocess_png_to_pkl.py`
- `export_wm_result_images.py`

---

## 1. preprocess_png_to_pkl.py

### 功能说明

该脚本用于将原始 PNG 图像和对应文本标签，预处理成仓库已经使用的 pickle 数据格式：

- `img_data/{font_name}.pkl`
- `attack_pair/{font_name}-{case}.pkl`

其中：
- `img_data` 存储模型输入图像数据及真实文本
- `attack_pair` 存储真实文本与攻击目标文本对

### 输入文件和目录

必须准备：

- `png_dir`：包含待处理 PNG 图像的目录
- `gt_txt`：真实文本文件，每行对应一张 PNG 图片的真实文本
- `target_txt`：攻击目标文本文件，每行对应一张 PNG 图片的目标攻击文本

文件顺序要求：

- `png_dir` 中按文件名排序后的图片顺序必须与 `gt_txt`、`target_txt` 行顺序一致
- 每张 PNG 会对应 `gt_txt` 中的一行和 `target_txt` 中的一行

### 参数说明

- `--png_dir`：PNG 文件目录，必填
- `--gt_txt`：真实文本文件，必填
- `--target_txt`：目标文本文件，必填
- `--font_name`：字体名，用于生成 `img_data/{font_name}.pkl`
- `--case`：攻击 case 名称，用于生成 `attack_pair/{font_name}-{case}.pkl`
- `--height`：统一缩放后的图像高度，默认 `48`
- `--pad_width`：可选的固定宽度；如果不填，则使用当前所有图片的最大宽度
- `--output_img_data`：输出 `img_data` 目录，默认 `img_data`
- `--output_attack_pair`：输出 `attack_pair` 目录，默认 `attack_pair`
- `--ext`：PNG 文件后缀，默认 `png`

### 输出说明

脚本会生成：

- `img_data/{font_name}.pkl`
  - `(input_img, len_x, gt_txt)`
  - `input_img`：`numpy.ndarray`，形状 `(N, width, height)`，值范围 `[0, 1]`
  - `len_x`：每张图片对应的长度值列表
  - `gt_txt`：真实文本列表

- `attack_pair/{font_name}-{case}.pkl`
  - `(gt_txt, target_txt)`
  - `gt_txt`：真实文本列表
  - `target_txt`：目标攻击文本列表

### 使用示例

```bash
python preprocess_png_to_pkl.py \
  --png_dir ./my_pngs \
  --gt_txt ./gt.txt \
  --target_txt ./target.txt \
  --font_name Courier \
  --case easy
```

---

## 2. export_wm_result_images.py

### 功能说明

该脚本用于从 `wm_result/*.pkl` 文件中提取图片，生成 PNG 文件。

`wm_result` pickle 通常包含：

- `adv_img`：最终对抗图像
- `wm0_img`：带初始水印的图像
- `rgb_img`：可视化 RGB 输出图像

### 输入文件和目录

- `--input`：支持单个 `wm_result` pickle 文件，也支持目录
- 目录输入时，会批量处理匹配 `--pattern` 的所有 `.pkl` 文件

### 参数说明

- `--input`：输入文件或目录，默认 `wm_result`
- `--output`：输出目录，默认 `exported_wm_images`
- `--save_adv`：导出 `adv_img` 图像
- `--save_rgb`：导出 `rgb_img` 图像
- `--save_wm0`：导出 `wm0_img` 图像
- `--pattern`：当输入为目录时，匹配的文件模式，默认 `*.pkl`

### 输出说明

脚本会在输出目录下生成子目录：

- `{pkl_name}_adv`
- `{pkl_name}_rgb`
- `{pkl_name}_wm0`

每个子目录里会按 `adv_0000.png`、`rgb_0000.png`、`wm0_0000.png` 的方式保存对应图像。

### 使用示例

批量导出目录里所有 `wm_result`：

```bash
python export_wm_result_images.py --input wm_result --save_adv --save_rgb
```

导出单个 `pkl`：

```bash
python export_wm_result_images.py --input wm_result/Arial-easy-linf-eps0.3-ieps0.05-iter10-positive.pkl --save_adv --save_rgb
```
