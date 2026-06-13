#!/bin/bash

# =========================================
# FAWA OCR Watermark Attack run.sh
#
# 启动方式:
#   ./run.sh '{"mission_id":"2059839135698386944","model_name":"CalamariOCR","model_class":"text","method":"Watermark","epoch":20}'
#
# 参数说明:
# - mission_id: 任务 ID
# - model_name: 固定只能是 CalamariOCR
# - model_class: 固定 text
# - method: 固定 Watermark
# - epoch: 攻击迭代轮次，同时用于 basic_grad.py 和 wm_grad.py 的 nb_iter
#
# 数据集 zip:
#   /app/seed/<mission_id>.zip
#
# 解压后期望:
#   user_dataset/
#       png_dir/
#       value.txt 或 gt.txt
#       target.txt
#
# 最终落盘:
#   /app/seed/<mission_id>/user_dataset/
#
# 权重 zip:
#   /app/weight/<mission_id>.zip
#
# 解压后允许任意顶层文件夹名，例如:
#   some_model_name/
#       4.ckpt.json
#       4.ckpt.index
#       4.ckpt.data-00000-of-00001
#
# 最终落盘:
#   /app/weight/<mission_id>/<任意文件夹名>/
#
# 脚本会递归查找唯一 .json，并把其所在目录作为 --model_dir 传入攻击脚本。
#
# 最终输出:
#   /app/adv_sample/Attack_generation_CalamariOCR_<mission_id>/
#   /app/adv_sample/<mission_id>.zip
#
# zip 内顶层目录:
#   Attack_generation_CalamariOCR_<mission_id>/
# =========================================

SILENT_MODE=True

APP_ROOT="/app"
SEED_ROOT="${APP_ROOT}/seed"
WEIGHT_ROOT="${APP_ROOT}/weight"
ADV_SAMPLE_ROOT="${APP_ROOT}/adv_sample"
LOG_DIR="/app/run_logs"

# model_name 固定为 CalamariOCR
VALID_MODEL_NAME="CalamariOCR"

# 注意：项目内部 pkl 文件名仍然依赖 font_name。
# 当前数据和权重流程使用 Arial，所以这里固定为 Arial。
FONT_NAME="Arial"

# 固定攻击参数
CASE_NAME="easy"
PERT_TYPE="inf"
EPS="0.6"
EPS_ITER="0.05"
BATCH_SIZE="100"
CLIP_MIN="0.0"
CLIP_MAX="1.0"

# =========================================
# JSON 返回函数
# =========================================
return_json() {
    local code="$1"
    local msg="$2"
    local status="$3"

    echo "{
    \"code\": ${code},
    \"message\": \"${msg}\",
    \"data\": {
        \"status\": \"${status}\"
    }
}"
}

# =========================================
# JSON 字段读取
# =========================================
json_get() {
    local raw_json="$1"
    local key="$2"

    python3 - "$raw_json" "$key" <<'PY' 2>/dev/null
import sys
import json

raw = sys.argv[1]
key = sys.argv[2]

try:
    data = json.loads(raw)
    value = data.get(key, "None")
except Exception:
    value = "None"

if value is None:
    print("None")
else:
    print(value)
PY
}

# =========================================
# 日志初始化
# =========================================
init_log() {
    mkdir -p "$LOG_DIR"

    RUN_TS=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="${LOG_DIR}/run_${mission_id}_${RUN_TS}.log"
    LATEST_LOG_FILE="${LOG_DIR}/run_${mission_id}_latest.log"

    touch "$LOG_FILE"
    ln -sfn "$LOG_FILE" "$LATEST_LOG_FILE"

    {
        echo "============================================================"
        echo "FAWA run log started"
        echo "mission_id: ${mission_id}"
        echo "model_name: ${test_model}"
        echo "internal_font_name: ${FONT_NAME}"
        echo "model_class: ${model_class}"
        echo "method: ${test_method}"
        echo "epoch: ${epoch}"
        echo "timestamp: ${RUN_TS}"
        echo "log_file: ${LOG_FILE}"
        echo "latest_log_file: ${LATEST_LOG_FILE}"
        echo "SILENT_MODE: ${SILENT_MODE}"
        echo "============================================================"
    } >> "$LOG_FILE"
}

# =========================================
# 1. 参数解析
# =========================================
json_input="$1"

if [ -z "$json_input" ]; then
    return_json 400 "参数不合法" 3
    exit 1
fi

mission_id=$(json_get "$json_input" "mission_id")
test_model=$(json_get "$json_input" "model_name")
model_class=$(json_get "$json_input" "model_class")
test_method=$(json_get "$json_input" "method")
epoch=$(json_get "$json_input" "epoch")
time_out=$(json_get "$json_input" "timeout")

if [ -z "$mission_id" ] || [ "$mission_id" = "None" ]; then
    return_json 400 "mission_id 不能为空" 3
    exit 1
fi

if ! [[ "$mission_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    return_json 400 "mission_id 不合法" 3
    exit 1
fi

init_log

{
    echo "json_input: ${json_input}"
    echo "parsed mission_id: ${mission_id}"
    echo "parsed model_name: ${test_model}"
    echo "internal FONT_NAME: ${FONT_NAME}"
    echo "parsed model_class: ${model_class}"
    echo "parsed method: ${test_method}"
    echo "parsed epoch: ${epoch}"
    echo "parsed timeout: ${time_out}"
} >> "$LOG_FILE"

# =========================================
# 2. 参数检查
# =========================================
if [ -z "$test_model" ] || [ "$test_model" = "None" ]; then
    echo "model_name 为空" >> "$LOG_FILE"
    return_json 400 "model_name 不能为空" 3
    exit 1
fi

if [ "$test_model" != "$VALID_MODEL_NAME" ]; then
    echo "model_name 无效: ${test_model}" >> "$LOG_FILE"
    return_json 400 "model_name 无效，仅支持 CalamariOCR" 3
    exit 1
fi

safe_model_name="CalamariOCR"

if [ -z "$model_class" ] || [ "$model_class" = "None" ]; then
    echo "model_class 为空" >> "$LOG_FILE"
    return_json 400 "model_class 不能为空" 3
    exit 1
fi

if [ "$model_class" != "text" ]; then
    echo "model_class 无效: ${model_class}" >> "$LOG_FILE"
    return_json 400 "model_class 无效" 3
    exit 1
fi

if [ -z "$test_method" ] || [ "$test_method" = "None" ]; then
    echo "method 为空" >> "$LOG_FILE"
    return_json 400 "method 不能为空" 3
    exit 1
fi

if [ "$test_method" != "Watermark" ]; then
    echo "method 无效: ${test_method}" >> "$LOG_FILE"
    return_json 400 "method 无效" 3
    exit 1
fi

if [ -z "$epoch" ] || [ "$epoch" = "None" ]; then
    echo "epoch 为空" >> "$LOG_FILE"
    return_json 400 "epoch 不能为空" 3
    exit 1
fi

if ! [[ "$epoch" =~ ^[0-9]+$ ]]; then
    echo "epoch 不是正整数: ${epoch}" >> "$LOG_FILE"
    return_json 400 "epoch 必须是正整数" 3
    exit 1
fi

if [ "$epoch" -le 0 ]; then
    echo "epoch 小于等于 0: ${epoch}" >> "$LOG_FILE"
    return_json 400 "epoch 必须大于 0" 3
    exit 1
fi

# timeout 可选。如果没传，不限制阶段执行时间
if [ -z "$time_out" ] || [ "$time_out" = "None" ]; then
    time_out="None"
else
    if ! [[ "$time_out" =~ ^[0-9]+$ ]]; then
        echo "timeout 不是正整数: ${time_out}" >> "$LOG_FILE"
        return_json 400 "timeout 必须是正整数" 3
        exit 1
    fi
fi

# =========================================
# 3. seed / weight 文件存在性检查
# seed 不等待，找不到直接 400
# =========================================
seed_zip="${SEED_ROOT}/${mission_id}.zip"
seed_dir="${SEED_ROOT}/${mission_id}"
seed_user_dataset="${seed_dir}/user_dataset"

weight_zip="${WEIGHT_ROOT}/${mission_id}.zip"
weight_dir="${WEIGHT_ROOT}/${mission_id}"
default_weight_json="${APP_ROOT}/ocr_model/4.ckpt.json"

{
    echo "seed_zip: ${seed_zip}"
    echo "seed_dir: ${seed_dir}"
    echo "seed_user_dataset: ${seed_user_dataset}"
    echo "weight_zip: ${weight_zip}"
    echo "weight_dir: ${weight_dir}"
    echo "default_weight_json: ${default_weight_json}"
} >> "$LOG_FILE"

if [ ! -f "$seed_zip" ] && [ ! -d "$seed_user_dataset" ]; then
    echo "seed 文件或目录不存在" >> "$LOG_FILE"
    return_json 400 "seed 文件不存在" 3
    exit 1
fi

if [ ! -f "$weight_zip" ] && [ ! -d "$weight_dir" ] && [ ! -f "$default_weight_json" ]; then
    echo "weight 文件或默认权重不存在" >> "$LOG_FILE"
    return_json 400 "weight 文件不存在" 3
    exit 1
fi

mkdir -p "$ADV_SAMPLE_ROOT"

# =========================================
# 4. 生成后台 runner
# =========================================
TASK_RUNNER_DIR="/tmp/fawa_task_runner"
mkdir -p "$TASK_RUNNER_DIR"
TASK_RUNNER_PATH="${TASK_RUNNER_DIR}/run_fawa_${mission_id}.sh"

cat > "$TASK_RUNNER_PATH" <<EOF
#!/bin/bash
set +e

SILENT_MODE="${SILENT_MODE}"

APP_ROOT="${APP_ROOT}"
SEED_ROOT="${SEED_ROOT}"
WEIGHT_ROOT="${WEIGHT_ROOT}"
ADV_SAMPLE_ROOT="${ADV_SAMPLE_ROOT}"

mission_id="${mission_id}"
test_model="${test_model}"
safe_model_name="${safe_model_name}"
FONT_NAME="${FONT_NAME}"
model_class="${model_class}"
test_method="${test_method}"
epoch="${epoch}"
time_out="${time_out}"

CASE_NAME="${CASE_NAME}"
PERT_TYPE="${PERT_TYPE}"
EPS="${EPS}"
EPS_ITER="${EPS_ITER}"
BATCH_SIZE="${BATCH_SIZE}"
CLIP_MIN="${CLIP_MIN}"
CLIP_MAX="${CLIP_MAX}"

LOG_FILE="${LOG_FILE}"
LATEST_LOG_FILE="${LATEST_LOG_FILE}"

EOF

cat >> "$TASK_RUNNER_PATH" <<'EOF'
log_msg() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [mission_id=${mission_id}] ${msg}"
}

if [ "$SILENT_MODE" = "True" ]; then
    exec >> "$LOG_FILE" 2>&1
else
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

STATUS_FILE="/tmp/fawa_status_${mission_id}"

PID_PREPROCESS="/tmp/pid_fawa_preprocess_${mission_id}"
PID_BASIC="/tmp/pid_fawa_basic_grad_${mission_id}"
PID_WM="/tmp/pid_fawa_wm_grad_${mission_id}"
PID_EXPORT="/tmp/pid_fawa_export_${mission_id}"

SEED_ZIP="${SEED_ROOT}/${mission_id}.zip"
SEED_DIR="${SEED_ROOT}/${mission_id}"
SEED_USER_DATASET="${SEED_DIR}/user_dataset"

WEIGHT_ZIP="${WEIGHT_ROOT}/${mission_id}.zip"
WEIGHT_DIR="${WEIGHT_ROOT}/${mission_id}"

MODEL_DIR_FILE="/tmp/fawa_model_dir_${mission_id}"
MODEL_JSON_FILE="/tmp/fawa_model_json_${mission_id}"

GENERATED_FOLDER="Attack_generation_${safe_model_name}_${mission_id}"
GENERATED_FOLDER_PATH="${ADV_SAMPLE_ROOT}/${GENERATED_FOLDER}"
OLD_GENERATED_FOLDER_PATH="${APP_ROOT}/${GENERATED_FOLDER}"

TMP_EXPORT_DIR="${ADV_SAMPLE_ROOT}/_export_tmp_${mission_id}"
OLD_TMP_EXPORT_DIR="${APP_ROOT}/_export_tmp_${mission_id}"
FINAL_ZIP="${ADV_SAMPLE_ROOT}/${mission_id}.zip"

ADV_EVAL_DIR="${APP_ROOT}/adv_eval"
ADV_EVAL_FILE="${ADV_EVAL_DIR}/${mission_id}.txt"

BASIC_RESULT_PKL="${APP_ROOT}/attack_result/${mission_id}-${FONT_NAME}-${CASE_NAME}-l${PERT_TYPE}-eps${EPS}-ieps${EPS_ITER}-iter${epoch}.pkl"
WM_RESULT_PKL="${APP_ROOT}/wm_result/${mission_id}-${FONT_NAME}-${CASE_NAME}-l${PERT_TYPE}-eps${EPS}-ieps${EPS_ITER}-iter${epoch}-positive.pkl"

write_status() {
    local stage="$1"
    local detail="$2"

    {
        echo "stage=${stage}"
        echo "detail=${detail}"
        echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
    } > "$STATUS_FILE"
}

cleanup_pid_files() {
    rm -f "$PID_PREPROCESS" "$PID_BASIC" "$PID_WM" "$PID_EXPORT"
}

fail_task() {
    local stage="$1"
    local ret="$2"
    local msg="$3"

    log_msg "任务失败: stage=${stage}, exit_code=${ret}, msg=${msg}"
    write_status "failed" "stage=${stage};exit_code=${ret};msg=${msg}"
    cleanup_pid_files
    exit "$ret"
}

run_stage() {
    local stage="$1"
    local pid_file="$2"
    shift 2

    write_status "$stage" "running"
    log_msg "开始阶段: ${stage}"
    log_msg "命令: $*"

    "$@" &
    local stage_pid=$!

    echo "$stage_pid" > "$pid_file"
    log_msg "${stage} pid: ${stage_pid}"
    log_msg "${stage} pid_file: ${pid_file}"

    wait "$stage_pid"
    local ret=$?

    rm -f "$pid_file"

    log_msg "阶段结束: ${stage}, exit_code=${ret}"

    if [ "$ret" -ne 0 ]; then
        fail_task "$stage" "$ret" "${stage} 执行失败"
    fi

    write_status "$stage" "finished"
}

# =========================================
# seed 处理
# 支持:
# 1. zip 顶层直接是 user_dataset/
# 2. zip 顶层是任意目录，里面有 user_dataset/
# 3. zip 顶层直接是 png_dir/value.txt/target.txt
# 最终统一为:
#   /app/seed/<mission_id>/user_dataset/
# =========================================
normalize_seed_after_unzip() {
    log_msg "开始规范化 seed 目录: ${SEED_DIR}"

    if [ -d "${SEED_USER_DATASET}" ]; then
        log_msg "检测到目标 seed 结构: ${SEED_USER_DATASET}"
    else
        nested_user_dataset=$(find "${SEED_DIR}" -mindepth 1 -maxdepth 4 -type d -name "user_dataset" | head -n 1)

        if [ -n "$nested_user_dataset" ] && [ -d "$nested_user_dataset" ]; then
            log_msg "检测到 user_dataset: ${nested_user_dataset}"

            tmp_user_dataset="/tmp/user_dataset_${mission_id}_$$"
            rm -rf "$tmp_user_dataset"

            mv "$nested_user_dataset" "$tmp_user_dataset"

            find "${SEED_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} \;

            mv "$tmp_user_dataset" "${SEED_USER_DATASET}"
        elif [ -d "${SEED_DIR}/png_dir" ]; then
            log_msg "检测到扁平 seed 结构，整理为 user_dataset"
            mkdir -p "${SEED_USER_DATASET}"

            mv "${SEED_DIR}/png_dir" "${SEED_USER_DATASET}/"

            if [ -f "${SEED_DIR}/value.txt" ]; then
                mv "${SEED_DIR}/value.txt" "${SEED_USER_DATASET}/"
            fi

            if [ -f "${SEED_DIR}/gt.txt" ]; then
                mv "${SEED_DIR}/gt.txt" "${SEED_USER_DATASET}/"
            fi

            if [ -f "${SEED_DIR}/target.txt" ]; then
                mv "${SEED_DIR}/target.txt" "${SEED_USER_DATASET}/"
            fi
        fi
    fi

    if [ ! -d "${SEED_USER_DATASET}" ]; then
        fail_task "prepare_seed" 1 "seed 解压后未找到 user_dataset 目录: ${SEED_USER_DATASET}"
    fi

    if [ ! -d "${SEED_USER_DATASET}/png_dir" ]; then
        fail_task "prepare_seed" 1 "seed 缺少 png_dir: ${SEED_USER_DATASET}/png_dir"
    fi

    if [ ! -f "${SEED_USER_DATASET}/value.txt" ] && [ ! -f "${SEED_USER_DATASET}/gt.txt" ]; then
        fail_task "prepare_seed" 1 "seed 缺少 value.txt 或 gt.txt"
    fi

    if [ ! -f "${SEED_USER_DATASET}/target.txt" ]; then
        fail_task "prepare_seed" 1 "seed 缺少 target.txt"
    fi

    log_msg "seed 目录规范化完成"
    log_msg "seed_user_dataset: ${SEED_USER_DATASET}"
    ls -lh "${SEED_USER_DATASET}" || true
}

prepare_seed() {
    write_status "prepare_seed" "running"

    if [ -f "$SEED_ZIP" ]; then
        log_msg "发现 seed zip: ${SEED_ZIP}"

        rm -rf "$SEED_DIR"
        mkdir -p "$SEED_DIR"

        unzip -q "$SEED_ZIP" -d "$SEED_DIR"
        local unzip_ret=$?

        log_msg "unzip seed exit_code: ${unzip_ret}"

        if [ "$unzip_ret" -ne 0 ]; then
            fail_task "prepare_seed" "$unzip_ret" "seed 解压失败"
        fi
    else
        log_msg "未发现 seed zip，使用已存在目录: ${SEED_USER_DATASET}"
    fi

    normalize_seed_after_unzip

    write_status "prepare_seed" "finished"
}

# =========================================
# weight 处理
# 支持:
# 1. /app/weight/<mission_id>.zip
# 2. /app/weight/<mission_id>/
# 3. 默认 /app/ocr_model/4.ckpt.*
#
# zip 解压后允许任意文件夹名。
# 脚本递归查找唯一 .json 文件，并将其所在目录作为 MODEL_DIR。
# =========================================
find_unique_model_json_dir() {
    local root_dir="$1"

    mapfile -t json_files < <(find "$root_dir" -type f -name "*.json" | sort)

    if [ "${#json_files[@]}" -eq 0 ]; then
        fail_task "prepare_weight" 1 "权重目录下没有 .json 文件: ${root_dir}"
    fi

    if [ "${#json_files[@]}" -gt 1 ]; then
        log_msg "权重目录下发现多个 .json 文件:"
        printf '%s\n' "${json_files[@]}"
        fail_task "prepare_weight" 1 "权重目录下只能有一个 .json 文件"
    fi

    local json_file="${json_files[0]}"
    local json_dir
    json_dir=$(dirname "$json_file")

    echo "$json_dir" > "$MODEL_DIR_FILE"
    echo "$json_file" > "$MODEL_JSON_FILE"

    log_msg "识别到模型 json: ${json_file}"
    log_msg "识别到模型目录: ${json_dir}"
}

validate_resolved_weight() {
    if [ ! -f "$MODEL_DIR_FILE" ] || [ ! -f "$MODEL_JSON_FILE" ]; then
        fail_task "prepare_weight" 1 "模型目录信息不存在"
    fi

    local model_dir
    local json_file

    model_dir=$(cat "$MODEL_DIR_FILE")
    json_file=$(cat "$MODEL_JSON_FILE")

    if [ ! -d "$model_dir" ]; then
        fail_task "prepare_weight" 1 "模型目录不存在: ${model_dir}"
    fi

    if [ ! -f "$json_file" ]; then
        fail_task "prepare_weight" 1 "模型 json 不存在: ${json_file}"
    fi

    local prefix="${json_file%.json}"

    if [ ! -f "${prefix}.index" ]; then
        fail_task "prepare_weight" 1 "缺少 TensorFlow checkpoint index 文件: ${prefix}.index"
    fi

    if ! ls "${prefix}".data-* >/dev/null 2>&1; then
        fail_task "prepare_weight" 1 "缺少 TensorFlow checkpoint data 文件: ${prefix}.data-*"
    fi

    log_msg "权重校验通过"
    log_msg "model_dir: ${model_dir}"
    log_msg "model_json: ${json_file}"
}

copy_default_weight() {
    log_msg "使用默认权重: ${APP_ROOT}/ocr_model/4.ckpt.*"

    rm -rf "$WEIGHT_DIR"
    mkdir -p "${WEIGHT_DIR}/default_model"

    cp "${APP_ROOT}/ocr_model/4.ckpt.json" "${WEIGHT_DIR}/default_model/" 2>/dev/null
    cp "${APP_ROOT}/ocr_model/4.ckpt.index" "${WEIGHT_DIR}/default_model/" 2>/dev/null
    cp "${APP_ROOT}/ocr_model/4.ckpt.data-00000-of-00001" "${WEIGHT_DIR}/default_model/" 2>/dev/null
    cp "${APP_ROOT}/ocr_model/4.ckpt.meta" "${WEIGHT_DIR}/default_model/" 2>/dev/null
    cp "${APP_ROOT}/ocr_model/4.ckpt.h5" "${WEIGHT_DIR}/default_model/" 2>/dev/null

    if [ ! -f "${WEIGHT_DIR}/default_model/4.ckpt.json" ]; then
        fail_task "prepare_weight" 1 "默认权重缺少 4.ckpt.json"
    fi
}

prepare_weight() {
    write_status "prepare_weight" "running"

    rm -f "$MODEL_DIR_FILE" "$MODEL_JSON_FILE"

    if [ -f "$WEIGHT_ZIP" ]; then
        log_msg "发现上传权重 zip: ${WEIGHT_ZIP}"

        rm -rf "$WEIGHT_DIR"
        mkdir -p "$WEIGHT_DIR"

        unzip -q "$WEIGHT_ZIP" -d "$WEIGHT_DIR"
        local unzip_ret=$?

        log_msg "unzip weight exit_code: ${unzip_ret}"

        if [ "$unzip_ret" -ne 0 ]; then
            fail_task "prepare_weight" "$unzip_ret" "weight 解压失败"
        fi

        log_msg "weight zip 解压后目录结构:"
        find "$WEIGHT_DIR" -maxdepth 4 -print | head -100 || true

    elif [ -d "$WEIGHT_DIR" ]; then
        log_msg "未发现 weight zip，使用已存在权重目录: ${WEIGHT_DIR}"
        find "$WEIGHT_DIR" -maxdepth 4 -print | head -100 || true
    else
        copy_default_weight
    fi

    find_unique_model_json_dir "$WEIGHT_DIR"
    validate_resolved_weight

    write_status "prepare_weight" "finished"
}

cleanup_old_outputs() {
    write_status "cleanup" "running"

    log_msg "清理旧输出"

    cleanup_pid_files

    # 清理旧版本和新版本可能留下的导出目录
    rm -rf "$TMP_EXPORT_DIR"
    rm -rf "$OLD_TMP_EXPORT_DIR"
    rm -rf "$GENERATED_FOLDER_PATH"
    rm -rf "$OLD_GENERATED_FOLDER_PATH"

    # 清理最终压缩包和旧版临时压缩包
    rm -f "$FINAL_ZIP"
    rm -f "${APP_ROOT}/${mission_id}.zip"

    # 清理状态、模型定位、进度文件。
    # STATUS_FILE 当前阶段由 write_status 维护，不在这里删除。
    rm -f "$MODEL_DIR_FILE"
    rm -f "$MODEL_JSON_FILE"
    rm -f "$ADV_EVAL_FILE"
    rm -f "${ADV_EVAL_FILE}.tmp"
    rm -f "/tmp/fawa_progress_${mission_id}.json"
    rm -f "/tmp/fawa_progress_${mission_id}.json.tmp"

    mkdir -p "${APP_ROOT}/img_data"
    mkdir -p "${APP_ROOT}/attack_pair"
    mkdir -p "${APP_ROOT}/attack_result"
    mkdir -p "${APP_ROOT}/wm_result"
    mkdir -p "$ADV_SAMPLE_ROOT"
    mkdir -p "$ADV_EVAL_DIR"

    rm -f "${APP_ROOT}/img_data/${mission_id}-${FONT_NAME}.pkl"
    rm -f "${APP_ROOT}/img_data/${mission_id}-${FONT_NAME}.meta.pkl"
    rm -f "${APP_ROOT}/attack_pair/${mission_id}-${FONT_NAME}-${CASE_NAME}.pkl"
    rm -f "${APP_ROOT}/attack_pair/${mission_id}-${FONT_NAME}-${CASE_NAME}.meta.pkl"
    rm -f "${APP_ROOT}/attack_result/${mission_id}-${FONT_NAME}-${CASE_NAME}-l${PERT_TYPE}-eps${EPS}-ieps${EPS_ITER}-iter${epoch}.pkl"
    rm -f "${APP_ROOT}/wm_result/${mission_id}-${FONT_NAME}-${CASE_NAME}-l${PERT_TYPE}-eps${EPS}-ieps${EPS_ITER}-iter${epoch}-positive.pkl"

    write_status "cleanup" "finished"
}

package_result() {
    write_status "package" "running"

    if [ ! -d "$GENERATED_FOLDER_PATH" ]; then
        fail_task "package" 1 "最终输出目录不存在: ${GENERATED_FOLDER_PATH}"
    fi

    mkdir -p "$ADV_SAMPLE_ROOT"
    cd "$ADV_SAMPLE_ROOT" || fail_task "package" 1 "无法进入 ${ADV_SAMPLE_ROOT}"

    rm -f "$FINAL_ZIP"

    log_msg "开始压缩: ${GENERATED_FOLDER_PATH} -> ${FINAL_ZIP}"
    zip -r "$FINAL_ZIP" "$GENERATED_FOLDER"
    local zip_ret=$?

    log_msg "zip exit_code: ${zip_ret}"

    if [ "$zip_ret" -ne 0 ]; then
        fail_task "package" "$zip_ret" "zip 压缩失败"
    fi

    log_msg "最终输出目录: ${GENERATED_FOLDER_PATH}"
    log_msg "最终 zip: ${FINAL_ZIP}"

    write_status "package" "finished"
}

run_basic_grad_stage() {
    local model_dir
    model_dir=$(cat "$MODEL_DIR_FILE")

    if [ "$time_out" = "None" ]; then
        run_stage "basic_grad" "$PID_BASIC" \
            python3 basic_grad.py \
                --mission_id "$mission_id" \
                --font_name="${FONT_NAME}" \
                --case="${CASE_NAME}" \
                --pert_type="${PERT_TYPE}" \
                --eps="${EPS}" \
                --eps_iter="${EPS_ITER}" \
                --nb_iter="${epoch}" \
                --batch_size="${BATCH_SIZE}" \
                --clip_min="${CLIP_MIN}" \
                --clip_max="${CLIP_MAX}" \
                --model_dir "$model_dir"
    else
        run_stage "basic_grad" "$PID_BASIC" \
            timeout --preserve-status "$time_out" \
            python3 basic_grad.py \
                --mission_id "$mission_id" \
                --font_name="${FONT_NAME}" \
                --case="${CASE_NAME}" \
                --pert_type="${PERT_TYPE}" \
                --eps="${EPS}" \
                --eps_iter="${EPS_ITER}" \
                --nb_iter="${epoch}" \
                --batch_size="${BATCH_SIZE}" \
                --clip_min="${CLIP_MIN}" \
                --clip_max="${CLIP_MAX}" \
                --model_dir "$model_dir"
    fi
}

run_wm_grad_stage() {
    local model_dir
    local model_json
    local model_json_name

    model_dir=$(cat "$MODEL_DIR_FILE")
    model_json=$(cat "$MODEL_JSON_FILE")
    model_json_name=$(basename "$model_json")

    if [ "$time_out" = "None" ]; then
        run_stage "wm_grad" "$PID_WM" \
            python3 wm_grad.py \
                "$FONT_NAME" \
                "$CASE_NAME" \
                "$PERT_TYPE" \
                "$EPS" \
                "$EPS_ITER" \
                "$epoch" \
                --mission_id "$mission_id" \
                --batch_size "$BATCH_SIZE" \
                --clip_min "$CLIP_MIN" \
                --clip_max "$CLIP_MAX" \
                --model_dir "$model_dir" \
                --model_path "$model_json_name"
    else
        run_stage "wm_grad" "$PID_WM" \
            timeout --preserve-status "$time_out" \
            python3 wm_grad.py \
                "$FONT_NAME" \
                "$CASE_NAME" \
                "$PERT_TYPE" \
                "$EPS" \
                "$EPS_ITER" \
                "$epoch" \
                --mission_id "$mission_id" \
                --batch_size "$BATCH_SIZE" \
                --clip_min "$CLIP_MIN" \
                --clip_max "$CLIP_MAX" \
                --model_dir "$model_dir" \
                --model_path "$model_json_name"
    fi
}

run_pipeline() {
    log_msg "============================================================"
    log_msg "FAWA 后台 runner 启动"
    log_msg "mission_id: ${mission_id}"
    log_msg "model_name: ${test_model}"
    log_msg "internal font_name: ${FONT_NAME}"
    log_msg "model_class: ${model_class}"
    log_msg "method: ${test_method}"
    log_msg "epoch/nb_iter: ${epoch}"
    log_msg "time_out: ${time_out}"
    log_msg "log_file: ${LOG_FILE}"
    log_msg "latest_log_file: ${LATEST_LOG_FILE}"
    log_msg "============================================================"

    write_status "starting" "runner started"

    cd "$APP_ROOT" || fail_task "starting" 1 "无法进入 ${APP_ROOT}"

    cleanup_old_outputs
    prepare_seed
    prepare_weight

    # =========================================
    # Stage 1: preprocess_png_to_pkl.py
    # =========================================
    run_stage "preprocess" "$PID_PREPROCESS" \
        python3 preprocess_png_to_pkl.py \
            --mission_id "$mission_id" \
            --font_name "$FONT_NAME" \
            --case "$CASE_NAME"

    if [ ! -f "${APP_ROOT}/img_data/${mission_id}-${FONT_NAME}.pkl" ]; then
        fail_task "preprocess" 1 "preprocess 后 img_data pkl 不存在"
    fi

    if [ ! -f "${APP_ROOT}/attack_pair/${mission_id}-${FONT_NAME}-${CASE_NAME}.pkl" ]; then
        fail_task "preprocess" 1 "preprocess 后 attack_pair pkl 不存在"
    fi

    # =========================================
    # Stage 2: basic_grad.py
    # epoch 用作 nb_iter
    # =========================================
    run_basic_grad_stage

    if [ ! -f "$BASIC_RESULT_PKL" ]; then
        fail_task "basic_grad" 1 "basic_grad 输出不存在: ${BASIC_RESULT_PKL}"
    fi

    # =========================================
    # Stage 3: wm_grad.py
    # epoch 同样用作 nb_iter
    # =========================================
    run_wm_grad_stage

    if [ ! -f "$WM_RESULT_PKL" ]; then
        fail_task "wm_grad" 1 "wm_grad 输出不存在: ${WM_RESULT_PKL}"
    fi

    # =========================================
    # Stage 4: export_wm_result_images.py
    # 先导出到临时目录，再移动为 /app/adv_sample/Attack_generation_CalamariOCR_<mission_id>
    # =========================================
    rm -rf "$TMP_EXPORT_DIR"
    mkdir -p "$TMP_EXPORT_DIR"

    run_stage "export" "$PID_EXPORT" \
        python3 export_wm_result_images.py \
            --mission_id "$mission_id" \
            --input "$WM_RESULT_PKL" \
            --output "$TMP_EXPORT_DIR" \
            --font_name "$FONT_NAME" \
            --case "$CASE_NAME" \
            --save_adv \
            --save_rgb

    exported_subdir=$(find "$TMP_EXPORT_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

    if [ -z "$exported_subdir" ] || [ ! -d "$exported_subdir" ]; then
        fail_task "export" 1 "export 后未找到结果子目录: ${TMP_EXPORT_DIR}"
    fi

    rm -rf "$GENERATED_FOLDER_PATH"

    mv "$exported_subdir" "$GENERATED_FOLDER_PATH"
    mv_ret=$?

    if [ "$mv_ret" -ne 0 ]; then
        fail_task "export" "$mv_ret" "重命名最终输出目录失败"
    fi

    rm -rf "$TMP_EXPORT_DIR"

    log_msg "最终输出目录: ${GENERATED_FOLDER_PATH}"
    find "$GENERATED_FOLDER_PATH" -maxdepth 3 -type f | head -80 || true

    package_result

    cleanup_pid_files

    write_status "done" "success"
    log_msg "任务执行完成"
    log_msg "最终 zip: ${FINAL_ZIP}"
    log_msg "日志文件: ${LOG_FILE}"

    rm -f "$0"
    exit 0
}

run_pipeline
EOF

chmod 700 "$TASK_RUNNER_PATH"

# =========================================
# 5. 后台启动 runner
# =========================================
# 清理可能残留的旧 runner pid。新的 pid 会在启动后重新写入。
rm -f "/tmp/fawa_runner_${mission_id}.pid"

if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$TASK_RUNNER_PATH" >> "$LOG_FILE" 2>&1 < /dev/null &
else
    nohup bash "$TASK_RUNNER_PATH" >> "$LOG_FILE" 2>&1 < /dev/null &
fi

runner_pid=$!
echo "$runner_pid" > "/tmp/fawa_runner_${mission_id}.pid"

disown "$runner_pid" 2>/dev/null || true

{
    echo "runner_pid: ${runner_pid}"
    echo "runner_pid_file: /tmp/fawa_runner_${mission_id}.pid"
    echo "task_runner_path: ${TASK_RUNNER_PATH}"
    echo "pid_preprocess: /tmp/pid_fawa_preprocess_${mission_id}"
    echo "pid_basic_grad: /tmp/pid_fawa_basic_grad_${mission_id}"
    echo "pid_wm_grad: /tmp/pid_fawa_wm_grad_${mission_id}"
    echo "pid_export: /tmp/pid_fawa_export_${mission_id}"
    echo "status_file: /tmp/fawa_status_${mission_id}"
    echo "model_dir_file: /tmp/fawa_model_dir_${mission_id}"
    echo "model_json_file: /tmp/fawa_model_json_${mission_id}"
    echo "adv_eval_file: /app/adv_eval/${mission_id}.txt"
    echo "generated_folder_path: /app/adv_sample/Attack_generation_CalamariOCR_${mission_id}"
} >> "$LOG_FILE"

# =========================================
# 6. 参数合法后立即返回
# =========================================
return_json 200 "参数合法" 1
exit 0