#!/bin/bash

# =========================================
# CalamariOCR ACC 启动脚本
#
# 启动方式:
#   ./acc.sh '{"mission_id":"2052665137923358720","model_name":"CalamariOCR","model_class":"text","eval_method":"ACC"}'
#
# 输入数据:
#   /app/seed/<mission_id>.zip
#   或已存在:
#   /app/seed/<mission_id>/user_dataset/
#
# seed zip 解压后支持:
#   user_dataset/
#       png_dir/
#       value.txt 或 gt.txt
#       target.txt 可有可无
#
# 权重:
#   /app/weight/<mission_id>.zip
#   或已存在:
#   /app/weight/<mission_id>/
#   或默认:
#   /app/ocr_model/4.ckpt.*
#
# 权重 zip 解压后允许任意文件夹名:
#   xxx/
#       4.ckpt.json
#       4.ckpt.index
#       4.ckpt.data-00000-of-00001
#
# 输出:
#   /app/adv_eval/acc_<mission_id>.txt
#       只写 ACC 数值，例如 98.50
#
#   /app/ACC_result/ACC_<mission_id>.txt
#       写逐样本结果:
#       img_0000.png toy foy
#
#   /app/ACC_result/<mission_id>.zip
#       zip 内只包含:
#       ACC_<mission_id>.txt
#
# 注意:
#   本脚本不会写 /app/adv_sample/<mission_id>.zip，
#   避免覆盖对抗攻击结果 zip。
#
# PID:
#   /tmp/eval_acc_task_<mission_id>.pid
#   /tmp/eval_acc_<mission_id>.pid
#
# 状态:
#   /tmp/eval_acc_status_<mission_id>
# =========================================

SILENT_MODE=True

APP_ROOT="/app"
SEED_ROOT="${APP_ROOT}/seed"
WEIGHT_ROOT="${APP_ROOT}/weight"
ADV_EVAL_ROOT="${APP_ROOT}/adv_eval"
ACC_RESULT_ROOT="${APP_ROOT}/ACC_result"
LOG_DIR="${APP_ROOT}/run_logs"

VALID_MODEL_NAME="CalamariOCR"
BATCH_SIZE="32"

# =========================================
# JSON 返回函数
# =========================================
json_response() {
    local code="$1"
    local message="$2"
    local status="$3"

    echo "{
    \"code\": ${code},
    \"message\": \"${message}\",
    \"data\": {
        \"status\": \"${status}\"
    }
}"
}

json_param_error() {
    echo "{
    \"code\": 400,
    \"message\": \"任务失败\",
    \"data\": {
        \"status\": \"3\",
        \"msg\": \"参数输入错误\"
    }
}"
}

json_file_error() {
    local msg="$1"

    echo "{
    \"code\": 400,
    \"message\": \"任务失败\",
    \"data\": {
        \"status\": \"3\",
        \"msg\": \"${msg}\"
    }
}"
}

fail_response() {
    json_param_error
    exit 1
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
    LOG_FILE="${LOG_DIR}/run_acc_${mission_id}_${RUN_TS}.log"
    LATEST_LOG_FILE="${LOG_DIR}/run_acc_${mission_id}_latest.log"

    touch "$LOG_FILE"
    ln -sfn "$LOG_FILE" "$LATEST_LOG_FILE"

    {
        echo "============================================================"
        echo "ACC run log started"
        echo "mission_id: ${mission_id}"
        echo "model_name: ${test_model}"
        echo "model_class: ${model_class}"
        echo "eval_method: ${eval_method}"
        echo "timestamp: ${RUN_TS}"
        echo "log_file: ${LOG_FILE}"
        echo "latest_log_file: ${LATEST_LOG_FILE}"
        echo "SILENT_MODE: ${SILENT_MODE}"
        echo "============================================================"
    } >> "$LOG_FILE"
}

# =========================================
# 1. 参数校验
# =========================================
if [ "$#" -ne 1 ]; then
    fail_response
fi

json_input="$1"

mission_id=$(json_get "$json_input" "mission_id")
test_model=$(json_get "$json_input" "model_name")
model_class=$(json_get "$json_input" "model_class")
eval_method=$(json_get "$json_input" "eval_method")

if [ -z "$mission_id" ] || [ "$mission_id" = "None" ]; then
    fail_response
fi

if ! [[ "$mission_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    fail_response
fi

init_log

{
    echo "json_input: ${json_input}"
    echo "parsed mission_id: ${mission_id}"
    echo "parsed test_model: ${test_model}"
    echo "parsed model_class: ${model_class}"
    echo "parsed eval_method: ${eval_method}"
} >> "$LOG_FILE"

if [ -z "$test_model" ] || [ "$test_model" = "None" ]; then
    echo "参数检查失败: model_name 为空" >> "$LOG_FILE"
    fail_response
fi

if [ "$test_model" != "$VALID_MODEL_NAME" ]; then
    echo "参数检查失败: 不支持的 model_name=${test_model}" >> "$LOG_FILE"
    json_file_error "model_name 无效，仅支持 CalamariOCR"
    exit 1
fi

if [ -z "$model_class" ] || [ "$model_class" = "None" ]; then
    echo "参数检查失败: model_class 为空" >> "$LOG_FILE"
    fail_response
fi

if [ "$model_class" != "text" ]; then
    echo "参数检查失败: model_class=${model_class}, 当前只支持 text" >> "$LOG_FILE"
    json_file_error "model_class 无效"
    exit 1
fi

if [ -z "$eval_method" ] || [ "$eval_method" = "None" ]; then
    echo "参数检查失败: eval_method 为空" >> "$LOG_FILE"
    fail_response
fi

if [ "$eval_method" != "ACC" ]; then
    echo "参数检查失败: eval_method=${eval_method}, 当前只支持 ACC" >> "$LOG_FILE"
    json_file_error "eval_method 无效"
    exit 1
fi

# =========================================
# 2. seed / weight 文件检查
# seed 不等待，找不到直接 400
# weight 如果没有上传，则尝试默认权重
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
    json_file_error "seed 文件不存在"
    exit 1
fi

if [ ! -f "$weight_zip" ] && [ ! -d "$weight_dir" ] && [ ! -f "$default_weight_json" ]; then
    echo "weight 文件或默认权重不存在" >> "$LOG_FILE"
    json_file_error "weight 文件不存在"
    exit 1
fi

mkdir -p "$ADV_EVAL_ROOT" "$ACC_RESULT_ROOT"

# =========================================
# 3. 生成后台 runner
# =========================================
TASK_RUNNER_DIR="/tmp/eval_acc_task_runner"
mkdir -p "$TASK_RUNNER_DIR"
TASK_RUNNER_PATH="${TASK_RUNNER_DIR}/run_acc_${mission_id}.sh"

cat > "$TASK_RUNNER_PATH" <<EOF
#!/bin/bash
set +e

SILENT_MODE="${SILENT_MODE}"

APP_ROOT="${APP_ROOT}"
SEED_ROOT="${SEED_ROOT}"
WEIGHT_ROOT="${WEIGHT_ROOT}"
ADV_EVAL_ROOT="${ADV_EVAL_ROOT}"
ACC_RESULT_ROOT="${ACC_RESULT_ROOT}"

mission_id="${mission_id}"
test_model="${test_model}"
model_class="${model_class}"
eval_method="${eval_method}"
BATCH_SIZE="${BATCH_SIZE}"

LOG_FILE="${LOG_FILE}"
LATEST_LOG_FILE="${LATEST_LOG_FILE}"

EOF

cat >> "$TASK_RUNNER_PATH" <<'EOF'
log_msg() {
    local msg="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [mission_id=${mission_id}] [ACC] ${msg}"
}

if [ "$SILENT_MODE" = "True" ]; then
    exec >> "$LOG_FILE" 2>&1
else
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

STATUS_FILE="/tmp/eval_acc_status_${mission_id}"

TASK_PID_FILE="/tmp/eval_acc_task_${mission_id}.pid"
EVAL_PID_FILE="/tmp/eval_acc_${mission_id}.pid"

MODEL_DIR_FILE="/tmp/eval_acc_model_dir_${mission_id}"
MODEL_JSON_FILE="/tmp/eval_acc_model_json_${mission_id}"

SEED_ZIP="${SEED_ROOT}/${mission_id}.zip"
SEED_DIR="${SEED_ROOT}/${mission_id}"
SEED_USER_DATASET="${SEED_DIR}/user_dataset"

WEIGHT_ZIP="${WEIGHT_ROOT}/${mission_id}.zip"
WEIGHT_DIR="${WEIGHT_ROOT}/${mission_id}"

ACC_VALUE_FILE="${ADV_EVAL_ROOT}/acc_${mission_id}.txt"
ACC_RESULT_FILE="${ACC_RESULT_ROOT}/ACC_${mission_id}.txt"
FINAL_ZIP="${ACC_RESULT_ROOT}/${mission_id}.zip"

write_status() {
    local stage="$1"
    local detail="$2"

    {
        echo "stage=${stage}"
        echo "detail=${detail}"
        echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
    } > "$STATUS_FILE"
}

bg_fail() {
    local reason="$1"
    log_msg "ERROR: ${reason}"
    write_status "failed" "$reason"
    rm -f "$EVAL_PID_FILE"
    exit 1
}

cleanup_old_outputs() {
    write_status "cleanup" "running"
    log_msg "清理旧 ACC 输出"

    rm -f "$EVAL_PID_FILE"
    rm -f "$MODEL_DIR_FILE"
    rm -f "$MODEL_JSON_FILE"

    mkdir -p "$ADV_EVAL_ROOT"
    mkdir -p "$ACC_RESULT_ROOT"

    rm -f "$ACC_VALUE_FILE"
    rm -f "${ADV_EVAL_ROOT}/acc_${mission_id}_detail.json"
    rm -f "$ACC_RESULT_FILE"
    rm -f "$FINAL_ZIP"

    write_status "cleanup" "finished"
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

            tmp_user_dataset="/tmp/acc_user_dataset_${mission_id}_$$"
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
        bg_fail "seed 解压后未找到 user_dataset 目录: ${SEED_USER_DATASET}"
    fi

    if [ ! -d "${SEED_USER_DATASET}/png_dir" ]; then
        bg_fail "seed 缺少 png_dir: ${SEED_USER_DATASET}/png_dir"
    fi

    if [ ! -f "${SEED_USER_DATASET}/value.txt" ] && [ ! -f "${SEED_USER_DATASET}/gt.txt" ]; then
        bg_fail "seed 缺少 value.txt 或 gt.txt"
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
        unzip_ret=$?

        log_msg "unzip seed exit_code: ${unzip_ret}"

        if [ "$unzip_ret" -ne 0 ]; then
            bg_fail "seed 解压失败"
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
        bg_fail "权重目录下没有 .json 文件: ${root_dir}"
    fi

    if [ "${#json_files[@]}" -gt 1 ]; then
        log_msg "权重目录下发现多个 .json 文件:"
        printf '%s\n' "${json_files[@]}"
        bg_fail "权重目录下只能有一个 .json 文件"
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
        bg_fail "模型目录信息不存在"
    fi

    local model_dir
    local json_file

    model_dir=$(cat "$MODEL_DIR_FILE")
    json_file=$(cat "$MODEL_JSON_FILE")

    if [ ! -d "$model_dir" ]; then
        bg_fail "模型目录不存在: ${model_dir}"
    fi

    if [ ! -f "$json_file" ]; then
        bg_fail "模型 json 不存在: ${json_file}"
    fi

    local prefix="${json_file%.json}"

    if [ ! -f "${prefix}.index" ]; then
        bg_fail "缺少 TensorFlow checkpoint index 文件: ${prefix}.index"
    fi

    if ! ls "${prefix}".data-* >/dev/null 2>&1; then
        bg_fail "缺少 TensorFlow checkpoint data 文件: ${prefix}.data-*"
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
        bg_fail "默认权重缺少 4.ckpt.json"
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
        unzip_ret=$?

        log_msg "unzip weight exit_code: ${unzip_ret}"

        if [ "$unzip_ret" -ne 0 ]; then
            bg_fail "weight 解压失败"
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

run_calc_acc() {
    write_status "calc_acc" "running"

    local model_dir
    local model_json
    local model_json_name

    model_dir=$(cat "$MODEL_DIR_FILE")
    model_json=$(cat "$MODEL_JSON_FILE")
    model_json_name=$(basename "$model_json")

    log_msg "开始执行 calc_acc.py"
    log_msg "model_dir=${model_dir}"
    log_msg "model_json_name=${model_json_name}"
    log_msg "ACC_VALUE_FILE=${ACC_VALUE_FILE}"
    log_msg "ACC_RESULT_FILE=${ACC_RESULT_FILE}"

    cd "$APP_ROOT" || bg_fail "无法进入 ${APP_ROOT}"

    python3 calc_acc.py \
        --mission_id "$mission_id" \
        --app_root "$APP_ROOT" \
        --model_dir "$model_dir" \
        --model_path "$model_json_name" \
        --batch_size "$BATCH_SIZE" &

    pid=$!
    echo "$pid" > "$EVAL_PID_FILE"

    log_msg "calc_acc.py started, pid=${pid}, saved to ${EVAL_PID_FILE}"

    wait "$pid"
    rc=$?

    rm -f "$EVAL_PID_FILE"

    log_msg "calc_acc.py finished, exit_code=${rc}"

    if [ "$rc" -ne 0 ]; then
        bg_fail "calc_acc.py 执行失败，exit_code=${rc}"
    fi

    if [ ! -f "$ACC_VALUE_FILE" ]; then
        bg_fail "ACC 数值文件不存在: ${ACC_VALUE_FILE}"
    fi

    if [ ! -f "$ACC_RESULT_FILE" ]; then
        bg_fail "ACC 结果明细文件不存在: ${ACC_RESULT_FILE}"
    fi

    log_msg "ACC 数值:"
    cat "$ACC_VALUE_FILE" || true

    log_msg "ACC 明细前 10 行:"
    head -10 "$ACC_RESULT_FILE" || true

    write_status "calc_acc" "finished"
}

package_acc_result() {
    write_status "package" "running"

    mkdir -p "$ACC_RESULT_ROOT"

    if [ ! -f "$ACC_RESULT_FILE" ]; then
        bg_fail "ACC 结果文件不存在: ${ACC_RESULT_FILE}"
    fi

    rm -f "$FINAL_ZIP"

    log_msg "开始压缩 ACC 结果: ${ACC_RESULT_FILE} -> ${FINAL_ZIP}"

    cd "$ACC_RESULT_ROOT" || bg_fail "无法进入 ACC_result 目录: ${ACC_RESULT_ROOT}"

    zip -j "$FINAL_ZIP" "ACC_${mission_id}.txt"
    zip_ret=$?

    log_msg "zip ACC result exit_code: ${zip_ret}"

    if [ "$zip_ret" -ne 0 ]; then
        bg_fail "压缩 ACC 结果失败: ${FINAL_ZIP}"
    fi

    if [ ! -f "$FINAL_ZIP" ]; then
        bg_fail "最终 zip 不存在: ${FINAL_ZIP}"
    fi

    log_msg "ACC 结果压缩完成: ${FINAL_ZIP}"
    unzip -l "$FINAL_ZIP" || true

    write_status "package" "finished"
}

run_pipeline() {
    log_msg "============================================================"
    log_msg "CalamariOCR ACC 后台 runner 启动"
    log_msg "mission_id: ${mission_id}"
    log_msg "model_name: ${test_model}"
    log_msg "model_class: ${model_class}"
    log_msg "eval_method: ${eval_method}"
    log_msg "log_file: ${LOG_FILE}"
    log_msg "latest_log_file: ${LATEST_LOG_FILE}"
    log_msg "============================================================"

    write_status "starting" "runner started"

    cleanup_old_outputs
    prepare_seed
    prepare_weight
    run_calc_acc
    package_acc_result

    write_status "done" "success"

    log_msg "ACC 任务执行完成"
    log_msg "ACC 数值文件: ${ACC_VALUE_FILE}"
    log_msg "ACC 结果明细: ${ACC_RESULT_FILE}"
    log_msg "最终 zip: ${FINAL_ZIP}"
    log_msg "日志文件: ${LOG_FILE}"

    rm -f "$0"
    exit 0
}

run_pipeline
EOF

chmod 700 "$TASK_RUNNER_PATH"

# =========================================
# 4. 后台启动任务，立即返回 JSON
# =========================================
task_pid_file="/tmp/eval_acc_task_${mission_id}.pid"

if command -v setsid >/dev/null 2>&1; then
    nohup setsid bash "$TASK_RUNNER_PATH" >> "$LOG_FILE" 2>&1 < /dev/null &
else
    nohup bash "$TASK_RUNNER_PATH" >> "$LOG_FILE" 2>&1 < /dev/null &
fi

task_pid=$!
echo "$task_pid" > "$task_pid_file"

{
    echo "task_runner_path: ${TASK_RUNNER_PATH}"
    echo "task_pid_file: ${task_pid_file}"
    echo "task_pid: ${task_pid}"
    echo "eval_acc_pid_file: /tmp/eval_acc_${mission_id}.pid"
    echo "status_file: /tmp/eval_acc_status_${mission_id}"
    echo "model_dir_file: /tmp/eval_acc_model_dir_${mission_id}"
    echo "model_json_file: /tmp/eval_acc_model_json_${mission_id}"
    echo "acc_value_file: /app/adv_eval/acc_${mission_id}.txt"
    echo "acc_result_file: /app/ACC_result/ACC_${mission_id}.txt"
    echo "acc_zip_file: /app/ACC_result/${mission_id}.zip"
} >> "$LOG_FILE"

disown "$task_pid" 2>/dev/null || true

json_response 200 "成功" "1"

exit 0