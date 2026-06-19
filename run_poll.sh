#!/bin/bash

# =========================================
# FAWA OCR Watermark Attack poll script
#
# 用法:
#   ./run_poll.sh '{"mission_id":"202606131512"}'
#
# 状态定义:
#   status: 1 = 正在执行中
#   status: 2 = 已结束 / 查询成功
#   status: 3 = 参数错误或任务失败
#
# 阶段返回:
#   数据处理
#   基础攻击
#   水印攻击
#   样本保存
#
# 水印阶段额外返回:
#   progress.epoch
#   progress.objective_loss
#
# 水印完整进度文件:
#   /app/adv_eval/<mission_id>.txt
# =========================================

json_input="$1"

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

mission_id=$(json_get "$json_input" "mission_id")

json_param_error() {
    echo "{
    \"code\": 400,
    \"message\": \"查询失败\",
    \"data\": {
        \"msg\": \"参数输入错误\",
        \"status\": \"3\"
    }
}"
}

json_running_stage() {
    local stage_name="$1"

    echo "{
    \"code\": 200,
    \"message\": \"任务正在执行中\",
    \"data\": {
        \"status\": \"1\",
        \"stage\": \"${stage_name}\"
    }
}"
}

json_running_wm() {
    local progress_file="$1"

    python3 - "$progress_file" <<'PY'
import json
import sys
from pathlib import Path

progress_file = Path(sys.argv[1])

epoch = 0
CTC_loss = None

if progress_file.exists():
    try:
        with progress_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        epoch = int(data.get("epoch", 0) or 0)

        objective_loss = data.get("objective_loss", None)
        if objective_loss is not None:
            CTC_loss = -float(objective_loss)
            CTC_loss = round(CTC_loss, 2)
    except Exception:
        epoch = 0
        CTC_loss = None

resp = {
    "code": 200,
    "message": "任务正在执行中",
    "data": {
        "status": "1",
        "stage": "水印攻击",
        "progress": {
            "epoch": epoch,
            "CTC_loss": CTC_loss
        }
    }
}

print(json.dumps(resp, ensure_ascii=False, indent=4))
PY
}

json_success_done() {
    echo "{
    \"code\": 200,
    \"message\": \"查询成功\",
    \"data\": {
        \"status\": \"2\",
        \"stage\": \"样本保存完成\"
    }
}"
}

json_task_failed() {
    local msg="$1"

    python3 - "$msg" <<'PY'
import json
import sys

msg = sys.argv[1]

resp = {
    "code": 200,
    "message": "任务失败",
    "data": {
        "status": "3",
        "stage": "任务失败",
        "msg": msg
    }
}

print(json.dumps(resp, ensure_ascii=False, indent=4))
PY
}

json_not_exist() {
    echo "{
    \"code\": 1002,
    \"message\": \"任务不存在。\",
    \"data\": {
    }
}"
}

# ===============================
# 参数检查
# ===============================
if [ -z "$mission_id" ] || [ "$mission_id" = "None" ]; then
    json_param_error
    exit 1
fi

if ! [[ "$mission_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    json_param_error
    exit 1
fi

# ===============================
# 路径定义
# ===============================
APP_ROOT="/app"
ADV_SAMPLE_ROOT="${APP_ROOT}/adv_sample"
ADV_EVAL_ROOT="${APP_ROOT}/adv_eval"
RUN_LOG_ROOT="${APP_ROOT}/run_logs"

runner_pid_file="/tmp/fawa_runner_${mission_id}.pid"

pid_preprocess="/tmp/pid_fawa_preprocess_${mission_id}"
pid_basic="/tmp/pid_fawa_basic_grad_${mission_id}"
pid_wm="/tmp/pid_fawa_wm_grad_${mission_id}"
pid_export="/tmp/pid_fawa_export_${mission_id}"

status_file="/tmp/fawa_status_${mission_id}"
progress_file="${ADV_EVAL_ROOT}/${mission_id}.txt"

final_zip="${ADV_SAMPLE_ROOT}/${mission_id}.zip"
generated_folder="${ADV_SAMPLE_ROOT}/Attack_generation_CalamariOCR_${mission_id}"
latest_log="${RUN_LOG_ROOT}/run_${mission_id}_latest.log"

# ===============================
# 工具函数
# ===============================
is_pid_running() {
    local pid_file="$1"

    if [ ! -f "$pid_file" ]; then
        return 1
    fi

    local pid
    pid=$(cat "$pid_file" 2>/dev/null)

    if [ -z "$pid" ]; then
        return 1
    fi

    if ps -p "$pid" >/dev/null 2>&1; then
        return 0
    fi

    return 1
}

read_status_field() {
    local key="$1"

    if [ ! -f "$status_file" ]; then
        return 1
    fi

    grep "^${key}=" "$status_file" 2>/dev/null | tail -n 1 | cut -d= -f2-
}

has_any_pid_file() {
    [ -f "$runner_pid_file" ] || \
    [ -f "$pid_preprocess" ] || \
    [ -f "$pid_basic" ] || \
    [ -f "$pid_wm" ] || \
    [ -f "$pid_export" ]
}

stage_to_display() {
    local raw_stage="$1"

    case "$raw_stage" in
        cleanup|prepare_seed|prepare_weight|preprocess|starting|running|unknown|"")
            echo "数据处理"
            ;;
        basic_grad)
            echo "基础攻击"
            ;;
        wm_grad)
            echo "水印攻击"
            ;;
        export|package)
            echo "样本保存"
            ;;
        done)
            echo "样本保存完成"
            ;;
        failed)
            echo "任务失败"
            ;;
        *)
            echo "数据处理"
            ;;
    esac
}

# ===============================
# 1. 正在执行中：优先根据四个阶段 pid 判断
# ===============================

if is_pid_running "$pid_preprocess"; then
    json_running_stage "数据处理"
    exit 0
fi

if is_pid_running "$pid_basic"; then
    json_running_stage "基础攻击"
    exit 0
fi

if is_pid_running "$pid_wm"; then
    json_running_wm "$progress_file"
    exit 0
fi

if is_pid_running "$pid_export"; then
    json_running_stage "样本保存"
    exit 0
fi

# ===============================
# 2. runner 还在运行，但当前没有 Python 阶段 pid
# 可能处于 cleanup / prepare_seed / prepare_weight / package 等 shell 阶段
# ===============================
if is_pid_running "$runner_pid_file"; then
    current_stage=$(read_status_field "stage")

    if [ "$current_stage" = "wm_grad" ]; then
        json_running_wm "$progress_file"
        exit 0
    fi

    display_stage=$(stage_to_display "$current_stage")
    json_running_stage "$display_stage"
    exit 0
fi

# ===============================
# 3. 已完成：最终 zip 存在
# ===============================
if [ -f "$final_zip" ]; then
    json_success_done
    exit 0
fi

# ===============================
# 4. 明确失败：status_file 标记 failed
# ===============================
if [ -f "$status_file" ]; then
    current_stage=$(read_status_field "stage")
    detail=$(read_status_field "detail")

    if [ "$current_stage" = "failed" ]; then
        if [ -z "$detail" ]; then
            detail="任务失败"
        fi
        json_task_failed "$detail"
        exit 0
    fi
fi

# ===============================
# 5. 曾经启动过，但当前无进程、无最终 zip
# ===============================
if has_any_pid_file && [ ! -f "$final_zip" ]; then
    json_task_failed "任务曾经启动，但当前无运行进程且没有最终 zip"
    exit 0
fi

# ===============================
# 6. 有中间输出目录但没有 zip，认为失败
# ===============================
if [ -d "$generated_folder" ] && [ ! -f "$final_zip" ]; then
    json_task_failed "存在输出目录但没有最终 zip"
    exit 0
fi

# ===============================
# 7. 有水印进度文件但没有 zip，认为失败
# ===============================
if [ -f "$progress_file" ] && [ ! -f "$final_zip" ]; then
    json_task_failed "存在水印攻击进度文件但没有最终 zip"
    exit 0
fi

# ===============================
# 8. 兜底：任务不存在
# ===============================
json_not_exist
exit 0