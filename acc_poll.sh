#!/bin/bash

# =========================================
# CalamariOCR ACC 轮询脚本
#
# 用法:
#   ./acc_poll.sh '{"mission_id":"2052665137923358720"}'
#
# 输出情况：
# 1. 参数不正确：
#    code=400, message=查询失败, status=3, msg=参数输入错误
#
# 2. 查询的 id 已经结束，被执行过，可以查到 ACC：
#    有 /app/adv_eval/acc_${mission_id}.txt
#    且有 /app/ACC_result/ACC_${mission_id}.txt
#    且有 /app/ACC_result/${mission_id}.zip
#    ACC 从 /app/adv_eval/acc_${mission_id}.txt 读取
#    code=200, message=查询成功, ACC=实际值, status=2
#
# 3. 查询的 id 正在执行中：
#    runner pid 或 calc_acc.py pid 正在运行
#    code=200, message=任务正在执行中, ACC=null, status=1
#
# 4. 查询的 id 任务失败：
#    曾经启动过，但当前没有进程，且没有完整结果
#    或有部分结果文件但缺少完整产物
#    code=200, message=任务失败, ACC=null, status=3
#
# 5. 查询的 id 未被启动过：
#    没有 pid file，也没有上述结果文件
#    code=1002, message=任务不存在, data={}
# =========================================

json_input="$1"

# =========================
# JSON 字段读取
# =========================
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

# =========================
# JSON 输出函数
# =========================
json_param_error() {
    echo "{
    \"code\": 400,
    \"message\": \"查询失败\",
    \"data\": {
        \"status\": \"3\",
        \"msg\": \"参数输入错误\"
    }
}"
}

json_success_done() {
    local acc="$1"

    echo "{
    \"code\": 200,
    \"message\": \"查询成功\",
    \"data\": {
        \"ACC\": \"$acc\",
        \"status\": \"2\"
    }
}"
}

json_running() {
    echo "{
    \"code\": 200,
    \"message\": \"任务正在执行中\",
    \"data\": {
        \"ACC\": \"null\",
        \"status\": \"1\"
    }
}"
}

json_task_failed() {
    echo "{
    \"code\": 200,
    \"message\": \"任务失败\",
    \"data\": {
        \"ACC\": \"null\",
        \"status\": \"3\"
    }
}"
}

json_not_exist() {
    echo "{
    \"code\": 1002,
    \"message\": \"任务不存在\",
    \"data\": {
    }
}"
}

# =========================
# 1. 参数检查
# =========================
if [ -z "$mission_id" ] || [ "$mission_id" = "None" ]; then
    json_param_error
    exit 1
fi

if ! [[ "$mission_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    json_param_error
    exit 1
fi

# =========================
# 2. 路径定义
# =========================
APP_ROOT="/app"

# ACC 数值文件：用于读取 ACC
middle_result_file="${APP_ROOT}/adv_eval/acc_${mission_id}.txt"

# ACC 明细文件：每行 样本名 实际标签 识别标签
final_result_file="${APP_ROOT}/ACC_result/ACC_${mission_id}.txt"

# ACC zip 文件：注意这里按你的要求放在 /app/ACC_result/
final_zip_file="${APP_ROOT}/ACC_result/${mission_id}.zip"

pid_file="/tmp/eval_acc_${mission_id}.pid"
task_pid_file="/tmp/eval_acc_task_${mission_id}.pid"
status_file="/tmp/eval_acc_status_${mission_id}"

# =========================
# 3. 判断 pid 是否还在运行
# =========================
is_pid_running() {
    local file="$1"

    if [ ! -f "$file" ]; then
        return 1
    fi

    local pid
    pid=$(cat "$file" 2>/dev/null)

    if ! [[ "$pid" =~ ^[0-9]+$ ]]; then
        return 1
    fi

    if kill -0 "$pid" 2>/dev/null; then
        return 0
    fi

    return 1
}

has_pid_file=0
pid_running=0

if [ -f "$pid_file" ] || [ -f "$task_pid_file" ]; then
    has_pid_file=1
fi

if is_pid_running "$pid_file" || is_pid_running "$task_pid_file"; then
    pid_running=1
fi

# =========================
# 4. 正在执行中
# =========================
if [ "$pid_running" -eq 1 ]; then
    json_running
    exit 0
fi

# =========================
# 5. 已完成：三个产物都存在
# =========================
if [ -f "$middle_result_file" ] && \
   [ -f "$final_result_file" ] && \
   [ -f "$final_zip_file" ]; then

    acc_line="$(head -n 1 "$middle_result_file" 2>/dev/null)"

    if [ -z "$acc_line" ]; then
        acc_value="null"
    else
        # 兼容：
        # ACC:0.8623
        # ACC: 86.23
        # 0.8623
        # 86.23
        acc_value="${acc_line#ACC:}"
        acc_value="$(echo "$acc_value" | xargs)"
    fi

    json_success_done "$acc_value"
    exit 0
fi

# =========================
# 6. 明确失败：status_file 标记 failed
# =========================
if [ -f "$status_file" ]; then
    current_stage="$(grep '^stage=' "$status_file" 2>/dev/null | tail -n 1 | cut -d= -f2-)"
    if [ "$current_stage" = "failed" ]; then
        json_task_failed
        exit 0
    fi
fi

# =========================
# 7. 任务失败：有部分结果，但缺少完整结果
# =========================

# 有 ACC 数值，但没有明细或 zip
if [ -f "$middle_result_file" ] && { [ ! -f "$final_result_file" ] || [ ! -f "$final_zip_file" ]; }; then
    json_task_failed
    exit 0
fi

# 有 ACC 明细，但没有 ACC 数值或 zip
if [ -f "$final_result_file" ] && { [ ! -f "$middle_result_file" ] || [ ! -f "$final_zip_file" ]; }; then
    json_task_failed
    exit 0
fi

# 有 zip，但没有 ACC 数值或明细
if [ -f "$final_zip_file" ] && { [ ! -f "$middle_result_file" ] || [ ! -f "$final_result_file" ]; }; then
    json_task_failed
    exit 0
fi

# =========================
# 8. 任务失败兜底
# 有 pid file，但是 pid 已经不运行，并且没有完整结果
# =========================
if [ "$has_pid_file" -eq 1 ] && [ "$pid_running" -eq 0 ]; then
    json_task_failed
    exit 0
fi

# =========================
# 9. 任务不存在
# =========================
json_not_exist
exit 0