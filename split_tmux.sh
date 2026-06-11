#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<'EOF'
用法:
  tmux_multi_ssh.sh [选项] SESSION HOST_LIST_FILE

参数:
  SESSION           要创建或附加的 tmux 会话名称
  HOST_LIST_FILE    包含主机名/IP 的文件，每行一个；支持空行和以 # 开头的注释行

选项:
  -h, --help        显示本帮助并退出

示例:
  tmux_multi_ssh.sh my-session ./host_list.txt
EOF
}

# 解析选项（当前仅支持 -h/--help）
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    --) # 显式结束选项
      shift
      break
      ;;
    -*) # 未知选项
      echo "错误: 未知选项: $1" >&2
      echo
      print_usage
      exit 2
      ;;
    *)
      args+=("$1")
      ;;
  esac
  shift
done

# 将剩余位置参数（若有）加入
while [[ $# -gt 0 ]]; do
  args+=("$1")
  shift
done

# 位置参数：SESSION 与 HOST_FILE
SESSION="${args[0]:-}"
HOST_FILE="${args[1]:-}"

if [[ -z "${SESSION}" || -z "${HOST_FILE}" ]]; then
  echo "错误: 需要提供 SESSION 和 HOST_LIST_FILE。" >&2
  echo
  print_usage
  exit 2
fi

# 依赖检查
if ! command -v tmux >/dev/null 2>&1; then
  echo "错误: 未找到 tmux，请先安装 tmux。" >&2
  exit 127
fi
if ! command -v ssh >/dev/null 2>&1; then
  echo "错误: 未找到 ssh，请确保已安装 OpenSSH 客户端。" >&2
  exit 127
fi

# 主机文件检查
if [[ ! -f "$HOST_FILE" ]]; then
  echo "错误: 主机列表文件不存在: $HOST_FILE" >&2
  exit 1
fi

# 读取主机列表（忽略空行与以 # 开头的注释）
hosts=()
while IFS= read -r line || [[ -n "$line" ]]; do
  # 去除首尾空白
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  # 跳过空行与注释
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  hosts+=("$line")
done < "$HOST_FILE"

if [[ ${#hosts[@]} -eq 0 ]]; then
  echo "错误: 主机列表为空（或仅包含注释/空行）。" >&2
  exit 1
fi

# 若会话已存在，则直接附加
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "提示: 会话已存在，直接附加: $SESSION"
  exec tmux attach-session -t "$SESSION"
fi

# 创建新的 tmux 会话
tmux new-session -d -s "$SESSION"

# 在第一个 pane 执行 ssh
tmux send-keys -t "$SESSION" "ssh ${hosts[0]}" C-m

# 对剩余主机名创建 split-pane，并执行 ssh
for i in "${!hosts[@]}"; do
  if [[ "$i" -eq 0 ]]; then
    continue
  fi
  tmux split-window -t "$SESSION"
  tmux select-layout -t "$SESSION" tiled
  tmux send-keys -t "$SESSION" "ssh ${hosts[i]}" C-m
done

# 开启 pane 输入同步
tmux set-window-option -t "$SESSION" synchronize-panes on

# 进入 tmux 会话
exec tmux attach-session -t "$SESSION"
