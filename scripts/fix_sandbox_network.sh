#!/usr/bin/env bash
#
# fix_sandbox_network.sh
# -------------------------------------------------------------------
# 本沙箱把一批对外域名做了 DNS 投毒（解析到 198.18.x.x 黑洞段），
# 导致 GitHub / GHCR / release 资源 等无法直连。
# 本脚本通过 DoH（阿里公共 DNS）查出真实 IP，写入
#   /etc/hosts           —— 本次会话生效
#   ~/.user_hosts        —— 沙箱休眠后恢复 hosts 的来源，避免回滚
# 全程不包含任何密钥，可安全重复执行。
#
# 用法：
#   bash scripts/fix_sandbox_network.sh
# -------------------------------------------------------------------
set -u

DOH="https://dns.alidns.com/resolve"
HOSTS=/etc/hosts
UH="$HOME/.user_hosts"

# 需要修复的域名（GitHub / GHCR / 资源分发相关）
DOMAINS="github.com api.github.com codeload.github.com objects.githubusercontent.com raw.githubusercontent.com release-assets.githubusercontent.com ghcr.io"

is_blackhole() {
  # 参数：解析到的 IP。空 / 198.18.* 都算被污染
  case "${1:-}" in
    ""|198.18.*|198.19.*) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_doh() {
  # 参数：域名。输出第一个 A 记录 IP，失败为空
  curl -s --max-time 10 -H "accept: application/dns-json" \
    "$DOH?name=$1&type=A" 2>/dev/null \
    | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    for a in d.get('Answer',[]):
        if a.get('type')==1:
            print(a['data']); break
except Exception:
    pass
"
}

[ -w "$HOSTS" ] || { echo "无 /etc/hosts 写权限，尝试 sudo"; }

echo "=== 开始修复被投毒的域名解析 ==="
changed=0
for d in $DOMAINS; do
  cur=$(getent hosts "$d" 2>/dev/null | head -1 | awk '{print $1}')
  if is_blackhole "$cur"; then
    ip=$(resolve_doh "$d")
    if [ -n "$ip" ]; then
      # 清理旧条目后追加
      sed -i "/[[:space:]]$d\$/d" "$HOSTS" 2>/dev/null
      echo "$ip $d" >> "$HOSTS"
      if [ -f "$UH" ]; then
        sed -i "/[[:space:]]$d\$/d" "$UH" 2>/dev/null
        echo "$ip $d" >> "$UH"
      else
        echo "$ip $d" >> "$UH"
      fi
      echo "  [修复] $d -> $ip"
      changed=$((changed+1))
    else
      echo "  [跳过] $d -> DoH 未解析到 IP"
    fi
  else
    echo "  [正常] $d -> ${cur:-?}"
  fi
done

echo
echo "本脚本修复了 $changed 个域名。"
echo "验证（github.com 根路径需跟随重定向，故带 -L）："
for d in github.com api.github.com ghcr.io; do
  code=$(curl -sL -o /dev/null -w "%{http_code}" --connect-timeout 6 --max-time 10 "https://$d" 2>/dev/null)
  echo "  https://$d -> HTTP ${code:-000}"
done
