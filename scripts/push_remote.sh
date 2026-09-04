#!/usr/bin/env bash
#
# 把本地仓库推送到远端。支持 GitHub / Gitee / 任意 Git 服务器。
#
# 用法:
#   ./scripts/push_remote.sh github [repo名] [public|private]
#   ./scripts/push_remote.sh gitee  [repo名] [public|private]
#   ./scripts/push_remote.sh custom <远端URL>
#
# 凭证来源（按优先级）:
#   GitHub -> $GITHUB_TOKEN，否则尝试 CodeBuddy OAuth 网关自动获取
#   Gitee  -> $GITEE_TOKEN（https://gitee.com/profile/personal_access_tokens 生成）
#
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[ok]${NC} $*"; }
warn() { echo -e "${YELLOW}[!!]${NC} $*"; }
die()  { echo -e "${RED}[xx]${NC} $*" >&2; exit 1; }

PLATFORM="${1:-github}"
REPO="${2:-rag-platform}"
VISIBILITY="${3:-public}"
PRIVATE="false"; [ "$VISIBILITY" = "private" ] && PRIVATE="true"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"

# ---------- 连通性预检 ----------
precheck() {
  local host="$1"
  local code
  # 注意：不要加 `|| echo 000`，curl 失败时 -w 本身就会输出 000，两者会拼接成 000000
  code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 6 --max-time 10 "https://$host" 2>/dev/null)
  [ -n "$code" ] || code="000"
  if [ "${code:0:3}" = "000" ]; then
    echo -e "${RED}--------------------------------------------------------------${NC}"
    echo -e "${RED}  网络不可达: https://$host${NC}"
    echo -e "${RED}  当前环境无法访问该平台，请换一台能访问的环境执行本脚本，${NC}"
    echo -e "${RED}  或改用下方列出的可达平台。${NC}"
    echo -e "${RED}--------------------------------------------------------------${NC}"
    return 1
  fi
  log "连通性检查通过 ($host -> HTTP $code)"
  return 0
}

# ---------- 获取 GitHub Token ----------
# 优先级: $GH_PAT（用户自备的 Personal Access Token）> $GITHUB_TOKEN > OAuth 网关
# 注意：OAuth 网关给出的是 GitHub App 令牌(ghu_ 前缀)，它能读仓库、调 REST API，
#       但没有 git-over-HTTPS 的写权限，也无法新建仓库 —— 建仓/推送必须用真正的 PAT。
fetch_github_token() {
  if [ -n "${GH_PAT:-}" ]; then
    GITHUB_TOKEN="$GH_PAT"; export GITHUB_TOKEN
    log "使用 GH_PAT（Personal Access Token）"; return 0
  fi
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    case "$GITHUB_TOKEN" in
      ghu_*|ghs_*)
        warn "检测到 GitHub App 令牌（${GITHUB_TOKEN:0:4} 前缀）"
        warn "这类令牌无法新建仓库、也无法 git push，请改用 PAT："
        warn "  export GH_PAT=github_pat_xxxx   # https://github.com/settings/tokens"
        return 1 ;;
      *) log "使用环境中的 GITHUB_TOKEN"; return 0 ;;
    esac
  fi
  local script="/root/.codebuddy/skills/github-connector/scripts/get_token.sh"
  [ -f "$script" ] || { warn "未找到 OAuth 脚本，请 export GH_PAT=你的PAT"; return 1; }
  # shellcheck disable=SC1090
  source "$script" github >/dev/null 2>&1
  if [ -n "${GITHUB_TOKEN:-}" ]; then
    case "$GITHUB_TOKEN" in
      ghu_*|ghs_*)
        warn "OAuth 网关返回的是 GitHub App 令牌，权限不足以建仓/推送"
        warn "请自备 PAT：export GH_PAT=github_pat_xxxx"
        return 1 ;;
      *) log "已通过 OAuth 网关获取 GITHUB_TOKEN"; return 0 ;;
    esac
  fi
  warn "未能获取可用令牌，请 export GH_PAT=你的PAT"
  return 1
}

# ---------- 推送前权限自检（避免推到一半才 403）----------
check_push_permission() {
  local token="$1" repo_full="$2"
  local perms
  perms=$(curl -s -H "Authorization: Bearer $token" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/repos/$repo_full" \
          | sed -n 's/.*"permissions" *: *{\([^}]*\)}.*/\1/p')
  [ -n "$perms" ] || return 0   # 仓库刚建好可能查不到，跳过自检
  case "$perms" in
    *'"push":true'*) log "令牌对该仓库有 push 权限"; return 0 ;;
    *)
      warn "令牌对 $repo_full 没有 push 权限（permissions: {$perms}）"
      warn "请换一个带 repo 作用域的 PAT"
      return 1 ;;
  esac
}

# ---------- 创建远端仓库 ----------
create_remote() {
  local api="$1" token="$2" payload="$3"
  curl -s -X POST "$api" \
    -H "Authorization: Bearer $token" \
    -H "Content-Type: application/json" \
    -H "Accept: application/vnd.github+json" \
    -d "$payload" > /tmp/_mk_repo.json 2>/dev/null || { warn "创建仓库请求失败"; return 1; }

  if grep -q '"full_name"\|"html_url"\|"path"' /tmp/_mk_repo.json 2>/dev/null; then
    log "远端仓库已就绪"
    return 0
  fi
  if grep -q 'already exists' /tmp/_mk_repo.json 2>/dev/null; then
    warn "仓库已存在，将直接推送到它"
    return 0
  fi
  warn "创建仓库返回异常:"; head -c 300 /tmp/_mk_repo.json; echo
  return 1
}

# ---------- 主流程 ----------
cd "$(git rev-parse --show-toplevel)"

echo "分支: $BRANCH   平台: $PLATFORM   仓库: $REPO   可见性: $VISIBILITY"
echo

case "$PLATFORM" in
  github)
    precheck "api.github.com" || exit 1
    fetch_github_token || exit 1
    OWNER=$(curl -s -H "Authorization: Bearer $GITHUB_TOKEN" \
              -H "Accept: application/vnd.github+json" \
              https://api.github.com/user | sed -n 's/.*"login" *: *"\([^"]*\)".*/\1/p' | head -1)
    [ -n "$OWNER" ] || die "无法解析 GitHub 用户名，请检查 token 权限"
    log "GitHub 账号: $OWNER"

    create_remote "https://api.github.com/user/repos" "$GITHUB_TOKEN" \
      "{\"name\":\"$REPO\",\"private\":$PRIVATE,\"description\":\"RAG 全流程实操平台\"}" || exit 1

    check_push_permission "$GITHUB_TOKEN" "${OWNER}/${REPO}" || exit 1

    # PAT 用 oauth2:，GitHub App 安装令牌用 x-access-token:
    case "$GITHUB_TOKEN" in
      ghu_*|ghs_*) AUTH_PREFIX="x-access-token" ;;
      *)           AUTH_PREFIX="oauth2" ;;
    esac
    REMOTE="https://${AUTH_PREFIX}:${GITHUB_TOKEN}@github.com/${OWNER}/${REPO}.git"
    WEBURL="https://github.com/${OWNER}/${REPO}"
    ;;

  gitee)
    precheck "gitee.com" || exit 1
    [ -n "${GITEE_TOKEN:-}" ] || die "请先 export GITEE_TOKEN=你的码云私人令牌（需 repo 权限）"
    create_remote "https://gitee.com/api/v5/user/repos" "$GITEE_TOKEN" \
      "{\"name\":\"$REPO\",\"private\":$PRIVATE,\"description\":\"RAG 全流程实操平台\"}" || exit 1
    REMOTE="https://oauth2:${GITEE_TOKEN}@gitee.com/${GITEE_USER:-$(curl -s "https://gitee.com/api/v5/user?access_token=$GITEE_TOKEN" | sed -n 's/.*"login" *: *"\([^"]*\)".*/\1/p' | head -1)}/${REPO}.git"
    WEBURL="https://gitee.com/${REPO}"
    ;;

  custom)
    REMOTE="${2:-}"
    [ -n "$REMOTE" ] || die "用法: $0 custom <远端URL>"
    WEBURL="$REMOTE"
    ;;

  *)
    die "不支持的平台: $PLATFORM（可选 github / gitee / custom）"
    ;;
esac

# ---------- 清理旧 remote 并推送 ----------
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE"
log "remote origin 已设置"

git push -u origin "$BRANCH" 2>&1 | sed -E 's/(ghp_|gho_|ghu_|ghs_|github_pat_)[A-Za-z0-9_]+/[REDACTED]/g'

# ---------- 清理含 token 的 remote，避免明文凭证留在 .git/config ----------
if [ "$PLATFORM" != "custom" ]; then
  CLEAN=$(echo "$REMOTE" | sed -E 's#//[^@]*@#//#' )
  git remote set-url origin "$CLEAN"
  log "已清除 remote 中的明文凭证"
fi

echo
log "推送完成 -> $WEBURL"
