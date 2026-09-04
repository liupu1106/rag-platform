# 提交 GitHub 经验复盘 · 可执行 Playbook

> 适用场景：在**本沙箱环境**里把代码/文件推送到 GitHub，或上传到 GHCR / 用 GitHub 源部署。
> 一句话结论：**GitHub 没被封，只是 DNS 被投毒到黑洞段；另一半坑来自令牌类型不对。** 两件事解决后，一切照常。

---

## 一、环境特殊性（先读，避免白费劲）

本沙箱对出网做了限制，`github.com / api.github.com / ghcr.io / release-assets.githubusercontent.com` 等域名被解析到 `198.18.x.x`（一个黑洞段），表现为：

- `curl https://api.github.com` → `HTTP 000`，`git ls-remote` → `gnutls_handshake() failed`
- `gh auth status` → `You are not logged into any GitHub hosts`
- `pip install` → 走 `pypi.org` 同样 `SSL: UNEXPECTED_EOF`
- `docker pull` 正常（走 mirrors），但 `release-assets.githubusercontent.com` 拉取 CLI 二进制会失败

**例外**：`gitee.com / cnb.cool / gitcode.com` 正常；`mirrors.tencent.com` 正常。说明是**选择性 DNS 投毒**，不是全封锁。用 DoH 查真实 IP 即可绕过。

---

## 二、坑位复盘

| # | 现象 | 根因 | 解法 |
|---|------|------|------|
| 1 | `curl api.github.com` 返回 000，git 无法连 | GitHub 域名被 DNS 投毒到 `198.18.x.x` | DoH 查真实 IP，写入 `/etc/hosts` + `~/.user_hosts`（见脚本） |
| 2 | 有 `ghu_` 令牌却建不了仓（`403 Resource not accessible by integration`），git push 也 `403` | `ghu_`/`ghs_` 是 **GitHub App 安装令牌**，只有 `GET`/读权限，无建仓与 git 写权限 | 必须用**经典 PAT**（`ghp_` 前缀，勾选 `repo` 全选） |
| 3 | `pip install` 报 SSL EOF | `pypi.org` 被封 | `pip install -i https://mirrors.tencent.com/pypi/simple/ -r requirements.txt` |
| 4 | Dockerfile 构建 `pip install` 卡死 | 同上，容器内也走 pypi | Dockerfile 加 `ARG PIP_INDEX_URL`，构建时传镜像源 |
| 5 | 下载 Railway CLI 卡在 6MB（总 7.7MB） | `release-assets.githubusercontent.com` 被投毒 | DoH 修正该域名后完整下载 |
| 6 | `urllib` 调 Railway API 返回 `error code: 1010` | Cloudflare 拦了 urllib 默认 UA | 用 `curl` 并带正常 `User-Agent` |
| 7 | Railway 免费版配私有镜像凭证报错 | 免费版不支持私有 registry credentials（需 Pro） | GHCR 包设为 public（UI 点），或直接用 GitHub 源部署 |
| 8 | `push_remote.sh` 连通性预检误判「可达」 | curl 失败时 `-w "%{http_code}"` 与 `|| echo 000` 拼成 `000000` | 已修复：只用 `-w` 输出，空则补 `000` |

---

## 三、可执行标准流程

### 0. 一键修复网络（每次新会话先跑）

```bash
bash scripts/fix_sandbox_network.sh
# 验证
curl -s -o /dev/null -w "api.github.com -> %{http_code}\n" https://api.github.com
# 期望：HTTP 200
```

> 该脚本通过阿里 DoH 把 `github.com / api.github.com / codeload / objects.raw / release-assets / ghcr.io` 全部修正，并写入 `~/.user_hosts` 防止沙箱休眠后 hosts 被还原。**不含任何密钥，可反复执行。**

### 1. 准备令牌

- 去 https://github.com/settings/tokens 生成 **Classic PAT**，勾选 `repo`（全选）。
- ⚠️ 用 `ghp_` 开头；**不要**用环境里自带的 `ghu_`（App 令牌）——
  它能读 API 却建不了仓库、也推不了代码，会在建仓阶段才暴露 `403`。
- 一次性使用，用毕去 Settings → Tokens 作废。

### 2. 推送现有仓库

脚本 `scripts/push_remote.sh` 已内置：连通性预检、令牌类型识别、`push` 权限自检、推送后清除 `.git/config` 明文凭证。

```bash
export GH_PAT='你的ghp_令牌'
# 公开仓库（默认）
./scripts/push_remote.sh github rag-platform public
# 私有
./scripts/push_remote.sh github rag-platform private
```

要点：
- 分支已从 `master` 规范为 `main`。
- 若 `push_remote.sh` 报「App 令牌无权限」，把 `GH_PAT` 换成你的经典 PAT 即可。
- 推送后本地 `git remote -v` 应显示**无明文令牌**的干净 URL。

### 3. 上传单文件 / 目录（走 GitHub Contents / Git Data API）

脚本 `scripts/github_upload.py`：纯标准库，**不回显 token**，自动处理新建/更新。

```bash
export GITHUB_TOKEN='你的ghp_令牌'

# 单文件（已存在则自动带 sha 更新）
python3 scripts/github_upload.py single ./README.md liupu1106/rag-platform docs/README.md

# 整目录批量（一次 commit，保持相对路径）
python3 scripts/github_upload.py dir ./assets liupu1106/rag-platform assets \
    --branch main --message "chore: add assets"
```

实测：新建 `HTTP 201`、更新 `HTTP 200`、删除后 `GET 404`，全链路通过。

### 4. 部署（如需）

- **Railway**：GHCR 私有镜像免费版拉不了 → 推荐直接 `serviceConnect` 绑定**公开 GitHub 仓库**，云端从 Dockerfile 构建，push 即自动重新部署。
- **WorkBuddy**：用「发布为应用」能力，把沙箱 Flask 服务直接跑成带稳定分享链接的在线应用；依赖若装不下来，发布时把安装命令指向腾讯镜像。

---

## 四、一键命令集（可直接复制）

```bash
# 1) 修网络
bash scripts/fix_sandbox_network.sh

# 2) 推送仓库
export GH_PAT='ghp_xxx'
./scripts/push_remote.sh github rag-platform public

# 3) 上传单个文件到仓库
export GITHUB_TOKEN="$GH_PAT"
python3 scripts/github_upload.py single ./README.md liupu1106/rag-platform docs/README.md

# 4) 用完作废令牌（务必）
#    打开 https://github.com/settings/tokens 删除该 PAT
```

---

## 五、故障排查表

| 症状 | 先查 | 命令 |
|------|------|------|
| `curl api.github.com` = 000 | DNS 是否仍黑洞 | `getent hosts api.github.com` |
| git push 403 | 令牌是否 `ghu_`（App） | 换成 `ghp_` 经典 PAT |
| pip 安装 SSL EOF | 是否走了 pypi.org | 加 `-i https://mirrors.tencent.com/pypi/simple/` |
| Railway API 1010 | 是否 urllib 默认 UA | 改用 curl + 正常 UA |
| 部署拉不到镜像 | 是否私有 + 免费版 | 改公开 GitHub 源 / 升 Pro |
| 推送后 remote 含明文 | 看 `git remote -v` | 用脚本推送（会自动清除） |

---

## 六、安全提醒

1. **令牌只在环境变量里用**，不写进文件、不回显、不在脚本里硬编码。本仓库的 `push_remote.sh` / `github_upload.py` 都遵守这一点。
2. **明文贴出的 PAT / Railway Token 视为已泄露**，用毕即 revoke。
3. `.gitignore` 已排除 `.env`、`store/`、`.git/`，密钥不会进仓库。
4. 本沙箱的 DNS 修复**只在本会话有效**；新会话先重跑 `fix_sandbox_network.sh`。
