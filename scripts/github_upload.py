#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用 GitHub REST API 上传文件/目录到仓库。

特点
----
- 纯标准库实现（urllib），无需 pip install 任何依赖。
- 支持两种模式：
    single  上传单个文件（走 Contents API，自动处理「新建 / 更新」）。
    dir     批量上传整个目录（走 Git Data API 一次性提交，效率更高）。
- 认证：环境变量 GITHUB_TOKEN（推荐，不回显），或 --token 传入。
- 仅做最小必要请求；失败时打印清晰的错误信息与 HTTP 状态码。

注意：脚本本身不会打印 token。请在安全环境使用，用毕及时 revoke。

示例
----
  # 上传单个文件
  export GITHUB_TOKEN=ghp_xxx
  python github_upload.py single ./README.md liupu1106/rag-platform docs/README.md

  # 批量上传整个目录（保持相对路径结构）
  python github_upload.py dir ./assets liupu1106/rag-platform assets \\
      --branch main --message "chore: add assets"
"""
import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.github.com"
UA = "rag-github-upload/1.0 (+https://github.com/liupu1106/rag-platform)"


def _api(method, path, token, body=None):
    """发起一个 GitHub API 请求，返回 (status, json_or_text)。"""
    url = API + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", UA)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw)
            except Exception:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw
    except Exception as e:  # DNS / TLS / 网络
        return None, {"message": f"网络错误: {e}"}


def _require_token(token):
    if not token:
        sys.exit("缺少 token：请 export GITHUB_TOKEN=... 或加 --token")
    return token


def upload_single(repo, local_path, remote_path, token, branch, message, author):
    """用 Contents API 上传/更新单个文件。"""
    token = _require_token(token)
    if not os.path.isfile(local_path):
        sys.exit(f"本地文件不存在: {local_path}")
    with open(local_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")

    # 先看文件是否已存在，存在则需要提供其 sha 才能更新
    _, existing = _api("GET", f"/repos/{repo}/contents/{remote_path}?ref={branch}", token)
    sha = existing.get("sha") if isinstance(existing, dict) else None
    action = "更新" if sha else "新建"

    payload = {
        "message": message or f"{action} {remote_path}",
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    if author:
        payload["committer"] = {"name": author, "email": f"{author}@users.noreply.github.com"}

    status, resp = _api("PUT", f"/repos/{repo}/contents/{remote_path}", token, payload)
    if status in (200, 201) and isinstance(resp, dict):
        html = (resp.get("content") or {}).get("html_url", "")
        print(f"[{action}] {remote_path}  ->  HTTP {status}")
        print(f"     {html}")
        return True
    print(f"[失败] {remote_path}  ->  HTTP {status}")
    print("     ", resp if isinstance(resp, (dict, str)) else resp)
    return False


def upload_dir(repo, local_dir, remote_prefix, token, branch, message, author):
    """用 Git Data API 把整个目录作为一次 commit 批量上传。"""
    token = _require_token(token)
    if not os.path.isdir(local_dir):
        sys.exit(f"本地目录不存在: {local_dir}")

    # 1) 取当前分支头指针
    s, ref = _api("GET", f"/repos/{repo}/git/refs/heads/{branch}", token)
    if s != 200 or not isinstance(ref, dict):
        print(f"[失败] 无法读取分支 {branch} 的头指针 -> HTTP {s}")
        print("     ", ref)
        return False
    head_sha = ref["object"]["sha"]

    # 2) 取基准树（作为新树的 base_tree）
    s, commit = _api("GET", f"/repos/{repo}/git/commits/{head_sha}", token)
    base_tree_sha = commit.get("tree", {}).get("sha") if isinstance(commit, dict) else None

    # 3) 为每个文件创建 blob
    blobs = []
    for root, _, files in os.walk(local_dir):
        for name in files:
            lp = os.path.join(root, name)
            with open(lp, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            s, blob = _api("POST", f"/repos/{repo}/git/blobs", token,
                           {"content": b64, "encoding": "base64"})
            if s != 201 or not isinstance(blob, dict):
                print(f"[失败] 创建 blob 失败: {lp} -> HTTP {s}: {blob}")
                return False
            rel = os.path.relpath(lp, local_dir).replace(os.sep, "/")
            remote_path = f"{remote_prefix.rstrip('/')}/{rel}" if remote_prefix else rel
            blobs.append((remote_path, blob["sha"]))

    # 4) 创建新树
    tree = [{"path": p, "mode": "100644", "type": "blob", "sha": sha} for p, sha in blobs]
    s, new_tree = _api("POST", f"/repos/{repo}/git/trees", token,
                       {"base_tree": base_tree_sha, "tree": tree})
    if s != 201 or not isinstance(new_tree, dict):
        print(f"[失败] 创建 tree 失败 -> HTTP {s}: {new_tree}")
        return False

    # 5) 创建 commit
    commit_payload = {"message": message or f"batch upload {len(blobs)} files",
                      "tree": new_tree["sha"], "parents": [head_sha]}
    if author:
        commit_payload["committer"] = {"name": author, "email": f"{author}@users.noreply.github.com"}
    s, new_commit = _api("POST", f"/repos/{repo}/git/commits", token, commit_payload)
    if s != 201 or not isinstance(new_commit, dict):
        print(f"[失败] 创建 commit 失败 -> HTTP {s}: {new_commit}")
        return False

    # 6) 更新分支引用
    s, _ = _api("PATCH", f"/repos/{repo}/git/refs/heads/{branch}", token,
                {"sha": new_commit["sha"], "force": False})
    if s == 200:
        print(f"[成功] 批量上传 {len(blobs)} 个文件到 {repo}@{branch} -> HTTP {s}")
        print(f"     commit: {new_commit.get('html_url', '')}")
        return True
    print(f"[失败] 更新分支引用失败 -> HTTP {s}")
    return False


def main():
    ap = argparse.ArgumentParser(description="调用 GitHub API 上传文件/目录")
    ap.add_argument("mode", choices=["single", "dir"], help="single=单文件；dir=整目录")
    ap.add_argument("local", help="本地文件或目录路径")
    ap.add_argument("repo", nargs="?", default="liupu1106/rag-platform",
                    help="目标仓库 owner/name（默认 liupu1106/rag-platform）")
    ap.add_argument("remote", help="远端路径：single 为文件全路径；dir 为目录前缀")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="")
    ap.add_argument("--author", default="rag-bot")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""),
                    help="GitHub token（也可走 GITHUB_TOKEN 环境变量）")
    args = ap.parse_args()

    if args.mode == "single":
        ok = upload_single(args.repo, args.local, args.remote, args.token,
                           args.branch, args.message, args.author)
    else:
        ok = upload_dir(args.repo, args.local, args.remote, args.token,
                        args.branch, args.message, args.author)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
