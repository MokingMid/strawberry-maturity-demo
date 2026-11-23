"""通过 GitHub API 创建空仓库。

凭据来源：用 `git credential fill`（git 自身的 plumbing 命令）向已配置的
credential helper 索取 github.com 的凭据。**token 不会打印到任何输出**。

只做一件事：POST /user/repos 创建公开空仓库，不带任何初始化文件。
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request

OWNER = "MokingMid"
NAME = "strawberry-maturity-demo"
DESCRIPTION = "🍓 草莓成熟度检测 Demo：YOLO 检测 + ResNet18 三状态分类（成熟/半熟/未熟），含可视化网站"


def get_credential() -> tuple[str, str] | None:
    """向 git credential helper 索取 github.com 凭据，返回 (user, token)。"""
    try:
        proc = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None
    user = token = ""
    for line in proc.stdout.splitlines():
        if line.startswith("username="):
            user = line.split("=", 1)[1].strip()
        elif line.startswith("password="):
            token = line.split("=", 1)[1].strip()
    if not token:
        return None
    return user, token


def api(method: str, url: str, token: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "strawberry-demo-setup")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read().decode())


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cred = get_credential()
    if cred is None:
        print("✗ 未找到 github.com 的凭据，请先手动登录一次：")
        print("  cd E:\\临时\\github_repo")
        print("  git push -u origin main     # 浏览器登录后 Ctrl+C 取消即可")
        return 1
    user, token = cred
    print(f"✓ 找到凭据，用户名：{user}（token 不显示）")

    # 先看仓库是否已存在
    try:
        status, info = api("GET", f"https://api.github.com/repos/{OWNER}/{NAME}", token)
        print(f"! 仓库已存在：{info['html_url']}")
        print("  直接推送即可")
        return 0
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            print(f"✗ 查询失败：HTTP {exc.code}")
            return 1

    # 创建公开空仓库
    try:
        status, info = api(
            "POST", "https://api.github.com/user/repos", token,
            {
                "name": NAME,
                "description": DESCRIPTION,
                "private": False,
                "auto_init": False,
                "has_issues": True,
                "has_wiki": False,
                "license_template": "mit",
            },
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        print(f"✗ 创建失败：HTTP {exc.code}")
        try:
            print("  " + json.loads(detail).get("message", detail))
        except Exception:
            print("  " + detail[:300])
        return 1
    except Exception as exc:
        print(f"✗ 请求异常：{exc}")
        return 1

    print(f"✓ 仓库已创建：{info['html_url']}")
    print(f"  可见性：{info['visibility']} | 默认分支：{info.get('default_branch')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
