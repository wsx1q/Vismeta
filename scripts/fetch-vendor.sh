#!/usr/bin/env bash
# 把 GPU 版构建要用、版本固定的第三方源码下载到 vendor/，之后构建镜像直接用本地的，不再访问 GitHub。
# 已经有且版本一致就跳过；要升级时改下面的版本号再运行一次。
#   ./scripts/fetch-vendor.sh
# 国内网络：先 export https_proxy=http://127.0.0.1:7897，或设 GITHUB_PREFIX=https://ghfast.top/
set -euo pipefail
cd "$(dirname "$0")/.."

SAM3_REF=2345a4ad109ac29c569da749c91d84f10dc08c40
LLAMA_CPP_REF=4413a192410fad1f054e9216114087a3e01527a0
GITHUB_PREFIX="${GITHUB_PREFIX:-}"

fetch() {  # fetch 目录名 GitHub仓库 提交号
  local name="$1" repo="$2" ref="$3" dest="vendor/$1"
  if [ -f "$dest/.ref" ] && [ "$(cat "$dest/.ref")" = "$ref" ] && { [ "$name" != sam3 ] || { [ -f "$dest/.patch.sha256" ] && [ "$(cat "$dest/.patch.sha256")" = "$(sha256sum sam3.patch | cut -d ' ' -f1)" ]; }; }; then
    echo "vendor/$name 已是 ${ref:0:8}，跳过"
    return 1
  fi
  echo "下载 $repo@${ref:0:8} …"
  local tmp
  tmp="$(mktemp -d)"
  curl -fL --retry 3 "${GITHUB_PREFIX}https://codeload.github.com/$repo/tar.gz/$ref" | tar -xz -C "$tmp"
  rm -rf "$dest"
  mkdir -p vendor
  mv "$tmp"/* "$dest"
  rmdir "$tmp"
  echo "$ref" > "$dest/.ref"
}

if fetch sam3 facebookresearch/sam3 "$SAM3_REF"; then
  # 只留安装要用的；打上本项目的补丁
  (cd vendor/sam3 && rm -rf assets examples scripts test .github && patch -p1 --forward < ../../sam3.patch)
  sha256sum sam3.patch | cut -d ' ' -f1 > vendor/sam3/.patch.sha256
fi

if fetch llama.cpp ggml-org/llama.cpp "$LLAMA_CPP_REF"; then
  # 测试用的词表模型、文档、基准测试编译时用不到
  (cd vendor/llama.cpp && rm -rf models docs media benches .github)
fi

du -sh vendor/* 2>/dev/null
