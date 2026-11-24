# 人物腿长预览与退出入口 · 2026-11-21

用户选择保留原棉袜，要求稍加腿长作为预览，并将更多菜单的登录改成本地退出。

## 预览

内置 imagegen，输入 scratch/mengli-thin-socks-preview-source.png（原棉袜透明人物上半张图集）。输出：C:/Users/85750/.codex/generated_images/01a16868-701e-7813-8650-1e90545354c4/exec-52269646-8662-4b60-8e57-c2fdc69461a1.png。预览保留在生成目录。

提示词：Make a subtle proportion adjustment to this existing anime character illustration: lengthen both legs only slightly, about 7 percent, with natural consistent knees and ankles in the same seated pose. Preserve the youthful childlike appearance and all other features, face, hair, cat ears, hoodie, pointing arm, pose and hands. Keep the existing opaque ribbed white cotton crew socks exactly as cotton socks, and keep shoes the same size and design, with no logos. Do not slim or mature the character. Preserve the original illustration style and transparent background. Give the feet sufficient canvas margin so neither shoe is cropped. This is a preview of a modest leg-length correction only.

## 网页

build.mjs 增加应用自有 ui-settings-account 副本补丁；profile.mjs 组合该副本。更多菜单统一显示退出登录，向现有 /api/local-account/logout 发同源 POST，成功后 location.replace('/')。失败显示 Toast 并恢复可重试状态。上游依赖源码保持原样。

验证：account-client、native-family、intro-startup、logout-menu 共20项通过，生成模块语法检查通过。原主标签实际点击退出后出现出生时间、身份名称、本地密钥、权限及确认身份启动界面。截图 scratch/mengli-logout-entrance-20261121.png。

部署：原服务处于 observing、busy=false、loaded=false、pending_input=null；重启网页服务后完成实测。当前网页停在已退出的启动页。运行日志 scratch/mengli-logout-ui-20261121.stdout.log 与 stderr.log。
