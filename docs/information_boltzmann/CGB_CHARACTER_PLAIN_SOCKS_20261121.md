# 人物袜子标志去除

用户要求去掉两只袜子上的耐克标志。使用内置image_gen编辑当前动画atlas，再将生成结果中的三块袜面局部修补取回原始atlas（上层两块，下层一块），保持其余像素及原alpha逐像素一致。活动资源`mengli-pointing.png`和`mengli-pointing-rig-atlas.png`同步更新；保留`mengli-pointing-plain-socks.png`为此次最终编辑版。

生成原件：`C:/Users/85750/.codex/generated_images/01a16868-701e-7813-8650-1e90545354c4/exec-5ea8b33c-f726-4c2f-a8b9-5812e6ae2dfe.png`。

局部合成：`scratch/mengli-socks-patch.ffscript`，三个区域分别为(x,y,w,h)：(628,1180,70,64)、(772,1147,65,59)、(772,2683,65,59)，6像素边缘过渡，合成后还原原atlas的完整alpha。

检查：最终1024×3072 RGBA，改变11083像素；区域之外像素变化0，整张alpha变化0。构建及生成JS语法检查通过。编辑前可恢复副本`scratch/mengli-pointing-before-plain-socks-20261121.png`。

## 实际编辑提示词

Use case: precise-object-edit. Edit target: supplied 1024x3072 RGBA animation texture atlas containing TWO vertically stacked 1024x1536 panels of the same seated anime boy. Primary request: erase ONLY the blue Nike swoosh logos on the white socks: TWO logos in the upper panel, ONE logo in the lower panel. Replace each tiny blue logo area with matching plain white/light lavender ribbed sock fabric, continuing the existing vertical knit texture, shading and folds. This is an exact texture repair, not a redraw. Keep the original canvas dimensions 1024x3072, the two-panel layout, all character positions and proportions, face, hair, pointing hand, limbs, shoes and every other detail exactly aligned and unchanged. Preserve original transparency/alpha, including soft translucent edges and halo. Do not change any pixel outside the socks. Lower panel deliberately has only one lower leg and shoe; preserve that atlas layout. Output the repaired two-panel transparent PNG atlas.
