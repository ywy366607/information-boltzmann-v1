# 薄棉中筒袜上线 · 2026-11-22

用户从袜长及材质预览中选择薄棉中筒袜并授权替换。采用平整细织白棉袜，保留原人物轮廓及动画坐标。

成品：scripts/ib_local/mengli_harness/native-family-client/identity-assets/mengli-pointing-thin-midcalf.png，同步到mengli-pointing.png及mengli-pointing-rig-atlas.png。原棉袜图集保留在mengli-pointing-plain-socks.png。

生成工具：内置imagegen。输入1为原1024×3072双层动画图集；输入2为用户选择的中筒/长筒对照图，取左侧材质。生成原件：C:/Users/85750/.codex/generated_images/01a16868-701e-7813-8650-1e90545354c4/exec-45aaccbb-c370-46b1-88a2-52cedf1fc5d6.png。

合成：scratch/mengli-thin-cotton-midcalf-final.ffscript。生成图缩放回原尺寸、对齐后，仅将三处袜面合回原图集，完整恢复原alpha。最终图集1024×3072，改变29141像素，三处袜区外变化0，alpha变化0。构建及生成JS语法核对通过。

实际提示词：
Use case: precise-object-edit. IMAGE 1 is the EDIT TARGET: a 1024x3072 transparent animation texture atlas, two vertically stacked 1024x1536 panels. IMAGE 2 is MATERIAL REFERENCE ONLY: use the LEFT character's thin smooth white mid-calf cotton socks. Edit ONLY the white sock fabric in image 1: upper panel TWO socks, lower panel ONE sock. Replace thick coarse vertical ribbed fabric and bulky cuffs with smooth lightweight opaque white cotton, fine flat knit, thin flat cuff, very few soft creases, naturally fitted. Preserve existing cuff height (mid-calf), ankle silhouettes, leg proportions, exact positions, shoe shapes and every detail outside socks. Preserve original layout precisely: upper panel two legs and shoes, lower panel deliberately only one lower leg/shoe; this must remain exactly aligned. Do NOT copy the second image's character, text, layout or background. No logos, no sheer fabric. Preserve transparency including original translucent edges and halo. Output the same TWO-PANEL tall atlas, NOT a single character or comparison sheet.
