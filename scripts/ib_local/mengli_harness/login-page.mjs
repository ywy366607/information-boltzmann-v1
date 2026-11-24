import fs from 'node:fs';

const assets = new URL('./native-family-client/identity-assets/', import.meta.url);
const client = new URL('./native-family-client/', import.meta.url);
const read = file => fs.readFileSync(new URL(file, assets), 'utf8');
const uri = (type, file) => `data:${type};base64,${fs.readFileSync(new URL(file, assets)).toString('base64')}`;

// Only numerical display readings reach the anonymous artwork. The full card,
// checkpoint paths, history and account remain behind their original routes.
export function entranceReadings(card) {
  if (card?.schema !== 1 || !Number.isSafeInteger(card.source?.cycle) || card.source.cycle < 0) return null;
  const number = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
  const momentum = card.latest?.momentum_norms;
  const frequency = card.dynamics?.frequencies ?? card.dynamics?.frequency;
  if (!Array.isArray(momentum) || !Array.isArray(frequency)) return null;
  return { schema: 1, source: { cycle: card.source.cycle },
    age: { byte_events: Number.isSafeInteger(card.age?.byte_events) && card.age.byte_events >= 0 ? card.age.byte_events : null,
      born_at: typeof card.age?.born_at === 'string' && Number.isFinite(Date.parse(card.age.born_at)) ? card.age.born_at : null },
    latest: { momentum_norms: momentum.slice(0, 256).map(number) },
    dynamics: { frequency: frequency.slice(0, 256).map(row =>
      Array.isArray(row) ? row.slice(0, 256).map(number) : number(row)) } };
}
export function loginPage(card = null) {
  const definitions = `const MENGLI_LOGIN_PAGE=true;
const MENGLI_MARK='';
const MENGLI_LI_GLYPH=${JSON.stringify(uri('image/svg+xml', 'mengli-li-glyph.svg'))};
const MENGLI_CHARACTER=${JSON.stringify(uri('image/png', 'mengli-pointing.png'))};
const MENGLI_NEBULA=${JSON.stringify(uri('image/png', 'mengli-nebula.png'))};`;
  const state = fs.readFileSync(new URL('life-core-state.mjs', client), 'utf8').replaceAll('export function ', 'function ');
  const poem = fs.readFileSync(new URL('birth-poem.mjs', client), 'utf8').replaceAll('export ', '');
  const script = definitions + '\n' + state + '\n' + poem + '\n' + read('nebula-renderer.js') + '\n' + read('life-core.js');
  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><title>梦漓</title>
<style>
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:#000;color:#ebe8f3;font-family:system-ui,"Microsoft YaHei",sans-serif}
${read('native-intro.css')}
#login-fallback{position:absolute;left:50%;top:52%;transform:translate(-50%,-50%);width:min(280px,80vw)}
#login-fallback h1{font-size:26px;text-align:center;font-weight:450}
</style></head><body>
<main id="login-fallback"><h1>梦漓</h1><form class="ml-entry-form" method="post" action="/auth/login"><label for="fallback-key">本地密钥</label><input id="fallback-key" name="key" type="password" autocomplete="current-password" required maxlength="512"><button type="submit" class="ml-entry-submit">进入</button></form></main>
<script>${script.replaceAll('</script', '<\\/script')}
installMengliIntro(); window.MengliIntro.updateLife(${JSON.stringify(entranceReadings(card))});
document.getElementById('login-fallback')?.remove();
</script></body></html>`;
}
