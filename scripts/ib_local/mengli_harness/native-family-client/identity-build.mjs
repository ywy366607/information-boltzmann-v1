import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const web = path.resolve(root, '../web');
const DEFAULT_PACKAGE_ROOT = 'F:/16、杂货行账册/ledger-dsh/node_modules/@deepseek-ai';
const DEFAULT_OUTPUT_ROOT = path.join(root, 'generated/identity');
const PRODUCT = '梦漓 Umwelt';
function replace(source, before, after) {
  if (source.split(before).length !== 2) throw new Error(`Identity hook anchor changed: ${before.slice(0, 90)}`);
  return source.replace(before, after);
}
const dataURL = (type, text) => `data:${type};base64,${Buffer.from(text).toString('base64')}`;

/** Compose these two helpers with the family's existing official-source patches. */
export function patchIdentityConversation(source) {
  source = replace(source, 'children: t("hero.headline")', `children: ${JSON.stringify(PRODUCT)}`);
  return replace(source, 'className: HeroShell_module_css_default.previewBadge,\n\t\t\t\t\t\t\t\tchildren: t("hero.preview")',
    'hidden: true,\n                                "aria-hidden": true');
}
export function patchIdentityLayout(source) {
  return replace(source, 'function DocumentTitle({ useSessions, usePanelInfo, productTitle }) {',
    `function DocumentTitle({ useSessions, usePanelInfo, productTitle }) {\n            productTitle = ${JSON.stringify(PRODUCT)};`);
}
function nativeIntro() {
  const state = fs.readFileSync(path.join(root, 'life-core-state.mjs'), 'utf8').replaceAll('export function ', 'function ');
  return state + '\n' + fs.readFileSync(path.join(root, 'identity-assets/life-core.js'), 'utf8');
}
function nativeStyles() {
  return fs.readFileSync(path.join(root, 'identity-assets/scene.css'), 'utf8')
    + '\n' + fs.readFileSync(path.join(root, 'identity-assets/native-intro.css'), 'utf8');
}
function manifest(name, official) {
  return { name, version: official.version, private: true, type: 'module', main: 'lib/index.js',
    exports: { '.': './lib/index.js', './client': './lib/client.js', './package.json': './package.json' },
    license: official.license, dsh: { client: { platform: 'web', inject: [
      '@deepseek-ai/dsh-client-ui-renderer', '@deepseek-ai/dsh-client-ui-sidebar', '@deepseek-ai/dsh-client-ui-conversation',
    ] } } };
}
/** Synchronous build. All written output is project-owned; upstream is read-only. */
export function buildIdentity({ packageRoot = DEFAULT_PACKAGE_ROOT, outputRoot = DEFAULT_OUTPUT_ROOT } = {}) {
  outputRoot = path.resolve(outputRoot);
  const from = path.join(packageRoot, 'dsh-client-ui-brand-official');
  const official = JSON.parse(fs.readFileSync(path.join(from, 'package.json'), 'utf8'));
  const mark = dataURL('image/svg+xml', fs.readFileSync(path.join(web, 'mengli-mark.svg'), 'utf8'));
  const liGlyphBrand = dataURL('image/svg+xml', fs.readFileSync(path.join(root, 'identity-assets/mengli-li-glyph.svg'), 'utf8'));
  let brand = fs.readFileSync(path.join(from, 'lib/client.js'), 'utf8');
  const markFunction = brand.match(/function OfficialBrandMark\(\{ size \}\) \{\s*return \(0, react_jsx_runtime\.jsx\)\(_deepseek_ai_dsh_client_ui_primitives\.FishLogo, \{ size \}\);\s*\}/)?.[0];
  if (!markFunction) throw new Error('Official brand mark hook changed');
  brand = replace(brand, markFunction, `function OfficialBrandMark({ size, className }) {
        return (0, react_jsx_runtime.jsx)("img", { src: ${JSON.stringify(mark)}, alt: "", width: size, height: size, className: "mengli-identity-mark " + (className || ""), style: { display: "block", objectFit: "contain" } });
      }`);
  const nameFunction = brand.match(/function OfficialBrandName\(\) \{[\s\S]*?\n\t\t\}/)?.[0];
  if (!nameFunction) throw new Error('Official brand name hook changed');
  brand = replace(brand, nameFunction, `function OfficialBrandName() {
        return (0, react_jsx_runtime.jsxs)("span", {
          style: {
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            whiteSpace: "nowrap",
            lineHeight: "24px",
            color: "inherit"
          },
          className: "mengli-brand-title",
          children: [
            (0, react_jsx_runtime.jsxs)("span", {
              style: {
                display: "inline-flex",
                alignItems: "center",
                fontFamily: "'Noto Serif SC', 'Songti SC', SimSun, serif",
                fontWeight: 400,
                fontSize: "17px",
                letterSpacing: "0.08em",
                lineHeight: 1
              },
              children: [
                "梦",
                (0, react_jsx_runtime.jsx)("img", {
                  src: ${JSON.stringify(liGlyphBrand)},
                  alt: "漓",
                  className: "mengli-li-brand-glyph",
                  style: {
                    width: "1.02em",
                    height: "1.02em",
                    display: "inline-block",
                    verticalAlign: "-0.12em",
                    marginLeft: "0.06em"
                  }
                })
              ]
            }),
            (0, react_jsx_runtime.jsx)("span", {
              style: {
                fontFamily: "ui-monospace, 'Cascadia Code', monospace",
                fontWeight: 400,
                fontSize: "12.5px",
                letterSpacing: "0.08em",
                opacity: 0.82
              },
              children: "Umwelt"
            })
          ]
        });
      }`);
  brand = replace(brand, '\n\t\tfunction apply(ctx) {', `\n\t\tfunction apply(ctx) {
      ctx.slots.inject("conversation.hero.brand.mark", () => ctx.slots.register({ name: "conversation.hero.brand.mark", id: "mengli-brand" }, OfficialBrandMark));`);
  const brandRoot = path.join(outputRoot, 'official/ui-brand-official');
  const identityRoot = path.join(outputRoot, 'client');
  for (const [directory, name, client] of [
    [brandRoot, '@deepseek-ai/dsh-client-ui-brand-official', brand],
    [identityRoot, '@mengli/native-identity', null],
  ]) {
    fs.mkdirSync(path.join(directory, 'lib'), { recursive: true });
    fs.writeFileSync(path.join(directory, 'package.json'), JSON.stringify(manifest(name, official), null, 2) + '\n');
    fs.writeFileSync(path.join(directory, 'lib/index.js'), 'export function apply() {}\n');
    fs.copyFileSync(path.join(from, 'LICENSE'), path.join(directory, 'LICENSE'));
    if (client) fs.writeFileSync(path.join(directory, 'lib/client.js'), client);
  }
  const notice = fs.readFileSync(path.join(web, 'THIRD_PARTY_NOTICES.md'), 'utf8');
  fs.writeFileSync(path.join(identityRoot, 'THIRD_PARTY_NOTICES.txt'), notice);
  const pwa = dataURL('application/manifest+json', JSON.stringify({ name: PRODUCT, short_name: '梦漓', start_url: '/', display: 'standalone',
    icons: [{ src: mark, sizes: 'any', type: 'image/svg+xml', purpose: 'any' }] }));
  const client = fs.readFileSync(path.join(root, 'identity-client.js'), 'utf8');
  const character = dataURL('image/png', fs.readFileSync(path.join(root, 'identity-assets/mengli-pointing.png')));
  const liGlyph = dataURL('image/svg+xml', fs.readFileSync(path.join(root, 'identity-assets/mengli-li-glyph.svg')));
  const nebula = dataURL('image/png', fs.readFileSync(path.join(root, 'identity-assets/mengli-nebula.png')));
  const source = `window.__ModuleLoader__.load({id:'@mengli/native-identity',factory(require){\n/* ${notice.replaceAll('*/', '* /')} */\nconst MENGLI_MARK=${JSON.stringify(mark)};\nconst MENGLI_MANIFEST=${JSON.stringify(pwa)};\nconst IDENTITY_STYLES=${JSON.stringify(nativeStyles())};\n${nativeIntro(mark)}\n${client}\nreturn { apply, inject };\n}});\n`;
  fs.writeFileSync(path.join(identityRoot, 'lib/client.js'), source.replace('const MENGLI_MARK=', `const MENGLI_LOGIN_PAGE=false;\nconst MENGLI_LI_GLYPH=${JSON.stringify(liGlyph)};\nconst MENGLI_CHARACTER=${JSON.stringify(character)};\nconst MENGLI_NEBULA=${JSON.stringify(nebula)};\nconst MENGLI_MARK=`));
  return { outputRoot, brandRoot, identityRoot, brandEntry: path.join(brandRoot, 'lib/index.js'), identityEntry: path.join(identityRoot, 'lib/index.js') };
}
/** Overlay after base bundle rows. Cordis name is a guard, never a replacement. */
export function identityPatchRows(options = {}) {
  const built = buildIdentity(options);
  return [{ id: 'ui-brand-official', disabled: true }, { insert: [
    { id: 'mengli-native-brand', name: built.brandEntry },
    { id: 'mengli-native-identity', name: built.identityEntry },
  ] }];
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(buildIdentity({ packageRoot: process.env.MENGLI_DSH_PACKAGE_ROOT || DEFAULT_PACKAGE_ROOT })));
}
