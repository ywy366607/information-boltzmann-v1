// Browser-only identity integration. SDK messages and input stay with native DSH.
const inject = ['slots'];
function apply(ctx) {
  const style = document.createElement('style');
  style.id = 'mengli-intro-styles';
  style.textContent = IDENTITY_STYLES;
  document.head.appendChild(style);
  document.body.classList.add('mengli-scene-native');

  // Metadata belongs to this product, not to the mother's real model provider.
  const links = [];
  for (const [rel, href, type] of [
    ['icon', MENGLI_MARK, 'image/svg+xml'],
    ['apple-touch-icon', MENGLI_MARK, 'image/svg+xml'],
    ['manifest', MENGLI_MANIFEST, 'application/manifest+json'],
  ]) {
    const link = document.createElement('link');
    link.rel = rel; link.href = href; link.type = type;
    document.querySelectorAll(`link[rel="${rel}"]`).forEach(old => old.remove());
    document.head.appendChild(link); links.push(link);
  }
  installMengliIntro();

  const bridge = window.__MENGLI_NATIVE_FAMILY__;
  let lastCard;
  let hasPushedCard = false;
  function fromFamily() {
    const state = bridge?.controller?.getSnapshot()?.state;
    const card = state?.life_card_data ?? state?.events?.at(-1)?.life_card_data ?? state?.latest?.life_card_data;
    if (card && card !== lastCard) {
      lastCard = card; hasPushedCard = true;
      window.MengliIntro?.updateLife(card, { committed: true });
    }
  }
  fromFamily();
  const unsubscribe = bridge?.subscribe(fromFamily);
  const fromTelemetry = event => {
    const card = event.detail?.card;
    window.MengliIntro?.updateLife(card, { committed: Boolean(card?.source?.sha256) });
  };
  window.addEventListener('mengli:life-state', fromTelemetry);
  // At most one initial read per tab. Refresh uses the latest pushed state instead.
  const readKey = 'mengli:life:read:v1';
  let read = false;
  try { read = sessionStorage.getItem(readKey) === 'yes'; sessionStorage.setItem(readKey, 'yes'); } catch {}
  let disposed = false;
  if (!read && !hasPushedCard) {
    fetch('/api/family/life', { credentials: 'same-origin' })
      .then(response => response.ok ? response.json() : null)
      .then(value => {
        if (!disposed && !hasPushedCard && value?.card) window.MengliIntro?.updateLife(value.card);
      }).catch(() => {});
  }
  ctx.effect(() => () => {
    disposed = true; unsubscribe?.(); window.removeEventListener('mengli:life-state', fromTelemetry);
    window.MengliIntro?.dispose?.();
    document.body.classList.remove('mengli-scene-native');
    style.remove(); links.forEach(link => link.remove());
  });
}
