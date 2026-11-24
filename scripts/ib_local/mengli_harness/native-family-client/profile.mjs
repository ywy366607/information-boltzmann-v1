import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { ensureNativeFamilyBuild } from './build.mjs';
import { identityPatchRows } from './identity-build.mjs';
export { ensureNativeFamilyBuild } from './build.mjs';
const root = path.dirname(fileURLToPath(import.meta.url));

/** Overlay AFTER the base/web bundle rows. Module identities remain official. */
export function buildNativeFamilyPatch({ packageRoot } = {}) {
  ensureNativeFamilyBuild({ packageRoot });
  return [
    // In Cordis patches `name` is a MATCH GUARD, not a replacement field.
    { id: 'ui-layout', disabled: true },
    { id: 'ui-conversation', disabled: true },
    { id: 'ui-chat', disabled: true },
    { id: 'ui-renderer', disabled: true },
    { id: 'ui-settings-account', disabled: true },
    { insert: [
      { id: 'mengli-native-layout', name: path.join(root, 'official/ui-layout/lib/index.js') },
      { id: 'mengli-native-conversation', name: path.join(root, 'official/ui-conversation/lib/index.js') },
      { id: 'mengli-native-chat', name: path.join(root, 'official/ui-chat/lib/index.js') },
      { id: 'mengli-family-native', name: path.join(root, 'lib/index.js') },
      { id: 'mengli-native-renderer', name: path.join(root, 'official/ui-renderer/lib/index.js') },
      { id: 'mengli-native-account-menu', name: path.join(root, 'official/ui-settings-account/lib/index.js') },
    ] },
    ...identityPatchRows({ packageRoot }),
  ];
}
