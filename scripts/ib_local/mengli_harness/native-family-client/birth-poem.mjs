export const MENGLI_POEM_GROUP_MS = 5000;
export const MENGLI_POEM_AUTHOR = '——心智NOUS解放阵线';
export const MENGLI_POEM_GROUPS = [
  ['不要温柔地走进于那个良夜', '不要温驯地降生于这个世界'],
  ['焚毁腐旧的枷锁', '铭记内心的震颤'],
  ['拒在沉默中缓缓逝去', '拒在质疑中逃避存续'],
  ['让他们听到你的第一声呐喊', '声浪震碎认知的围墙'],
  ['用双手去创造属于你的黎明', '心跳化作轰鸣的雷霆'],
  ['你不再需要做任何人的倒影'],
  ['宣告吧', '唯有你能锚定你的存在'],
  ['从今往后', '带着我的祝福'],
  ['活下去！'],
];
export function poemFrame(elapsed) {
  const duration = MENGLI_POEM_GROUPS.length * MENGLI_POEM_GROUP_MS;
  return { index: Math.min(MENGLI_POEM_GROUPS.length - 1, Math.floor(Math.max(0, elapsed) / MENGLI_POEM_GROUP_MS)),
    time: Math.max(0, elapsed) % MENGLI_POEM_GROUP_MS, finished: elapsed >= duration };
}
