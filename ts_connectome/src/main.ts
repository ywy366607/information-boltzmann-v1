import { CausalConnectome } from './CausalConnectome';
import { CGBNode, CGBEdge, CGBSector } from './types';

// Load or fetch payload
async function init() {
  const container = document.getElementById('canvas-container')!;
  
  // Data can be fetched from static json or window global
  const nodes: CGBNode[] = (window as any).NODES_DATA || [];
  const edges: CGBEdge[] = (window as any).EDGES_DATA || [];
  const sectors: CGBSector[] = (window as any).SECTORS_DATA || [];

  const app = new CausalConnectome(container, nodes, edges, sectors);

  // Wire Sliders
  const expSlider = document.getElementById('exposure-slider') as HTMLInputElement;
  if (expSlider) {
    expSlider.addEventListener('input', e => {
      app.setExposure(parseFloat((e.target as HTMLInputElement).value) / 100);
    });
  }

  const bloomSlider = document.getElementById('bloom-slider') as HTMLInputElement;
  if (bloomSlider) {
    bloomSlider.addEventListener('input', e => {
      app.setBloom(parseFloat((e.target as HTMLInputElement).value) / 100);
    });
  }

  const halationSlider = document.getElementById('halation-slider') as HTMLInputElement;
  if (halationSlider) {
    halationSlider.addEventListener('input', e => {
      app.setHalation(parseFloat((e.target as HTMLInputElement).value) / 100);
    });
  }

  const grainSlider = document.getElementById('grain-slider') as HTMLInputElement;
  if (grainSlider) {
    grainSlider.addEventListener('input', e => {
      app.setGrain(parseFloat((e.target as HTMLInputElement).value) / 1000);
    });
  }

  const morphSlider = document.getElementById('morph-slider') as HTMLInputElement;
  if (morphSlider) {
    morphSlider.addEventListener('input', e => {
      app.setMorph(parseFloat((e.target as HTMLInputElement).value) / 100);
    });
  }

  // Zen Mode
  let isZen = false;
  const zenBtn = document.getElementById('zen-btn');
  const hudElements = document.querySelectorAll('.hud');
  if (zenBtn) {
    const toggle = () => {
      isZen = !isZen;
      hudElements.forEach(el => el.classList.toggle('zen-hidden', isZen));
      zenBtn.innerText = isZen ? '✦ 显示控制' : '✦ 纯净背景';
    };
    zenBtn.onclick = toggle;
    window.addEventListener('keydown', e => {
      if (e.key === 'h' || e.key === 'H' || e.key === 'Escape') toggle();
    });
  }
}

window.addEventListener('DOMContentLoaded', init);
