/* Display-only tracing of sampled energy currents and tensor direction lines. */
window.MediumFlow = (() => {
  const wrap = x => ((x % 1) + 1) % 1;
  const norm = v => Math.hypot(...v);
  function sample(vectors, shape, p, reference=null) {
    const q=p.map((x,k)=>wrap(x)*shape[k]),base=q.map(Math.floor),t=q.map((x,k)=>x-base[k]);
    const out=[0,0,0];
    for(let a=0;a<2;a++)for(let b=0;b<2;b++)for(let c=0;c<2;c++) {
      const bits=[a,b,c],i=bits.map((v,k)=>(base[k]+v)%shape[k]);
      const weight=bits.reduce((w,v,k)=>w*(v?t[k]:1-t[k]),1);
      const value=vectors[(i[0]*shape[1]+i[1])*shape[2]+i[2]];
      const sign=reference&&value.reduce((s,x,k)=>s+x*reference[k],0)<0?-1:1;
      for(let k=0;k<3;k++)out[k]+=weight*value[k]*sign;
    }
    return out;
  }
  function directionLines(snapshot) {
    const vectors=snapshot.material_principal_axis,values=snapshot.material_tensor_eigenvalues;
    if(!vectors||!values)return [];
    const lines=[],step=.32/Math.max(...snapshot.shape),stride=Math.max(1,Math.floor(vectors.length/48));
    for(let seed=0;seed<vectors.length;seed+=stride) {
      const e=values[seed];if(e[2]-e[0]<1e-6*Math.max(1,e[2]))continue;
      const initial=snapshot.coordinates[seed],parts=[];
      for(const sign of [-1,1]) {
        let p=initial.slice(),previous=vectors[seed].map(x=>sign*x),points=[p];
        for(let n=0;n<32;n++) {
          // Eigenvectors are unoriented. Align corner signs before interpolation.
          let v=sample(vectors,snapshot.shape,p,previous);
          const size=norm(v);if(!size)break;
          const next=p.map((x,k)=>x+step*v[k]/size);
          if(next.some(x=>x<0||x>1))break; // split at the periodic display seam
          points.push(next);previous=v;p=next;
        }
        parts.push(points);
      }
      lines.push(parts[0].reverse().concat(parts[1].slice(1)));
    }
    return lines;
  }
  class Tracers {
    constructor(count=144) {
      this.particles=Array.from({length:count},(_,i)=>({p:[wrap(i*.61803398875),wrap(i*.41421356237+.2),wrap(i*.73205080757+.4)],trail:[]}));
      this.last=0;this.snapshot=null;this.scale=0;
    }
    set(snapshot) {
      this.snapshot=snapshot;
      const sizes=(snapshot.transport_energy_current||[]).map(norm).sort((a,b)=>a-b);
      this.scale=sizes[Math.floor(.9*(sizes.length-1))]||0;
    }
    draw(ctx,ts,project,w,h,visible) {
      const dt=Math.min(.05,Math.max(0,(ts-this.last)/1000));this.last=ts;
      const s=this.snapshot;if(!s?.transport_energy_current||!this.scale)return;
      if(Date.now()/1000-s.wall_time>60)return;
      ctx.strokeStyle='#8ce8e4aa';ctx.fillStyle='#d1fff3';ctx.lineWidth=1;
      for(const particle of this.particles) {
        const v=sample(s.transport_energy_current,s.shape,particle.p),length=norm(v);
        if(length<1e-12*this.scale){particle.trail=[];continue;}
        // Global display normalization preserves signs/directions; it is not
        // the physical particle velocity or a numerical time integrator.
        const gain=.14/Math.max(this.scale,length);
        const next=particle.p.map((x,k)=>wrap(x+dt*gain*v[k]));
        if(next.some((x,k)=>Math.abs(x-particle.p[k])>.5))particle.trail=[];
        particle.p=next;particle.trail.push(next.slice());if(particle.trail.length>9)particle.trail.shift();
        if(!visible(next))continue;
        ctx.beginPath();let started=false;
        for(const p of particle.trail){const q=project(p,w,h);if(started)ctx.lineTo(q[0],q[1]);else{ctx.moveTo(q[0],q[1]);started=true;}}
        ctx.stroke();const q=project(next,w,h);ctx.beginPath();ctx.arc(q[0],q[1],1.4,0,Math.PI*2);ctx.fill();
      }
    }
  }
  return {sample,directionLines,Tracers};
})();
