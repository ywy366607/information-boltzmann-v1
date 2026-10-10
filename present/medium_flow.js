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
  function sampleCurrent(vectors,shape,p) {
    // J_a(i) is on the outgoing edge midpoint i + e_a/2, not at node i.
    return [0,1,2].map(axis=>{
      const q=p.map((x,k)=>wrap(x-(k===axis?.5/shape[k]:0))*shape[k]),base=q.map(Math.floor),t=q.map((x,k)=>x-base[k]);let out=0;
      for(let a=0;a<2;a++)for(let b=0;b<2;b++)for(let c=0;c<2;c++) {
        const i=(((base[0]+a)%shape[0])*shape[1]+(base[1]+b)%shape[1])*shape[2]+(base[2]+c)%shape[2];
        out+=vectors[i][axis]*(a?t[0]:1-t[0])*(b?t[1]:1-t[1])*(c?t[2]:1-t[2]);
      }
      return out;
    });
  }
  function currentLines(snapshot,{seeds,count=64,maxLength=1.5}={}) {
    const vectors=snapshot.transport_energy_current;if(!vectors)return [];
    const lines=[],step=.3/Math.max(...snapshot.shape),steps=Math.ceil(maxLength/(2*step));
    const scale=Math.max(...vectors.map(norm));if(!scale)return [];
    let seedList = [];
    if(seeds && seeds.length > 0) {
      const delta = 1.0 / Math.max(...snapshot.shape);
      const offsets = [
        [0,0,0],
        [delta,0,0],[-delta,0,0],[0,delta,0],[0,-delta,0],[0,0,delta],[0,0,-delta],
        [delta*.7,delta*.7,0],[-delta*.7,-delta*.7,0],[0,delta*.7,delta*.7],[0,-delta*.7,-delta*.7]
      ];
      for(const s of seeds) {
        for(const off of offsets) {
          seedList.push(s.map((x,k)=>wrap(x+off[k])));
        }
      }
    } else {
      seedList = Array.from({length:count},(_,i)=>[wrap(i*.61803398875),wrap(i*.41421356237+.2),wrap(i*.73205080757+.4)]);
    }
    for(let i=0;i<seedList.length;i++) {
      const initial=seedList[i];
      const segments=[];let length=0;
      for(const sign of [-1,1]) {
        let p=initial.slice(),part=[{p:p.slice(),speed:norm(sampleCurrent(vectors,snapshot.shape,p)),ratio:1,gap:1}];
        for(let k=0;k<steps;k++) {
          const v=sampleCurrent(vectors,snapshot.shape,p),size=norm(v);if(size<1e-8*scale)break;
          const mid=p.map((x,a)=>wrap(x+sign*.5*step*v[a]/size)),mv=sampleCurrent(vectors,snapshot.shape,mid),ms=norm(mv);
          if(ms<1e-8*scale||v.reduce((sum,x,a)=>sum+x*mv[a],0)<.5*size*ms)break;
          const unwrapped=p.map((x,a)=>x+sign*step*mv[a]/ms),next=unwrapped.map(wrap);
          if(unwrapped.some(x=>x<0||x>1)){if(part.length>1)segments.push(part);part=[]}
          part.push({p:next.slice(),speed:ms,ratio:1,gap:1});length+=step;p=next;
        }
        if(part.length>1)segments.push(part);
      }
      if(segments.length)lines.push({segments,length});
    }
    return lines;
  }
  function currentAlignment(snapshot) {
    const j=snapshot.transport_energy_current,e=snapshot.material_principal_axis;if(!j||!e)return null;
    let total=0,projected=0;
    for(let i=0;i<e.length;i++) {
      const value=sampleCurrent(j,snapshot.shape,snapshot.coordinates[i]),size=norm(value);if(!size)continue;
      const dot=value.reduce((sum,x,a)=>sum+x*e[i][a],0)/size;total+=size;projected+=size*dot*dot;
    }
    return total?projected/total:null;
  }
  function junctionSummary(snapshot) {
    const j=snapshot?.hopf_branch;
    if(j?.protocol!=='capacity_paid_persistent_flux_junction_v2'||
       j.capacity_metric!=='squared_coupling_norm'||!j.signed_fraction?.length)return null;
    const rows=[0,1,2].map(k=>{
      const full=j.full_row_capacity.reduce((sum,row)=>sum+row[k]**2,0);
      const used=j.junction_row_capacity.reduce((sum,row)=>sum+row[k]**2,0);
      const rates=j.rates.map(row=>row[k]);
      return {pair:j.pairs[k],owner:j.resource_owner_rows[k],mask:j.route_mask[k],
        paidShare:full?used/full:0,meanRate:rates.reduce((sum,x)=>sum+x,0)/rates.length,
        meanAbsKappa:j.signed_fraction.reduce((sum,row)=>sum+Math.abs(row[k]),0)/j.signed_fraction.length,
        maxAbsRate:Math.max(...rates.map(Math.abs)),physicalLength:j.physical_length[k]};
    });
    const full=j.full_row_capacity.flat().reduce((sum,x)=>sum+x*x,0);
    const used=j.junction_row_capacity.flat().reduce((sum,x)=>sum+x*x,0);
    return {rows,paidShare:full?used/full:0,maxAbsRate:Math.max(...rows.map(row=>row.maxAbsRate)),
      balance:j.capacity_balance_error,forest:j.forest,scope:j.scope};
  }
  function directionLines(snapshot) {
    return streamlines(snapshot,{count:48}).flatMap(line=>line.segments.map(part=>part.map(value=>value.p)));
  }
  // Interpolate the positive tensor, not signed eigenvectors. Jacobi gives
  // its unoriented principal axis; the top eigengap measures axis ambiguity.
  function principal(tensor, reference=null) {
    const a=tensor.map(row=>row.slice()),v=[[1,0,0],[0,1,0],[0,0,1]];
    for(let sweep=0;sweep<6;sweep++)for(const [p,q] of [[0,1],[0,2],[1,2]]) {
      if(Math.abs(a[p][q])<1e-12*Math.max(1,...a.map((row,i)=>Math.abs(row[i]))))continue;
      const theta=.5*Math.atan2(2*a[p][q],a[q][q]-a[p][p]),c=Math.cos(theta),s=Math.sin(theta);
      const ap=a[p][p],aq=a[q][q],off=a[p][q];
      a[p][p]=c*c*ap-2*s*c*off+s*s*aq;
      a[q][q]=s*s*ap+2*s*c*off+c*c*aq;a[p][q]=a[q][p]=0;
      for(let k=0;k<3;k++)if(k!==p&&k!==q){const x=a[k][p],y=a[k][q];a[k][p]=a[p][k]=c*x-s*y;a[k][q]=a[q][k]=s*x+c*y}
      for(let k=0;k<3;k++){const x=v[k][p],y=v[k][q];v[k][p]=c*x-s*y;v[k][q]=s*x+c*y}
    }
    const order=[0,1,2].sort((i,j)=>a[i][i]-a[j][j]),e=order.map(i=>Math.max(0,a[i][i]));
    let axis=v.map(row=>row[order[2]]);if(reference&&axis.reduce((sum,x,i)=>sum+x*reference[i],0)<0)axis=axis.map(x=>-x);
    return {axis,speed:Math.sqrt(e[2]),ratio:Math.sqrt(e[2]/Math.max(e[0],1e-20)),gap:(e[2]-e[1])/Math.max(e[2],1e-20)};
  }
  function tensorField(snapshot) {
    if(snapshot.effective_transport_factor)return snapshot.effective_transport_factor.map(b=>
      b.map(row=>b.map(other=>row.reduce((sum,x,k)=>sum+x*other[k],0))));
    // Compatibility for legacy snapshots: reconstruct all entries before tracing.
    const speed=snapshot.speed;
    if(!speed)return null;
    return speed.map((c,i)=>{const sh=snapshot.shear?.[i]||[0,0,0];
      const b=[[c[0],0,0],[c[1]*sh[0],c[1],0],[c[2]*sh[1],c[2]*sh[2],c[2]]];
      return b.map(row=>b.map(other=>row.reduce((sum,x,k)=>sum+x*other[k],0)))});
  }
  function sampleTensor(tensors,shape,p,reference=null) {
    const q=p.map((x,k)=>wrap(x)*shape[k]),base=q.map(Math.floor),t=q.map((x,k)=>x-base[k]),out=[[0,0,0],[0,0,0],[0,0,0]];
    for(let a=0;a<2;a++)for(let b=0;b<2;b++)for(let c=0;c<2;c++) {
      const bits=[a,b,c],i=bits.map((x,k)=>(base[k]+x)%shape[k]),w=bits.reduce((u,x,k)=>u*(x?t[k]:1-t[k]),1),value=tensors[(i[0]*shape[1]+i[1])*shape[2]+i[2]];
      for(let j=0;j<3;j++)for(let k=0;k<3;k++)out[j][k]+=w*value[j][k];
    }
    return principal(out,reference);
  }
  function streamlines(snapshot,{count=64,minGap=.05,maxLength=1.5,seeds=null}={}) {
    const tensors=tensorField(snapshot);if(!tensors)return [];
    const cutoff=Math.max(1e-8,minGap); // exactly degenerate axes have no preferred direction
    const lines=[],step=.3/Math.max(...snapshot.shape),steps=Math.ceil(maxLength/(2*step));
    for(let seed=0;seed<(seeds?seeds.length:count);seed++) {
      const initial=seeds?seeds[seed].map(wrap):[wrap((seed+.5)*.61803398875),wrap((seed+.5)*.41421356237+.2),wrap((seed+.5)*.73205080757+.4)];
      const first=sampleTensor(tensors,snapshot.shape,initial);if(first.gap<cutoff||first.speed<1e-12)continue;
      const parts=[];let length=0;
      for(const sign of [-1,1]) {
        let p=initial.slice(),previous=first.axis.map(x=>sign*x),part=[{p:p.slice(),...first}],segments=[];
        for(let k=0;k<steps;k++) {
          const current=sampleTensor(tensors,snapshot.shape,p,previous);if(current.gap<cutoff)break;
          const mid=p.map((x,j)=>wrap(x+.5*step*current.axis[j])),nextInfo=sampleTensor(tensors,snapshot.shape,mid,current.axis);
          if(nextInfo.gap<cutoff||nextInfo.axis.reduce((sum,x,j)=>sum+x*previous[j],0)<.5)break;
          const unwrapped=p.map((x,j)=>x+step*nextInfo.axis[j]),next=unwrapped.map(wrap);
          if(unwrapped.some(x=>x<0||x>1)){if(part.length>1)segments.push(part);part=[]}
          part.push({p:next.slice(),...nextInfo});length+=step;previous=nextInfo.axis;p=next;
        }
        if(part.length>1)segments.push(part);parts.push(...segments);
      }
      if(parts.length)lines.push({segments:parts,length});
    }
    return lines;
  }
  class Tracers {
    constructor(count=144) {
      this.particles=Array.from({length:count},(_,i)=>({p:[wrap(i*.61803398875),wrap(i*.41421356237+.2),wrap(i*.73205080757+.4)],trail:[]}));
      this.last=0;this.snapshot=null;this.scale=0;
    }
    set(snapshot) {
      if(this.snapshot&&this.snapshot.wall_time!==snapshot.wall_time)this.particles.forEach(p=>p.trail=[]);
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
        const v=sampleCurrent(s.transport_energy_current,s.shape,particle.p),length=norm(v);
        if(length<1e-12*this.scale){particle.trail=[];continue;}
        // Global display normalization preserves signs/directions; it is not
        // the physical particle velocity or a numerical time integrator.
        const gain=.14/Math.max(this.scale,length);
        const mid=particle.p.map((x,k)=>wrap(x+.5*dt*gain*v[k]));
        const vm=sampleCurrent(s.transport_energy_current,s.shape,mid),magnitude=norm(vm),mgain=.14/Math.max(this.scale,magnitude);
        const next=particle.p.map((x,k)=>wrap(x+dt*mgain*vm[k]));
        if(next.some((x,k)=>Math.abs(x-particle.p[k])>.5))particle.trail=[];
        particle.p=next;particle.trail.push(next.slice());if(particle.trail.length>9)particle.trail.shift();
        if(!visible(next))continue;
        ctx.beginPath();let started=false;
        for(const p of particle.trail){const q=project(p,w,h);if(started)ctx.lineTo(q[0],q[1]);else{ctx.moveTo(q[0],q[1]);started=true;}}
        ctx.stroke();const q=project(next,w,h);ctx.beginPath();ctx.arc(q[0],q[1],1.4,0,Math.PI*2);ctx.fill();
      }
    }
  }
  function vorticityField(snapshot) {
    const vectors = snapshot.transport_energy_current;
    if (!vectors) return null;
    const [nx, ny, nz] = snapshot.shape;
    const dx = 1 / nx, dy = 1 / ny, dz = 1 / nz;
    const getJ = (x, y, z) => {
      const idx = (((x % nx + nx) % nx) * ny + ((y % ny + ny) % ny)) * nz + ((z % nz + nz) % nz);
      return vectors[idx];
    };
    const out = new Float64Array(nx * ny * nz);
    for (let x = 0; x < nx; x++) {
      for (let y = 0; y < ny; y++) {
        for (let z = 0; z < nz; z++) {
          const idx = (x * ny + y) * nz + z;
          const jz_py = getJ(x, y + 1, z)[2], jz_my = getJ(x, y - 1, z)[2];
          const jy_pz = getJ(x, y, z + 1)[1], jy_mz = getJ(x, y, z - 1)[1];
          const wx = (jz_py - jz_my) / (2 * dy) - (jy_pz - jy_mz) / (2 * dz);

          const jx_pz = getJ(x, y, z + 1)[0], jx_mz = getJ(x, y, z - 1)[0];
          const jz_px = getJ(x + 1, y, z)[2], jz_mx = getJ(x - 1, y, z)[2];
          const wy = (jx_pz - jx_mz) / (2 * dz) - (jz_px - jz_mx) / (2 * dx);

          const jy_px = getJ(x + 1, y, z)[1], jy_mx = getJ(x - 1, y, z)[1];
          const jx_py = getJ(x, y + 1, z)[0], jx_my = getJ(x, y - 1, z)[0];
          const wz = (jy_px - jy_mx) / (2 * dx) - (jx_py - jx_my) / (2 * dy);

          out[idx] = Math.hypot(wx, wy, wz);
        }
      }
    }
    return Array.from(out);
  }
  return {sample,sampleCurrent,currentLines,currentAlignment,junctionSummary,directionLines,principal,sampleTensor,tensorField,streamlines,Tracers,vorticityField};
})();
