/* Read-only statistics on actual periodic training sites. No invented paths. */
window.MediumStructure = (() => {
  const palette=['#69edcb','#ffb36b','#b6a1ff','#f387cb','#9be27e','#79bcff','#ffe27c','#e4a092'];
  function quantile(values,q){const s=values.slice().sort((a,b)=>a-b),x=(s.length-1)*q,i=Math.floor(x);return s[i]+(s[Math.min(i+1,s.length-1)]-s[i])*(x-i)}
  function neighbors(i,shape){const z=i%shape[2],y=Math.floor(i/shape[2])%shape[1],x=Math.floor(i/(shape[1]*shape[2])),p=[x,y,z],out=[];for(let a=0;a<3;a++)for(const d of [-1,1]){const q=p.slice();q[a]=(q[a]+d+shape[a])%shape[a];out.push((q[0]*shape[1]+q[1])*shape[2]+q[2])}return out}
  function analyze(values,shape,top=20){
    if(!values.length)return null;
    const mean=values.reduce((a,b)=>a+b,0)/values.length,std=Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/values.length),threshold=quantile(values,1-top/100);
    const high=values.map(v=>v>threshold),visited=new Set(),groups=[];
    for(let i=0;i<values.length;i++){if(!high[i]||visited.has(i))continue;const group=[],todo=[i];visited.add(i);while(todo.length){const j=todo.pop();group.push(j);for(const k of neighbors(j,shape))if(high[k]&&!visited.has(k)){visited.add(k);todo.push(k)}}groups.push(group)}
    groups.sort((a,b)=>b.length-a.length);const labels=values.map(()=>-1);groups.forEach((group,j)=>group.forEach(i=>labels[i]=j));
    return {mean,std,cv:std/Math.max(Math.abs(mean),1e-12),min:Math.min(...values),max:Math.max(...values),lo:quantile(values,.05),hi:quantile(values,.95),threshold,high,labels,sizes:groups.map(g=>g.length),count:high.filter(Boolean).length};
  }
  function field(mode,parameter,snapshot,material){
    if(mode==='capacity')return material?.fields?.[parameter||'capacity'];
    if(mode==='speed')return snapshot?.material_tensor_eigenvalues?.map(v=>Math.sqrt(Math.max(0,v[2])));
    if(mode==='anisotropy')return snapshot?.material_tensor_eigenvalues?.map(v=>Math.sqrt(Math.max(0,v[2])/Math.max(1e-20,v[0])));
    if(mode==='material'){
      if(parameter==='composition')return snapshot?.material?.map(v=>Math.hypot(...v));
      if(parameter.startsWith('m:'))return snapshot?.material?.map(v=>v[Number(parameter.slice(2))]);
      return material?.fields?.[parameter];
    }
    return null;
  }
  function compositionRGB(value){return value.slice(0,3).map(v=>Math.round(55+190*(.5+.5*Math.tanh(v/1.5))))}
  return {palette,quantile,neighbors,analyze,field,compositionRGB};
})();
