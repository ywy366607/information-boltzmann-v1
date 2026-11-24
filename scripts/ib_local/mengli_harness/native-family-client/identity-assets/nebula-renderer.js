// Visible radial transport with brief chromatic discontinuity on a telemetry scan.
// All texture coordinates keep their angle: the cloud has no rotational flow.
function createNebulaRenderer(canvas, onReady) {
  canvas.width=680;canvas.height=480;
  const gl=canvas.getContext('webgl',{alpha:true,premultipliedAlpha:false,antialias:false,depth:false});
  function fallback(){const im=document.createElement('img');im.src=MENGLI_NEBULA;im.className=canvas.className;im.alt='';canvas.replaceWith(im);return{sync(){},draw(){return 0;}};}
  if(!gl)return fallback();
  try {
    const program=gl.createProgram();
    const vertex='attribute vec2 p;varying vec2 uv;void main(){uv=p*.5+.5;gl_Position=vec4(p,0.,1.);}';
    const fragment=`precision highp float;varying vec2 uv;uniform sampler2D cloud;
      uniform float time,strength,transport,beat,learning,commit;
      vec4 gas(vec2 p,float scale){return texture2D(cloud,p/(.74*scale)+.5);}
      vec4 field(vec2 p){
        vec4 base=gas(p,1.);vec3 rgb=base.rgb*base.a*.35;float alpha=base.a*.35;
        for(int i=0;i<3;i++){
          float age=fract(transport+float(i)/3.);
          float scale=.58+1.04*age;
          float fade=pow(sin(age*3.14159265),2.);
          vec4 veil=gas(p,scale);
          float weight=fade*(.44+.22*strength)*smoothstep(.025,.11,length(p));
          rgb+=veil.rgb*veil.a*weight;alpha+=veil.a*weight;
        }
        return vec4(rgb/max(alpha,.0001),clamp(alpha,0.,1.));
      }
      void main(){
        vec2 p=(uv-vec2(.5,.56))*vec2(1.416667,1.);
        float r=length(p);vec2 direction=p/max(r,.025);
        // Split colours briefly; preserve luminosity and the cloud silhouette.
        float shock=max(beat,learning*.8);
        float band=step(.63,fract(uv.y*19.+floor(time*9.)));
        vec2 offset=direction*(.006+.005*band)*shock;
        vec4 centre=field(p);
        vec3 colour=vec3(field(p-offset).r,centre.g,field(p+offset).b);
        float envelope=1.-smoothstep(.40,.59,r);
        float event=learning*exp(-pow((r-(.025+.46*commit))/.06,2.));
        colour+=vec3(.06,.08,.12)*event;
        gl_FragColor=vec4(colour,centre.a*envelope);
      }`;
    for(const[type,source]of[[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]){
      const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));gl.attachShader(program,shader);
    }
    gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
    const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);
    const p=gl.getAttribLocation(program,'p');gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,2,gl.FLOAT,false,0,0);
    const uniforms=Object.fromEntries(['time','strength','transport','beat','learning','commit'].map(k=>[k,gl.getUniformLocation(program,k)]));
    const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);
    for(const name of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,name,gl.LINEAR);
    for(const name of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,name,gl.CLAMP_TO_EDGE);
    let ready=false,lost=false,lastTime=0,transport=0;const image=new Image();image.onload=()=>{gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);ready=true;onReady();};image.src=MENGLI_NEBULA;
    canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();lost=true;});
    return {
      sync(surfaces){canvas.dataset.flowLayers=String(surfaces.length);canvas.dataset.transport='radial-outward';},
      draw(surfaces,phases,learning,time=0,commit=-1,beat=0,focus=null){
        if(!ready||lost)return 0;
        const strength=focus?.amplitude??0,rate=bodyVisualDynamics(focus).transportRate;
        transport+=Math.max(0,time-lastTime)*rate;lastTime=time;
        canvas.dataset.transportRate=rate.toFixed(5);canvas.dataset.frequency=String(focus?.frequency??'missing');
        gl.useProgram(program);gl.viewport(0,0,680,480);
        for(const[k,v]of Object.entries({time,strength,transport,beat,learning,commit}))gl.uniform1f(uniforms[k],v);
        gl.drawArrays(gl.TRIANGLES,0,6);return beat;
      },
    };
  }catch(error){console.warn('Nebula renderer uses static artwork.',error);return fallback();}
}
