"""Offline anime video matte + flow interpolation; no runtime inference needed.
Requires SkyTNT anime-seg ISNetDIS source and safetensors downloaded from the
upstream repositories. Packed MP4 stores premultiplied RGB above alpha, decoded
by life-core.js so browser codec alpha support is irrelevant.
"""
import argparse, json, pathlib, subprocess, sys, time
import cv2
import numpy as np
import torch
from safetensors.torch import load_file

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--model-dir',required=True);ap.add_argument('--output',required=True);ap.add_argument('--ffmpeg',required=True);ap.add_argument('--preview-dir',required=True)
    ap.add_argument('--loop-seconds',type=float,default=10)
    args=ap.parse_args();sys.path.insert(0,args.model_dir)
    from model_isnet import ISNetDIS
    torch.set_num_threads(2);cv2.setNumThreads(2)
    model=ISNetDIS().eval().cuda();weights=load_file(str(pathlib.Path(args.model_dir)/'model.safetensors'))
    model.load_state_dict({k[4:]:v for k,v in weights.items() if k.startswith('net.')});del weights
    capture=cv2.VideoCapture(args.source);fps=capture.get(cv2.CAP_PROP_FPS);count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));preview=pathlib.Path(args.preview_dir);preview.mkdir(parents=True,exist_ok=True)
    width,height=384,454
    raw_output=args.output+'.raw.mp4'
    cmd=[args.ffmpeg,'-y','-hide_banner','-loglevel','warning','-threads','2','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{width}x{height*2}','-framerate','60','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',raw_output]
    encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    flow_engine=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    yy,xx=np.mgrid[:height,:width].astype(np.float32)
    def matte(frame):
        h,w=frame.shape[:2];s=640;nw=int(s*w/h);left=(s-nw)//2
        x=np.zeros((s,s,3),np.float32);x[:,left:left+nw]=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(nw,s))/255.
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=model(torch.from_numpy(x.transpose(2,0,1)[None]).cuda())[0][0].sigmoid()
        p=pred.float().cpu().numpy()[0,0,:,left:left+nw];p=cv2.resize(p,(w,h))
        # Remove low-confidence seat/background; retain a narrow antialiased edge.
        p=np.clip((p-.24)/.64,0,1);p=p*p*(3-2*p)
        n,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(p>.5),8)
        if n>1:
            largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
            support=cv2.dilate(np.uint8(labels==largest),np.ones((5,5),np.uint8))
            p*=support
        return cv2.resize(frame[238:1088],(width,height)),cv2.resize(p[238:1088],(width,height))
    def emit(colour,alpha):
        premult=np.uint8(np.clip(colour*alpha[:,:,None],0,255));a=np.uint8(np.clip(alpha*255,0,255));packed=np.concatenate([premult,cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)],axis=0);encoder.stdin.write(packed.tobytes())
    previous=None;next_out=0;start=time.time();areas=[]
    for i in range(count):
        ok,frame=capture.read()
        if not ok:break
        rgb,alpha=matte(frame);areas.append(float((alpha>.5).mean()))
        if i in [0,48,96,144,192,240,288]:
            cv2.imwrite(str(preview/f'cutout-{i:03}.png'),np.dstack([rgb,np.uint8(alpha*255)]))
            checker=((xx//16+yy//16)%2)*26+42
            check=np.uint8(rgb*alpha[:,:,None]+checker[:,:,None]*(1-alpha[:,:,None]));cv2.imwrite(str(preview/f'check-{i:03}.jpg'),check)
        if previous is not None:
            old,oa=previous;g0=cv2.cvtColor(old,cv2.COLOR_BGR2GRAY);g1=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
            f=flow_engine.calc(g0,g1,None);b=flow_engine.calc(g1,g0,None)
            # Transport colour and matte with exactly the same bidirectional flow.
            # Smooth confidence edges temporally only after motion compensation.
            warped=cv2.remap(oa,xx+b[:,:,0],yy+b[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            edge=(alpha>.02)&(alpha<.98)&(np.abs(alpha-warped)<.18)
            alpha=np.where(edge,.85*alpha+.15*warped,alpha)
            while next_out/60 <= i/fps + 1e-7:
                t=np.float32(np.clip((next_out/60-(i-1)/fps)*fps,0,1))
                c0=cv2.remap(old,xx-f[:,:,0]*t,yy-f[:,:,1]*t,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
                c1=cv2.remap(rgb,xx-b[:,:,0]*(1-t),yy-b[:,:,1]*(1-t),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
                a0=cv2.remap(oa,xx-f[:,:,0]*t,yy-f[:,:,1]*t,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                a1=cv2.remap(alpha,xx-b[:,:,0]*(1-t),yy-b[:,:,1]*(1-t),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                a=a0*(1-t)+a1*t
                # Blend premultiplied colours to prevent dark backgrounds leaking
                # into moving hair and the gaps between the alternating legs.
                c=(c0*a0[:,:,None]*(1-t)+c1*a1[:,:,None]*t)/np.maximum(a[:,:,None],1e-6)
                emit(c,a);next_out+=1
        previous=rgb,alpha
        if i%24==0: print(json.dumps({'frame':i,'total':count,'seconds':round(time.time()-start,1),'encoded':next_out}),flush=True)
    # Continue the last native exposure to its original endpoint.
    while next_out/60<count/fps:emit(previous[0],previous[1]);next_out+=1
    encoder.stdin.close();code=encoder.wait();capture.release()
    if code:raise RuntimeError(f'ffmpeg exit {code}')
    graph=f'[0:v]trim=end={args.loop_seconds},setpts=PTS-STARTPTS,split[a][b];[b]reverse,setpts=PTS-STARTPTS[r];[a][r]concat=n=2:v=1:a=0[v]'
    subprocess.run([args.ffmpeg,'-y','-hide_banner','-loglevel','warning','-threads','2','-i',raw_output,'-filter_complex',graph,'-map','[v]','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',args.output],check=True)
    pathlib.Path(raw_output).unlink()
    meta={'source':args.source,'native_fps':fps,'native_frames':len(areas),'output_fps':60,'output_frames':next_out,'size':[width,height],'packing':'top premultiplied BGR; bottom grayscale alpha','matte_model':'skytnt/anime-seg ISNetDIS','alpha_threshold':[.24,.88],'flow':'OpenCV DIS medium bidirectional colour+alpha transport','foreground_area_range':[min(areas),max(areas)],'seconds':time.time()-start}
    meta['loop']={'trim_seconds':args.loop_seconds,'playback':'forward then reverse','duration':args.loop_seconds*2}
    pathlib.Path(args.output+'.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
