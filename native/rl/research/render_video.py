"""Record an ordinary native replay without touching any training game process."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]

def replay_ticks(path):
    """Tick count from a BOPLREP2 header (includes warmup), or None."""
    try:
        raw=path.read_bytes()
        if not raw.startswith(b'BOPLREP2'): return None
        count,=struct.unpack_from('B',raw,146)
        offset=147
        for _ in range(count):
            slots,=struct.unpack_from('B',raw,offset+4)
            offset+=5+slots*2
        ticks,=struct.unpack_from('<i',raw,offset)
        return ticks if ticks>0 else None
    except (OSError,struct.error): return None

def stop(process):
    if process is None or process.poll() is not None: return
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL); process.wait()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay',type=Path); parser.add_argument('output',type=Path)
    parser.add_argument('--seconds',type=float,default=None,
                        help='Recording length; defaults to the replay tick count (warmup included) plus slack, so the whole match is covered')
    args=parser.parse_args()
    replay=args.replay.resolve(); output=args.output.resolve()
    if not replay.is_file(): raise FileNotFoundError(replay)
    if args.seconds is None:
        ticks=replay_ticks(replay)
        # The replay plays in real time from game start; recording begins ~3 s
        # after launch, so cover load time and the end screen with slack.
        args.seconds=ticks/60+8 if ticks else 40
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    log_path=output.with_suffix('.render.log')
    temporary=output.with_name(output.stem+'.partial.mp4')
    xvfb=game=encoder=None
    with log_path.open('w') as log:
        try:
            xvfb=subprocess.Popen(['Xvfb','-displayfd','1','-screen','0','960x540x24','-nolisten','tcp'],
                                  stdout=subprocess.PIPE,stderr=log,text=True,start_new_session=True)
            if not select.select([xvfb.stdout],[],[],10)[0]: raise RuntimeError('Xvfb startup timed out')
            display=':'+xvfb.stdout.readline().strip()
            if display==':': raise RuntimeError('Xvfb did not assign a display')
            env=dict(os.environ,DISPLAY=display,__NV_PRIME_RENDER_OFFLOAD='1',__GLX_VENDOR_LIBRARY_NAME='nvidia')
            env.pop('BOPL_RL_SOCKET',None); env.pop('BOPL_TRACE',None)
            game_log=output.with_suffix('.game.log')
            game=subprocess.Popen([str(ROOT/'Builds/Linux/BoplBattle.x86_64'),str(replay),
                '-screen-width','960','-screen-height','540','-screen-fullscreen','0','-logFile',str(game_log)],
                env=env,stdout=log,stderr=log,start_new_session=True)
            # The replay starts from a saved round setup and contains four seconds
            # of warmup. Wait for process startup, then include the complete match.
            time.sleep(3)
            if game.poll() is not None: raise RuntimeError(f'Replay player exited: {game.returncode}')
            subprocess.run(['xdotool','mousemove','959','539'],env=env,check=True)
            encoder=subprocess.Popen(['ffmpeg','-y','-nostdin','-f','x11grab','-video_size','960x540',
                '-framerate','30','-i',display,'-t',str(args.seconds),'-an','-c:v','h264_nvenc',
                '-preset','p4','-cq','23','-threads','2','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(temporary)],stdout=log,stderr=log,start_new_session=True)
            code=encoder.wait(timeout=args.seconds+30)
            if code: raise RuntimeError(f'Video encoder exited: {code}')
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
                'format=duration:stream=codec_name,width,height','-of','json',str(temporary)],text=True))
            if float(info['format']['duration'])<args.seconds-1: raise RuntimeError('Incomplete recording')
            temporary.replace(output)
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-hwaccel','cuda',
                            '-ss',str(min(7,args.seconds/2)),'-i',str(output),'-frames:v','1','-y',
                            str(output.with_suffix('.jpg'))],stdout=log,stderr=log,check=True)
            output.with_suffix('.video.json').write_text(json.dumps(dict(replay=str(replay),seconds=args.seconds,probe=info),indent=2)+'\n')
            print(json.dumps(dict(status='VIDEO_READY',path=str(output),seconds=args.seconds)),flush=True)
        finally:
            stop(encoder); stop(game); stop(xvfb)

if __name__=='__main__': main()
