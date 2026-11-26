"""Optional W&B metrics for online trainers, plus import/follow of existing JSONL.

python -m training.wandb_logging --outdir RUN_DIR --project carla-sage --follow
The follower only reads logs; it does not connect to CARLA or change training.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time
import warnings


def numeric_metrics(record,prefix):
    result={}
    for key,value in record.items():
        name=f'{prefix}/{key}'
        if isinstance(value,dict): result.update(numeric_metrics(value,name))
        elif isinstance(value,(int,float)) and math.isfinite(value): result[name]=value
    return result


class WandbLogger:
    def __init__(self,args,config):
        self.run=self.sdk=None
        self.enabled=False
        self.started=time.monotonic()
        self.window=deque(maxlen=getattr(args,'log_interval',100))
        self.episodes=deque(maxlen=100)
        if not getattr(args,'wandb',False): return
        try: import wandb
        except ImportError as error:
            raise RuntimeError('W&B requested: install requirements-wandb.txt, then run wandb login or choose --wandb-mode offline') from error
        self.sdk=wandb
        self.run=wandb.init(project=args.wandb_project,entity=args.wandb_entity,
            name=args.wandb_name or args.outdir.name,group=args.wandb_group,
            mode=args.wandb_mode,dir=str(args.outdir.resolve()),config=config,save_code=False)
        self.run.define_metric('environment_step')
        self.run.define_metric('*',step_metric='environment_step')
        self.enabled=True
        (args.outdir/'wandb_run.json').write_text(json.dumps(dict(id=self.run.id,url=self.run.url,
            project=args.wandb_project,mode=args.wandb_mode),indent=2)+'\n')

    def log(self,metrics,step):
        if not self.enabled: return
        try: self.run.log(dict(environment_step=int(step),**metrics))
        except Exception as error:
            self.enabled=False
            warnings.warn(f'W&B logging disabled after error; local logs continue: {error}',RuntimeWarning)

    def transition(self,record):
        if not self.enabled: return
        row={k:record.get(k) for k in ('action','gap','ego_speed','leader_speed','closing_rate')}
        row.update(intervened=record['behavior']['intervened'],nominal_q=record['behavior']['nominal_q'])
        self.window.append(row)

    def progress(self,record,metrics=None):
        if not self.enabled: return
        data=numeric_metrics({k:v for k,v in record.items() if k!='step'},'progress')
        if metrics: data.update(numeric_metrics({k:v for k,v in metrics.items() if k!='step'},'train'))
        for key in ('action','gap','ego_speed','leader_speed','closing_rate','intervened','nominal_q'):
            values=[r[key] for r in self.window if isinstance(r[key],(int,float)) and math.isfinite(r[key])]
            if values:
                name='intervention_fraction' if key=='intervened' else f'mean_{key}'
                data[f'behavior/{name}']=sum(values)/len(values)
        if self.window:
            data['behavior/minimum_action']=min(r['action'] for r in self.window)
            data['behavior/maximum_action']=max(r['action'] for r in self.window)
        elapsed=time.monotonic()-self.started
        data.update({'time/elapsed_seconds':elapsed,
                     'time/effective_steps_per_second':record['step']/max(elapsed,1e-9)})
        self.log(data,record['step'])

    def episode(self,record):
        if not self.enabled: return
        data=numeric_metrics({k:v for k,v in record.items() if k!='step'},'episode')
        collision=bool(record.get('collision_events'))
        data['episode/collision']=int(collision)
        if record.get('admitted_to_replay',False):
            self.episodes.append(dict(collision=collision,failed=record['failed']))
        if self.episodes:
            for key in ('collision','failed'):
                data[f'episode/{key}_rate_last_100_valid']=sum(r[key] for r in self.episodes)/len(self.episodes)
            data['episode/rolling_valid_count']=len(self.episodes)
        counts=record.get('action_sources',{})
        data['episode/intervention_fraction']=counts.get('safety_actor',0)/max(sum(counts.values()),1)
        self.log(data,record['step'])

    def evaluation(self,summary,plot_path=None,records=None):
        if not self.enabled: return
        data={'eval/seconds':summary['seconds'],'eval/threshold':summary['threshold']}
        scopes={'all':summary}
        for kind in ('profile','start','braking'):
            scopes.update({f'{kind}/{name}':value for name,value in summary.get(f'by_{kind}',{}).items()})
        goal=summary.get('nominal_kind')=='goal'
        names={'pid':'goal' if goal else 'pid','safety':'safety','filter':'sage' if goal else 'filter',
               'pid_then_safety':'recovery_probe'}
        renames={'collision':'collision_rate','failure':'failure_rate','lane_departure':'lane_departure_rate',
                 'mean_speed':'mean_speed_mps','mean_absolute_gap_error':'gap_mae_m'}
        for scope,report in scopes.items():
            for key in ('paired_cases','excluded_cases'): data[f'eval/{scope}/{key}']=report[key]
            for mode,metrics in report['modes'].items():
                metrics={renames.get(key,key):value for key,value in metrics.items()}
                data.update(numeric_metrics(metrics,f'eval/{scope}/{names.get(mode,mode)}'))
        data.update(numeric_metrics(summary.get('braking_baseline',{}),'eval/braking_baseline'))
        data.update(numeric_metrics(summary.get('change_from_previous',{}),'eval/change_from_previous'))
        try:
            if plot_path is not None and Path(plot_path).is_file():
                data['eval/overview']=self.sdk.Image(str(plot_path))
            if records is not None:
                columns=['case','mode','valid','collision','initial_gap','initial_ego_speed','initial_leader_speed',
                         'probe_q','mean_action','brake_q','coast_q','throttle_q','safety_action','error']
                table=[]
                for record in records:
                    row=dict(record,**(record.get('action_scan') or {}))
                    table.append([row.get(key) for key in columns])
                data['eval/rollouts']=self.sdk.Table(columns=columns,data=table)
        except Exception as error:
            warnings.warn(f'W&B media logging skipped: {error}',RuntimeWarning)
        self.log(data,summary['step'])

    def finish(self,exit_code=0):
        if self.run is not None:
            try: self.run.finish(exit_code=exit_code)
            except Exception as error: warnings.warn(f'Could not finish W&B run: {error}',RuntimeWarning)


class JsonlTail:
    """Read complete new records only; leave partial last writes for the next poll."""
    def __init__(self,path): self.path,self.offset=Path(path),0
    def read(self):
        if not self.path.exists(): return []
        records=[]
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                start=stream.tell(); line=stream.readline()
                if not line or not line.endswith(b'\n'):
                    self.offset=start; break
                try: records.append(json.loads(line))
                except json.JSONDecodeError:
                    warnings.warn(f'Skipping malformed complete JSON line in {self.path}',RuntimeWarning)
                self.offset=stream.tell()
        return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    parser.add_argument('--project',dest='wandb_project',default='carla-sage')
    parser.add_argument('--entity',dest='wandb_entity')
    parser.add_argument('--name',dest='wandb_name')
    parser.add_argument('--group',dest='wandb_group')
    parser.add_argument('--mode',dest='wandb_mode',choices=('online','offline'),default='online')
    parser.add_argument('--follow',action='store_true')
    args=parser.parse_args(); args.wandb=True; args.log_interval=100
    config=json.loads((args.outdir/'manifest.json').read_text())
    config['wandb_log_import']=True
    logger=WandbLogger(args,config)
    tails={name:JsonlTail(args.outdir/f'{name}.jsonl') for name in ('updates','progress','episodes','evaluation')}
    try:
        while True:
            events=[(row['step'],name,row) for name,tail in tails.items() for row in tail.read()]
            for step,name,row in sorted(events,key=lambda item:(item[0],item[1])):
                if name=='updates': logger.log(numeric_metrics({k:v for k,v in row.items() if k!='step'},'train'),step)
                elif name=='progress': logger.log(numeric_metrics({k:v for k,v in row.items() if k!='step'},'progress'),step)
                elif name=='episodes': logger.episode(row)
                else: logger.evaluation(row)
            if not args.follow: break
            time.sleep(10.)
    except KeyboardInterrupt: pass
    finally: logger.finish()


if __name__=='__main__': main()
