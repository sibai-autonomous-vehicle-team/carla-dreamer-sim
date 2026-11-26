"""Tracking stays optional, uses environment steps, and follows complete records."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import json
import unittest
from unittest.mock import MagicMock,patch
import warnings

from training.wandb_logging import WandbLogger,JsonlTail,numeric_metrics,main


class WandbTests(unittest.TestCase):
    def args(self,path,enabled=True):
        return NS(outdir=Path(path),wandb=enabled,wandb_project='test',wandb_entity=None,
                  wandb_name=None,wandb_group=None,wandb_mode='offline',log_interval=2)

    def sdk(self):
        sdk=MagicMock()
        sdk.init.return_value.id='test-id'; sdk.init.return_value.url=None
        return sdk

    def test_disabled_does_not_import_sdk_or_write_anything(self):
        with TemporaryDirectory() as path,patch.dict('sys.modules',{'wandb':None}):
            logger=WandbLogger(self.args(path,False),{})
            logger.log({'test':1},100);logger.finish()
            self.assertIsNone(logger.run);self.assertEqual(list(Path(path).iterdir()),[])

    def test_numeric_filter_keeps_real_zeros_and_omits_missing_nonfinite(self):
        self.assertEqual(numeric_metrics(dict(a=0,b=None,c=float('nan'),d={'x':True},text='secret'),'train'),
                         {'train/a':0,'train/d/x':True})

    def test_logs_episode_progress_and_media_against_environment_steps(self):
        with TemporaryDirectory() as path:
            sdk=self.sdk()
            with patch.dict('sys.modules',{'wandb':sdk}): logger=WandbLogger(self.args(path),{'steps':1000})
            self.assertEqual(sdk.init.call_args.kwargs['mode'],'offline')
            logger.run.define_metric.assert_any_call('*',step_metric='environment_step')
            for action,gap,intervened in [(-3.,2.,True),(1.,4.,False)]:
                logger.transition(dict(action=action,gap=gap,ego_speed=15.,leader_speed=14.,closing_rate=1.,
                    behavior=dict(intervened=intervened,nominal_q=None)))
            logger.progress(dict(step=100,updates=20,replay_rows=30),dict(critic_mse=.1,goal_critic_mse=.2))
            payload=logger.run.log.call_args.args[0]
            self.assertEqual(payload['environment_step'],100)
            self.assertEqual(payload['behavior/intervention_fraction'],.5)
            self.assertEqual(payload['behavior/mean_action'],-1.)
            self.assertEqual(payload['train/goal_critic_mse'],.2)
            self.assertNotIn('step',logger.run.log.call_args.kwargs)
            logger.episode(dict(step=101,failed=True,collision_events=[[1,2]],admitted_to_replay=True,
                action_sources=dict(safety_actor=2,goal_actor=2),performance_return=-3.))
            self.assertEqual(logger.run.log.call_args.args[0]['episode/collision_rate_last_100_valid'],1.)
            report=dict(paired_cases=6,excluded_cases=0,modes={
                'pid':dict(collision=.5),'filter':dict(collision=0.),
                'pid_then_safety':dict(collision_rate_when_predicted_safe=None,predicted_safe_cases=0)})
            summary=dict(step=5000,seconds=30.,threshold=1.,nominal_kind='goal',**report,
                         by_start={'ordinary':report})
            image=Path(path)/'plot.png';image.touch()
            logger.evaluation(summary,image,[dict(case=0,mode='filter',valid=True,action_scan=dict(brake_q=2.))])
            payload=logger.run.log.call_args.args[0]
            self.assertEqual(payload['environment_step'],5000)
            self.assertEqual(payload['eval/all/sage/collision_rate'],0.)
            self.assertEqual(payload['eval/start/ordinary/goal/collision_rate'],.5)
            self.assertNotIn('eval/all/recovery_probe/collision_rate_when_predicted_safe',payload)
            sdk.Image.assert_called_once_with(str(image)); sdk.Table.assert_called_once()
            logger.finish(exit_code=1);logger.run.finish.assert_called_once_with(exit_code=1)

    def test_failed_upload_does_not_raise_and_disables_further_logs(self):
        with TemporaryDirectory() as path:
            sdk=self.sdk()
            with patch.dict('sys.modules',{'wandb':sdk}): logger=WandbLogger(self.args(path),{})
            logger.run.log.side_effect=RuntimeError('connection lost')
            with warnings.catch_warnings(record=True) as caught:
                logger.log({'x':1},1)
            self.assertTrue(caught);self.assertFalse(logger.enabled)
            logger.log({'x':2},2);self.assertEqual(logger.run.log.call_count,1)
            logger.finish();logger.run.finish.assert_called_once()

    def test_tail_does_not_lose_partial_records_or_duplicate_completed_ones(self):
        with TemporaryDirectory() as path:
            file=Path(path)/'updates.jsonl';tail=JsonlTail(file)
            self.assertEqual(tail.read(),[])
            file.write_bytes(b'{"step":1}\n{"step":')
            self.assertEqual(tail.read(),[{'step':1}]);self.assertEqual(tail.read(),[])
            with file.open('ab') as stream:stream.write(b'2}\n')
            self.assertEqual(tail.read(),[{'step':2}]);self.assertEqual(tail.read(),[])

    def test_importer_reads_existing_logs_in_step_order_without_training(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'manifest.json').write_text('{"steps":1000}')
            (root/'updates.jsonl').write_text('{"step":100,"critic_mse":0.5}\n')
            (root/'progress.jsonl').write_text('{"step":50,"replay_rows":20}\n')
            sdk=self.sdk()
            with patch.dict('sys.modules',{'wandb':sdk}),patch('sys.argv',[
                'wandb_logging','--outdir',directory,'--mode','offline']):main()
            self.assertEqual([call.args[0]['environment_step'] for call in sdk.init.return_value.log.call_args_list],[50,100])
            sdk.init.return_value.finish.assert_called_once()


if __name__=='__main__':unittest.main()
