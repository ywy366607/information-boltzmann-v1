"""Eight-round active versus sequential corpus trials from the archived 2805 boundary."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from scripts.ib_local import run_mother_continuation as continuation
from scripts.ib_local.inquiry_navigation import POLICIES
from scripts.run_native_supply_life import main as train

ROOT = Path.home()/'mengli-inquiry-2805-trial'
OLD = Path('D:/mengli-native-supply-life-bounded-20261122')

def run(policy):
    assert policy in POLICIES
    protocol=OLD/'protocol.json'
    spec=json.loads(protocol.read_text(encoding='utf8'))
    original=continuation.prepare_continuation
    def prepare(*a,**kw):
        loaded=original(*a,**kw)
        assert loaded.cycle==2805 and loaded.engine.world.navigation_policy=='stable_context_v1'
        loaded.engine.world.navigation_policy=policy
        loaded.config['text_navigation_policy']=policy
        return loaded
    dest=ROOT/policy
    args=['--supply-coefficient','0','--defer-initial-checkpoint','--activation-host-mib','512','--activation-directory',str(ROOT/'activations'),'--mode','natural','--checkpoint',spec['source'],'--work',str(dest),'--temporary',str(dest/'temp'),'--protocol',str(protocol)]
    with patch.object(continuation,'prepare_continuation',prepare),patch.object(sys,'argv',[sys.argv[0],*args]): train()

if __name__=='__main__': run(sys.argv[1])
