"""Five observable, semantic-free rooms for learning-progress diagnostics.

Legacy routing uses a/f .. e/j; the direct protocol uses the last three
byte parities, with all five destinations directly addressable. Payload alphabets
are disjoint, so room identity need not be remembered through a long cue.
No room score or entropy floor is supplied to the organism.
"""
import copy

import numpy as np


ROOMS = ('familiar', 'noise', 'easy', 'hard', 'changing')
ORDERS = (0, 0, 2, 6, 3)
FLIP_RATE = .15


class LearningRoomsWorld:
    def __init__(self, seed=11, switch_turn=256, routing='address'):
        if routing not in ('address','direct_bits'):
            raise ValueError('unknown routing protocol')
        self.routing=routing
        self.rng = np.random.default_rng(seed)
        self.anchor = 0
        self.moves = 0
        self.turns = 0
        self.emitted = 0
        self.switch_turn = int(switch_turn)
        self.offsets = [0]*5
        self.histories = [0,0,1,37,5]

    @property
    def offset(self):
        return self.offsets[self.anchor]

    @staticmethod
    def destination(utterance, current, routing='address'):
        if routing=='direct_bits':
            if len(utterance)<3:
                return int(current)
            key=0
            for byte in utterance[-3:]:
                key=2*key+((byte-ord('a'))&1)
            return (0,2,1,3,3,4,2,4)[key]
        if utterance and ord('a') <= utterance[-1] <= ord('j'):
            return (utterance[-1]-ord('a')) % 5
        return int(current)

    def _emit(self, room, n, changed):
        if room == 0:
            motif = np.frombuffer(b'aabb',np.uint8)
            return motif[(self.offsets[room]+np.arange(n))%4].copy()
        if room == 1:
            return self.rng.integers(ord('c'),ord('e'),n,dtype=np.uint8)
        k = ORDERS[room]
        history = self.histories[room]
        mask = (1<<k)-1
        data = np.empty(n,np.uint8)
        noise = self.rng.random(n) < FLIP_RATE
        for i in range(n):
            bit = (int(history).bit_count() & 1) ^ int(changed) ^ int(noise[i])
            data[i] = ord('a')+2*room+bit
            history = ((history<<1)|bit)&mask
        self.histories[room] = history
        return data

    def take(self, utterance, response_bytes, holdout_bytes):
        if min(response_bytes,holdout_bytes)<1:
            raise ValueError('positive real observation intervals required')
        previous = self.anchor
        self.anchor = self.destination(utterance,previous,self.routing)
        accepted = (len(utterance)>=3 if self.routing=='direct_bits' else
                    bool(utterance and ord('a') <= utterance[-1] <= ord('j')))
        changed = self.turns >= self.switch_turn
        n = response_bytes+holdout_bytes
        data = self._emit(self.anchor,n,changed and self.anchor==4)
        start = self.emitted
        self.emitted += n
        self.offsets[self.anchor] = (self.offsets[self.anchor]+n)%4
        self.moves += int(previous!=self.anchor)
        self.turns += 1
        info = dict(start=start,anchor=self.anchor,room=ROOMS[self.anchor],
            room_before=previous,match=int(accepted),moved=previous!=self.anchor,
            rule_changed=bool(changed and self.anchor==4),
            holdout_position_fresh=True,source='address' if accepted else 'stay')
        return data[:response_bytes],data[response_bytes:],info

    def state_dict(self):
        return dict(anchor=self.anchor,moves=self.moves,turns=self.turns,routing=self.routing,
            emitted=self.emitted,switch_turn=self.switch_turn,
            offsets=list(self.offsets),histories=list(self.histories),
            rng=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self,state):
        self.routing=state.get('routing','address')
        for key in ('anchor','moves','turns','emitted','switch_turn'):
            setattr(self,key,int(state[key]))
        self.offsets=list(state['offsets'])
        self.histories=list(state['histories'])
        self.rng.bit_generator.state=copy.deepcopy(state['rng'])


def validation_banks(length=1024,seed=1901):
    banks = {}
    for changed in (False,True):
        bank = []
        for room in range(5):
            world = LearningRoomsWorld(seed+room,switch_turn=0 if changed else 10**9)
            world.anchor=room
            world._emit(room,64,changed and room==4)  # independent prefix warmup
            bank.append(world._emit(room,length+1,changed and room==4))
        banks[int(changed)]=bank
    return banks


def entropy_floors():
    h=-FLIP_RATE*np.log(FLIP_RATE)-(1-FLIP_RATE)*np.log(1-FLIP_RATE)
    return [0.,float(np.log(2)),float(h),float(h),float(h)]
