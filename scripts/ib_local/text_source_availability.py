"""Private real-source change control, keeping content navigation and state.

Only externally available text changes. Addresses in receipts remain absolute;
the body receives no source labels. No neural/table/optimizer state is reset.
"""
class TextSourceAvailability:
    def __init__(self, world):
        if world.navigation_policy != 'stable_context_v1':
            raise ValueError('source control requires the current stable text world')
        self.world = world
        self.bounds = (0,len(world.base.data))

    def __getattr__(self, name):
        return getattr(self.world,name)

    def take(self, utterance, response_bytes, holdout_bytes):
        lo,hi = self.bounds
        base = self.world.base
        data,seen = base.data,base.seen
        prior_cursor,prior_anchor = base.cursor,base.anchor
        if not 0 <= lo < hi <= len(data) or hi-lo < base.window:
            raise ValueError('real available source must contain a complete world window')
        base.data,base.seen = data[lo:hi],seen[lo:hi]
        base.cursor = min(max(base.cursor-lo,0),hi-lo-1)
        succeeded = False
        try:
            response,continuation,info = self.world.take(utterance,response_bytes,holdout_bytes)
            succeeded = True
        finally:
            base.data,base.seen = data,seen
            if succeeded:
                base.cursor += lo
                base.anchor += lo
            else:
                base.cursor,base.anchor = prior_cursor,prior_anchor
        start = int(info['start'])+lo
        info.update(start=start,anchor=int(info['anchor'])+lo,
                    available_source_bounds=[lo,hi])
        manifest = self.world.source_manifest
        if manifest is not None:
            split = int(manifest['prefix_bytes'])
            def counts(begin,size):
                old = max(0,min(begin+size,split)-begin)
                return dict(inherited_prefix=old,natural_append=size-old)
            info.update(text_source_bytes=counts(start,len(response)+len(continuation)),
                text_response_source_bytes=counts(start,len(response)),
                text_holdout_source_bytes=counts(start+len(response),len(continuation)))
        return response,continuation,info
