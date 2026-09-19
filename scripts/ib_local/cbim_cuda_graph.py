"""Fixed-shape FP32 CBIM training replay; preserves full-window BPTT.

Capture removes Python/driver launches, not recurrent evolution. The runner owns
a fresh optimizer; load resumed optimizer state after construction if required.
"""
import torch


class CBIMGraphTrainer:
    def __init__(self, model, tokens=128, batch_size=1, lr=3e-4):
        self.model = model
        device = next(model.parameters()).device
        if device.type != 'cuda':
            raise ValueError('CUDA is required')
        self.ids = torch.zeros(batch_size, tokens, dtype=torch.long, device=device)
        self.targets = torch.zeros_like(self.ids)
        state_shape = getattr(model, 'state_shape', (model.L, model.d))
        if hasattr(model, "initial_state"):
            self.state = model.initial_state(batch_size, device=device)
        else:
            self.state = torch.zeros(batch_size, *state_shape, device=device)
        decay, no_decay = [], []
        for parameter in model.parameters():
            target = no_decay if getattr(parameter, "_no_weight_decay", False) else decay
            target.append(parameter)
        parameter_groups = [{"params": decay}]
        if no_decay:
            parameter_groups.append({"params": no_decay, "weight_decay": 0.0})
        self.optimizer = torch.optim.AdamW(
            parameter_groups, lr=lr, foreach=True, capturable=True)
        original = [p.detach().clone() for p in model.parameters()]
        initial_state = self.state.detach().clone()

        def update():
            self.optimizer.zero_grad(set_to_none=True)
            loss, state, diagnostics = model(self.ids, self.targets, self.state)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
            self.optimizer.step()
            with torch.no_grad():
                self.state.copy_(state)
            return loss, diagnostics, grad_norm

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                update()
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.diagnostics, self.grad_norm = update()
        # Capture/warmup must not count as training or alter initial weights.
        with torch.no_grad():
            for parameter, value in zip(model.parameters(), original):
                parameter.copy_(value)
            for state in self.optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()
            self.state.copy_(initial_state)

    def step(self, ids, targets):
        if ids.shape != self.ids.shape or targets.shape != self.targets.shape:
            raise ValueError('Batch/token dimensions must match captured shape')
        self.ids.copy_(ids)
        self.targets.copy_(targets)
        self.graph.replay()
        # Results use static storage; consume or clone before the next replay.
        return self.loss.detach(), self.state, self.diagnostics
