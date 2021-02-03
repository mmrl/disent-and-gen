"""
Auxiliary handlers for use during training.
"""


import os
import csv
# import numpy as np
import torch
import ignite.handlers as hdlr
from ignite.handlers import EarlyStopping
from ignite.engine import Events


class ModelCheckpoint(hdlr.ModelCheckpoint):
    @property
    def last_checkpoint_state(self):
        with open(self.last_checkpoint, mode='rb') as f:
            state_dict = torch.load(f)
        return state_dict

    @property
    def all_paths(self):
        def name_path_tuple(p):
            return p.filename, os.path.join(self.save_handler.dirname,
                                            p.filename)

        return [name_path_tuple(p) for p in self._saved]


class LRScheduler(object):
    def __init__(self, scheduler, loss):
        self.scheduler = scheduler
        self.loss = loss

    def __call__(self, engine):
        loss_val = engine.state.metrics[self.loss]
        self.scheduler.step(loss_val)

    def attach(self, engine):
        engine.add_event_handler(Events.COMPLETED, self)
        return self


class Tracer(object):
    def __init__(self, val_metrics, save_path=None, save_interval=1):
        self.metrics = ['loss']
        self.loss = []
        self.save_path = save_path
        self.save_interval = save_interval
        self._running_loss = 0
        self._n_inputs = 0

        template = 'val_{}'
        for k in val_metrics:
            name = template.format(k)
            setattr(self, name, [])
            self.metrics.append(name)

    def _initalize_traces(self, engine):
        for k in self.metrics:
            getattr(self, k).clear()

    def _save_batch_loss(self, engine):
        n_examples = engine.state.batch[1].size(0)
        self._running_loss += engine.state.output * n_examples
        self._n_inputs += n_examples

    def _compute_training_loss(self, engine):
        epoch_loss = self._running_loss / self._n_inputs
        self.loss.append(epoch_loss)
        self._running_loss = 0.0
        self._n_inputs = 0

    def _trace_validation(self, engine):
        metrics = engine.state.metrics
        template = 'val_{}'
        for k, v in metrics.items():
            trace = getattr(self, template.format(k))
            trace.append(v)

    def attach(self, trainer, evaluator=None):
        trainer.add_event_handler(Events.STARTED, self._initalize_traces)
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, self._save_batch_loss)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, self._compute_training_loss)

        if evaluator is not None:
            evaluator.add_event_handler(
                Events.COMPLETED, self._trace_validation)

        if self.save_path is not None:
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED, self._save_at_interval)

        return self

    def _save_at_interval(self, engine):
        if engine.state.iteration % self.save_interval == 0:
            self.save_traces()

    def save_traces(self):
        for loss in self.metrics:
            trace = getattr(self, loss)
            with open('{}/{}.csv'.format(self.save_path, loss), mode='w') as f:
                wr = csv.writer(f, quoting=csv.QUOTE_ALL)
                for i, v in enumerate(trace):
                    wr.writerow([i + 1, v])
