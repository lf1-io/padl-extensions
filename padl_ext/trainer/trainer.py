import collections
import json

import padl
import torch


@padl.transform
class Trainer:
    def __init__(self, train_model, optimizer, infer_model=None,
                 metrics=None, iterate_args=None):
        self.train_model = train_model
        self.infer_model = infer_model
        self.optimizer = optimizer
        self.metrics = metrics
        if iterate_args is None:
            self.iterate_args = {}
        self.metric_data = {}
        self.iteration = 0
        self.epoch = 0

    def pre_save(self, path, i):
        print(f'saving optimizer state to {path}/{i}.pt')
        torch.save(self.optimizer.state_dict(), path / f'{i}.pt')

        print(f'saving iterate args to {path}/{i}.iterate.json')
        with open(path / f'{i}.iterate.json', 'w') as f:
            json.dump(self.iterate_args, f)

        print(f'saving metrics to {path}/{i}.metrics.json')
        with open(path / f'{i}.metrics.json', 'w') as f:
            json.dump(self.metric_data, f)

        print(f'saving iteration to {path}/{i}.iteration.json')
        with open(path / f'{i}.iteration.json', 'w') as f:
            json.dump(self.iteration, f)

        print(f'saving epoch to {path}/{i}.epoch.json')
        with open(path / f'{i}.epoch.json', 'w') as f:
            json.dump(self.epoch, f)

    def post_load(self, path, i):
        sd = torch.load(path / f'{i}.pt', map_location=self.pd_device)
        print('loading optimizer from state-dict')
        self.optimizer.load_state_dict(sd)

        print(f'loading iterate args')
        with open(path / f'{i}.iterate.json') as f:
            self.iterate_args.update(json.load(f))

        print(f'loading metrics')
        with open(path / f'{i}.metrics.json') as f:
            self.metric_data = json.load(f)

        print(f'loading iteration')
        with open(path / f'{i}.iteration.json') as f:
            self.iteration = json.load(f)
        self.iteration += 1

        print(f'loading epoch')
        with open(path / f'{i}.epoch.json') as f:
            self.epoch = json.load(f)
        self.epoch += 1

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def train_step(self, output):
        self.optimizer.zero_grad()
        output.backward()
        self.optimizer.step()

    def aggregate_losses(self, losses):
        return sum([x.item() for x in losses]) / len(losses)

    def log_train(self, output):
        print(f'TRAIN epoch {self.epoch}; iteration {self.iteration}; loss {output};')

    def log_valid(self, metrics):
        str_ = ''.join([f'{k}: {v[-1]};' for k, v in metrics.items()])
        print(f'TRAIN epoch {self.epoch}; iteration {self.iteration};' + str_)

    def save(self, path, *args, **kwargs):
        self.pd_save(path, force_overwrite=True)

    def watch_function(self, metrics):
        try:
            return metrics['loss'][-1] == min(metrics['loss'])
        except (KeyError, IndexError):
            return True

    def train(self, train_data, path, valid_data=None, metric_data=None, ground_truth=None,
              n_epochs=100, max_iterations=1000000, save_interval=None,
              valid_interval=100, **kwargs):

        self.iterate_args.update(kwargs)

        metrics = collections.defaultdict(lambda: [])
        metrics.update(self.metric_data)

        for epoch in range(self.epoch, n_epochs):
            for output in self.train_model.train_apply(train_data, **self.iterate_args):
                self.train_step(output)
                self.log_train(output)

                if valid_data is None:
                    assert save_interval is not None
                    if self.iteration % save_interval == 0:
                        self.save(path, self.iteration, epoch)
                    self.iteration += 1
                    continue

                if self.iteration and self.iteration % valid_interval == 0 and valid_data is not None:
                    valid_loss = []
                    for output in self.train_model.eval_apply(valid_data, **self.iterate_args):
                        valid_loss.append(output)
                    valid_loss = self.aggregate_losses(valid_loss)
                    metrics['loss'].append(valid_loss)

                    if self.infer_model is not None and self.metrics is not None:
                        assert len(self.metrics)
                        predictions = list(self.infer_model.eval_apply(metric_data, **self.iterate_args))
                        for metric in self.metrics:
                            metrics[metric].append(self.metrics[metric](predictions, ground_truth))

                    self.log_valid(metrics)
                    if self.watch_function(metrics):
                        self.metric_data = metrics
                        self.save(path, self.iteration, epoch)

                if self.iteration >= max_iterations:
                    return
                self.iteration += 1

