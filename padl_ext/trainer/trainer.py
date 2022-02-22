import collections
import inspect
import padl
import torch


@padl.transform
class Trainer:
    def __init__(self, train_model, optimizer, infer_model=None,
                 metrics=None, batch_size=100):
        self.train_model = train_model
        self.infer_model = infer_model
        self.optimizer = optimizer
        self.metrics = metrics if metrics is not None else {}
        self.batch_size = batch_size

    def pre_save(self, path, i):
        print(f'saving optimizer state to {path}/{i}.pt')
        torch.save(self.optimizer.state_dict(), path / f'{i}.pt')

    def post_load(self, path, i):
        sd = torch.load(path / f'{i}.pt', map_location=self.pd_device)
        print('loading optimizer from state-dict')
        self.optimizer.load_state_dict(sd)

    def __call__(self, *args, **kwargs):
        if self.pd_mode == 'infer':
            return self.infer_model.infer_apply(args, **kwargs)
        elif self.pd_mode == 'eval':
            return self.infer_model.eval_apply(args, **kwargs)
        elif self.pd_mode == 'train':
            return self.infer_model.train_apply(args, **kwargs)
        else:
            raise NotImplementedError

    def train_step(self, output):
        self.optimizer.zero_grad()
        output.backward()
        self.optimizer.step()

    def aggregate_losses(self, losses):
        all_ = [x.item() for x in losses]
        norm_factor = sum(all_)
        return [x / norm_factor for x in all_]

    def _get_parameters(self, kwargs):
        sig_params = inspect.signature(self.train_model.train_apply).parameters
        kwargs.update(
            {k: getattr(self, k) for k in dir(self) if k not in kwargs and k in sig_params}
        )
        return kwargs

    def log_train(self, it, output):
        print(f'TRAIN iteration {it}; loss {output};')

    def log_valid(self, it, metrics):
        str_ = ''.join([f'{k}: {v[-1]};' for k, v in metrics.items()])
        print(f'VALID iteration {it};' + str_)

    def save(self, path, *args, **kwargs):
        self.pd_save(path, force_overwrite=True)

    def watch_function(self, metrics):
        try:
            return metrics['loss'][-1] == min(metrics['loss'])
        except (KeyError, IndexError):
            return True

    def train(self, train_data, path, valid_data=None, metric_data=None, ground_truth=None,
              n_epochs=100, max_iterations=1000000,
              valid_interval=100, **kwargs):

        kwargs = self._get_parameters(kwargs)
        metrics = collections.defaultdict(lambda: [])

        for epoch in range(n_epochs):
            for i, output in enumerate(self.train_model.train_apply(train_data, **kwargs)):
                self.train_step(output)
                self.log_train(i, output)

                if i % valid_interval == 0 and valid_data is not None:
                    valid_loss = []
                    for output in self.train_model.eval_apply(valid_data, **kwargs):
                        valid_loss.append(output)
                    valid_loss = self.aggregate_losses(valid_loss)
                    import pdb; pdb.set_trace()
                    metrics['loss'].append(valid_loss)

                    if self.infer_model is not None and self.metrics is not None:
                        assert len(self.metrics)
                        predictions = self.infer_model(metric_data)
                        for metric in self.metrics:
                            metrics[metric].append(metric(predictions, ground_truth))

                    self.log_valid(i, metrics)

                if i >= max_iterations:
                    return

                if self.watch_function(metrics):
                    self.save(path, i, epoch)
