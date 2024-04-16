import operator
from functools import reduce
from typing import Optional, List, Union, Callable, Any

import torch
from ignite.contrib.handlers import ClearMLLogger
from ignite.contrib.handlers.base_logger import BaseHandler
from ignite.engine import Engine, Events
from torch import nn


class SpikeDensityHandler(BaseHandler):
    def __init__(self,
                 model: nn.Module,
                 tag: Optional[str] = None,
                 whitelist: Optional[Union[List[str], Callable[[str, nn.Parameter], bool]]] = None,
                 ):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Argument model should be of type torch.nn.Module, but given {type(model)}")

        self.model = model
        self.tag = tag

        named_modules = {}
        if whitelist is None:
            named_modules = dict(model.named_modules())
        elif callable(whitelist):
            for n, m in model.named_modules():
                if whitelist(n, m):
                    named_modules[n] = m
        else:
            for n, m in model.named_modules():
                for item in whitelist:
                    if n.startswith(item):
                        named_modules[n] = m

        self.named_modules = named_modules

    def __call__(self, engine: Engine, logger: Any, event_name: Union[str, Events]) -> None:
        if not isinstance(logger, ClearMLLogger):
            raise RuntimeError("Handler WeightsScalarHandler works only with ClearMLLogger")

        global_step = engine.state.get_event_attrib_value(event_name)
        tag_prefix = f"{self.tag}/" if self.tag else ""
        for name, module in self.named_modules.items():
            if (hasattr(module, "log_spikes")):
                if module.log_spikes:
                    if (hasattr(module, "spike_log")):
                        spks = torch.stack(module.spike_log, dim=0)
                        spike_density = torch.sum(spks) / reduce(operator.mul, spks.shape)
                        module.spike_log = []
                        title_name, _, series_name = name.partition(".")
                        logger.clearml_logger.report_scalar(
                            title=f"{tag_prefix}spike_density/{title_name}",
                            series=series_name,
                            value=spike_density,
                            iteration=global_step,
                        )
                    else:
                        raise NotImplementedError("in order to log spikes, the layer needs a 'spike_log' attribute")
