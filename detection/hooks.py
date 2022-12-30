# eval all model weights in the $OUTPUT_DIR on test set

from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.engine import hooks

class EvalHook_(hooks.EvalHook):
    # For evaling all checkpoint in dir, only modify storage's setting
    def _do_eval(self):
        results = self._func()

        if results:
            assert isinstance(
                results, dict
            ), "Eval function must return a dict. Got {} instead.".format(results)

            flattened_results = flatten_results_dict(results)
            for k, v in flattened_results.items():
                try:
                    v = float(v)
                except Exception as e:
                    raise ValueError(
                        "[EvalHook] eval_function should return a nested dict of float. "
                        "Got '{}: {}' instead.".format(k, v)
                    ) from e
            storage = get_event_storage()

            storage.put_scalars(**flattened_results, smoothing_hint=False)
        comm.synchronize()