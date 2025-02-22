from ..other.fedbase import BasicServer
from ..other.fedbase import BasicClient as Client
from ...utils.fmodule import _model_to_tensor, _model_average
from ...utils.solver import NormSolver


class Server(BasicServer):

    def aggregate(self, models: list, *args, **kwargs):
        vectors = [_model_to_tensor(model) for model in models]
        norm_vectors = [v / v.norm() for v in vectors]
        sol, val = NormSolver.find_norm_element_FW(norm_vectors)
        self.model = _model_average(models, sol)