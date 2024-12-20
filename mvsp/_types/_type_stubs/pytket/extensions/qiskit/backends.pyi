from pytket._tket.circuit import Circuit
from pytket.utils.results import KwargTypes
from pytket.backends import BackendResult
from pytket.backends.resulthandle import ResultHandle 

class AerStateBackend:
    def get_compiled_circuit(
        self, circuit: Circuit, optimisation_level: int = 2
    ) -> Circuit: ...
    def run_circuit(
        self,
        circuit: Circuit,
        n_shots: int | None = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> BackendResult: ...
    def get_state(self, circuit: Circuit) -> BackendResult: ...
    def process_circuit(self, circuit: Circuit) -> ResultHandle: ...
    def process_circuits(self, circuits: list[Circuit]) -> list[ResultHandle]: ...
    def get_result(self, handle: ResultHandle) -> BackendResult: ...
