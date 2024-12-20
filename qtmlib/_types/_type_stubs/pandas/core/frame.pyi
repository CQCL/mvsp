from typing import (
    Any,
    Union,
    Literal,
)
from pandas._typing import Dtype, SequenceNotStr
from pandas.core.arrays.base import ExtensionArray
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from numpy.typing import NDArray

FromDictOrient = Literal["columns", "index", "tight"]

class DataFrame:
    @classmethod
    def from_dict(
        cls,
        data: dict[Any, Any],
        orient: Literal["columns", "index", "tight"] = "columns",
        dtype: Dtype | None = None,
        columns: (
            Union[
                Union[
                    Union["ExtensionArray", NDArray[Any]], "Index[Any]", "Series[Any]"
                ],
                SequenceNotStr[Any],
                range,
            ]
            | None
        ) = None,
    ) -> DataFrame: ...
