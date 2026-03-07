"""Compatibility factory exposing TD_FALCON naming for optimized TD-FALCON."""

from artlib.optimized.FALCONFactory import TDFALCONFactory


class TD_FALCONFactory(TDFALCONFactory):
    """Backward-compatible alias for :class:`TDFALCONFactory`."""

    pass
