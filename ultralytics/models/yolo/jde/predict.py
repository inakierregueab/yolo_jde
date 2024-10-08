# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class JDEPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a joint detection and embedding model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.jde import JDEPredictor

        args = dict(model="yolov8n-jde.pt", source=ASSETS)
        predictor = JDEPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the JDEPredictor with the provided configuration, overrides, and callbacks."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "jde"
