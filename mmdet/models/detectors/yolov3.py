from .single_stage import SingleStageDetector
from ..builder import DETECTORS


@DETECTORS.register_module
class YOLOV3(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOV3, self).__init__(backbone,
                                      neck,
                                      bbox_head,
                                      train_cfg,
                                      test_cfg,
                                      pretrained)