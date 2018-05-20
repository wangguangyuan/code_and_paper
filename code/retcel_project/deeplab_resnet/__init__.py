try:
    from retcel_project.deeplab_resnet.model import DeepLabResNetModel, UnetArtrous
except:
    from .model import DeepLabResNetModel, UnetArtrous
try:
    from retcel_project.deeplab_resnet.utils import decode_labels,dense_crf,inv_preprocess,prepare_label
except:
    from .utils import decode_labels, dense_crf, inv_preprocess, prepare_label