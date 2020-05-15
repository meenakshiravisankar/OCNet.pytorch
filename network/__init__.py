from .resnet101_baseline import get_resnet101_baseline
from .resnet101_base_oc import get_resnet101_base_oc_dsn
from .resnet101_pyramid_oc import get_resnet101_pyramid_oc_dsn
from .resnet101_asp_oc import get_resnet101_asp_oc_dsn
from .resnet101_interlaced import get_resnet101_interlaced_dsn
from .resnet101_interlaced import get_resnet101_interlaced_dsn_inf
from .resnet101_base_oc import get_resnet101_base_oc_dsn_inf


networks = {
    'resnet101_baseline': get_resnet101_baseline,
    'resnet101_base_oc_dsn': get_resnet101_base_oc_dsn,
    'resnet101_pyramid_oc_dsn': get_resnet101_pyramid_oc_dsn,
    'resnet101_asp_oc_dsn': get_resnet101_asp_oc_dsn,
    'resnet101_interlaced_dsn' : get_resnet101_interlaced_dsn,
    'resnet101_interlaced_dsn_inf' : get_resnet101_interlaced_dsn_inf,
    'resnet101_base_oc_dsn_inf': get_resnet101_base_oc_dsn_inf

}

def get_segmentation_model(name, **kwargs):
    return networks[name.lower()](**kwargs)
