MODEL_CONFIGS = {
    "rfdetr-nano":    {"class": "RFDETRNano",    "resolution": 384},
    "rfdetr-small":   {"class": "RFDETRSmall",   "resolution": 512},
    "rfdetr-medium":  {"class": "RFDETRMedium",  "resolution": 576},
    "rfdetr-large":   {"class": "RFDETRLarge",   "resolution": 704},
    "rfdetr-xlarge":  {"class": "RFDETRXLarge",  "resolution": 700},
    "rfdetr-2xlarge": {"class": "RFDETR2XLarge", "resolution": 880},
}

LARGE_MODELS = {"rfdetr-xlarge", "rfdetr-2xlarge"}


def get_model_class(name: str):
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_CONFIGS)}")
    if name in LARGE_MODELS:
        print(f"[WARNING] {name} may require 'rfdetr_plus' package. "
              "Install with: pip install rfdetr[plus]")
    import rfdetr
    cls_name = MODEL_CONFIGS[name]["class"]
    return getattr(rfdetr, cls_name)


def get_resolution(name: str) -> int:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[name]["resolution"]
