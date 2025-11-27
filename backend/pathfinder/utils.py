def as_int(value, default):
    try:
        return int(str(value).split()[0])
    except Exception:
        return default


def as_float(value, default):
    try:
        return float(value)
    except Exception:
        return default


def truthy(value):
    return str(value).lower() in {"1", "yes", "true"}


def to_tags(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        tags = []
        for v in value:
            if v is None:
                continue
            tags.extend(str(v).split(";"))
        return [tag.strip() for tag in tags if tag.strip()]
    return [tag.strip() for tag in str(value).split(";") if tag.strip()]
