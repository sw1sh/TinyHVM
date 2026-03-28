class Device:
    DEFAULT = "METAL"
    _canonical = {"METAL": "metal", "CPU": "cpu", "metal": "metal", "cpu": "cpu"}

    @staticmethod
    def canonicalize(d: str) -> str:
        return Device._canonical.get(d, d.lower())

    @staticmethod
    def default() -> str:
        return Device.DEFAULT
