import json

import numpy as np

fn_upper = r"C:\share\presets\m\microwave\Waldorf_microWAVE_V2.0\microWave_2.0_H.bin"
fn_lower = r"C:\share\presets\m\microwave\Waldorf_microWAVE_V2.0\microWave_2.0_L.bin"

with open(fn_upper, "rb") as f:
    data_upper = f.read()

with open(fn_lower, "rb") as f:
    data_lower = f.read()

assert len(data_upper) == len(data_lower)

start = 43544
end = 55832
assert (end - start) / 64 == 192.0

data_waves_upper = data_upper[start:end]
data_waves_lower = data_lower[start:end]

data_waves = np.frombuffer(data_waves_upper + data_waves_lower, dtype=np.uint8)
data_waves = data_waves.reshape((-1, 64))
waves_raw = np.concatenate((data_waves, -data_waves[..., ::-1]), axis=-1)
waves_raw = waves_raw.astype(int) - 128

assert not np.any(waves_raw < -128)
assert not np.any(waves_raw > 127)

waves = {i + 122 * (i > 245): waves_raw[i] for i in range(300)}

waves_ser = {k: v.tolist() for k, v in waves.items()}
with open("m_ppg_waves.json", "w") as f:
    json.dump(waves_ser, f)
