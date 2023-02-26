import os
import pickle

import numpy as np
from scipy.interpolate import interp1d

sysex_header = bytes.fromhex("F0 3E 00 00")
sysex_term = bytes.fromhex("F7")

BPRD = 0x50
UWVD = 0x53

MWAVE_UWT_OFFSET = 246
M_UWT_OFFSET = 8


def load_waves_rom():
    with open("mw_waves.pickle", "rb") as f:
        waves_rom_dict = pickle.load(f)

    waves_rom_dict = {k: v for k, v in waves_rom_dict.items() if k < MWAVE_UWT_OFFSET}
    return waves_rom_dict


def convert(data, n):
    assert len(data) == n
    accum = 0
    for i, x in enumerate(data):
        assert x & 0xF0 == 0
    return sum(((x & 0xF) << (4 * (n - 1 - i))) for i, x in enumerate(data))


def WORD(data):
    return convert(data, 4)


def BYTE(data):
    return convert(data, 2)


def santize_string(string):
    escapes = {chr(char): " " for char in range(1, 32) if chr(char) != "\n"}
    translator = str.maketrans(escapes)
    return string.translate(translator)


def parse_sections(fn):
    with open(fn, "rb") as f:
        data_all = f.read()

    sections = {}

    data = bytes(data_all)
    while len(data):
        start_idx = data.index(sysex_header)
        end_idx = data.index(sysex_term) + 1
        assert start_idx == 0

        idm = data[start_idx + 4]
        sections[idm] = data[start_idx:end_idx]
        data = data[end_idx:]

    assert len(sections[BPRD]) == 11527
    assert len(sections[UWVD]) == 10887

    return sections


def extract_programs(sections):
    MWAVE_UWT_IDX = 32

    data_bprd = sections[BPRD]

    programs = []
    for i in range(64):
        data = data_bprd[180 * i : 180 * (i + 1)]
        # for some reason, some of the names end with File Separator
        name = santize_string(data[153:169].decode())
        wt_idx = data[28] - MWAVE_UWT_IDX

        if wt_idx >= 0:
            programs.append((i, name, wt_idx))
        else:
            programs.append((i, name, None))

    return programs


def extract_waves(sections):
    data_uwvd = sections[UWVD]
    data_wavetables = data_uwvd[5:3077]
    data_waves = data_uwvd[3077:10885]

    user_wavetables = {}

    for i_wt in range(11):
        user_wavetable = []
        wt_start_idx = 0x100 * i_wt
        for i_w in range(64):
            wt_word_idx = wt_start_idx + i_w * 4
            wave_idx = WORD(data_wavetables[wt_word_idx : wt_word_idx + 4])
            if 307 <= wave_idx <= 505:
                breakpoint()
            user_wavetable.append(wave_idx)
        user_wavetables[i_wt] = np.array(user_wavetable)

    user_waves = {}

    for i_w in range(61):
        user_wave = []
        w_start_idx = 0x80 * i_w
        for i_d in range(64):
            w_word_idx = w_start_idx + 2 * i_d
            user_wave.append(BYTE(data_waves[w_word_idx : w_word_idx + 2]))
        user_wave = np.array(user_wave) - 128
        user_waves[i_w + MWAVE_UWT_OFFSET] = np.concatenate(
            [user_wave, -user_wave[::-1]]
        )

    return user_wavetables, user_waves


def interpolate_uwts(wavetables_user, waves_all, used_uwts=None):
    wavetables_interpolated = {}

    for user_wt_idx, user_wt in wavetables_user.items():
        if used_uwts is not None and user_wt_idx not in used_uwts:
            continue

        wavetable = np.zeros((64, 128))

        for i, wave_num in enumerate(user_wt[:61]):
            if wave_num == 65535:
                wavetable[i, :] = None
            else:
                wavetable[i] = waves_all[wave_num]
        wavetable[61:, :] = 0

        x = np.where(~np.isnan(wavetable[:, 0]))[0]
        y = wavetable[x]
        f = interp1d(x, y, axis=0)
        wavetables_interpolated[user_wt_idx] = f(np.arange(64)).astype(int)

    return wavetables_interpolated


def save_m_wavetable(wavetables_interpolated, wt_dir, save_only=None):
    for user_wt_idx, user_wt in wavetables_interpolated.items():
        if save_only is not None and user_wt_idx not in save_only:
            continue

        fn = os.path.join(wt_dir, f"wtslot{user_wt_idx + M_UWT_OFFSET :02d}")

        m_wt = []

        for row in user_wt:
            # performs interpolation from https://gist.github.com/endolith/1297227#file-sinc_interp-py
            x = row
            s = np.arange(len(x))
            u = np.arange(2 * len(x)) / 2
            sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))

            upsampled = np.dot(x, np.sinc(sincM)).round(0).astype(int)
            upsampled = (upsampled + 128).clip(0, 255)
            m_wt += list(upsampled)

        with open(fn, "wb") as f:
            f.write(bytes(m_wt))


def BPRD_to_file(sections, fn):
    with open(fn, "wb") as f:
        f.write(sections[BPRD])


def summary(programs, mwave_fn, out_dir):
    output = ""

    output += f"Microwave I cartidge dump file: {mwave_fn}\n"
    output += f"M output directory            : {out_dir}\n"
    output += "\n"

    hd1 = "Program "
    hd2 = "Program Name"
    hd3 = "UWT "
    output += f"{hd1:13}{hd2:20}{hd3:5}\n"

    for i, name, uwt in programs:
        i = str(i)
        if uwt:
            uwt = str(uwt + M_UWT_OFFSET)
        else:
            uwt = "None"
        output += f"{i:13}{name:20}{uwt:5}\n"

    return output


def process_cart_dump(fn):
    out_dir = os.path.join(os.path.dirname(fn), os.path.basename(fn)[:-4] + " M", "")
    m_fn = os.path.join(out_dir, os.path.basename(fn)[:-4] + "_M")

    sections = parse_sections(fn)
    programs = extract_programs(sections)

    used_uwts = list(set(pgm[2] for pgm in programs if pgm[2]))

    wavetables_user, waves_user = extract_waves(sections)

    waves_all = load_waves_rom()
    waves_all.update(waves_user)

    wavetables_interpolated = interpolate_uwts(wavetables_user, waves_all, used_uwts)
    
    try:
        os.makedirs(out_dir)
    except FileExistsError as e:
        print(f"WARNING: Output directory '{e.filename}' already exists, files will be overwritten!")
        
    BPRD_to_file(sections, m_fn)
    save_m_wavetable(wavetables_interpolated, out_dir, used_uwts)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary(programs, fn, out_dir))

        