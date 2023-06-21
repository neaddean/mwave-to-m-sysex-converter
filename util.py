import json
import os
import traceback
from itertools import product

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

sysex_header = bytes.fromhex("F0 3E 00 00")
sysex_term = bytes.fromhex("F7")

# IDM for different dump types
BPRD = 0x50
UWVD = 0x53

# microwave user waves offset into table
MWAVE_UW_OFFSET = 246
# microwave user wavetables into wave control tables
# TODO - i forgot why this is not 32 or 31
MWAVE_UWT_OFFSET = 33
# number of user wavetables on a microwave
MWAVE_NUM_UWT = 11
# offset in the M where the wavetable will point for the OSC
M_UWT_OFFSET = 8


def load_waves_rom():
    with open("m_ppg_waves.json") as f:
        waves_json = {int(k): np.array(v) for k, v in json.load(f).items()}

    waves_rom_dict = {k: v for k, v in waves_json.items() if k < MWAVE_UW_OFFSET}
    return waves_rom_dict


def convert(data, n):
    assert len(data) == n
    for i, x in enumerate(data):
        assert x & 0xF0 == 0
    return sum(((x & 0xF) << (4 * (n - 1 - i))) for i, x in enumerate(data))


# noinspection PyPep8Naming
def WORD(data):
    return convert(data, 4)


# noinspection PyPep8Naming
def BYTE(data):
    return convert(data, 2)


def sanitize_string(string):
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

    try:
        assert len(sections[BPRD]) == 11527
    except KeyError:
        raise ValueError("No BPRD section found!")

    try:
        assert len(sections[UWVD]) == 10887
    except KeyError:
        raise ValueError("No UWVD section found!")

    return sections


def extract_programs(sections):
    data_bprd = np.frombuffer(sections[BPRD], dtype=np.uint8, count=64 * 180).reshape((64, 180))
    names = [sanitize_string(s.decode()) for s in data_bprd[:, 153:169].view('S16').squeeze()]
    uwt_idxs = [x - MWAVE_UWT_OFFSET
                if (MWAVE_UWT_OFFSET + MWAVE_NUM_UWT) > x > MWAVE_UWT_OFFSET else None
                for x in data_bprd[:, 28]]

    return [(i, n, w) for i, (n, w) in enumerate(zip(names, uwt_idxs))]


def extract_waves(sections):
    data_uwvd = sections[UWVD]
    data_wavetables = data_uwvd[5:3077]
    data_waves = data_uwvd[3077:10885]

    user_wavetables = {}

    for i_wt in range(MWAVE_NUM_UWT):
        user_wavetable = []
        wt_start_idx = 0x100 * i_wt
        for i_w in range(64):
            wt_word_idx = wt_start_idx + i_w * 4
            wave_idx = WORD(data_wavetables[wt_word_idx: wt_word_idx + 4])
            if 307 <= wave_idx <= 505:
                # should never execute!
                breakpoint()
            user_wavetable.append(wave_idx)
        user_wavetables[i_wt] = np.array(user_wavetable)

    user_waves = {}

    for i_w in range(61):
        user_wave = []
        w_start_idx = 0x80 * i_w
        for i_d in range(64):
            w_word_idx = w_start_idx + 2 * i_d
            user_wave.append(BYTE(data_waves[w_word_idx: w_word_idx + 2]))
        user_wave = np.array(user_wave) - 128
        user_waves[i_w + MWAVE_UW_OFFSET] = np.concatenate([user_wave, -user_wave[::-1]])

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


def save_m_wavetable(wavetables_interpolated, wt_dir, save_only=None, *, int_interp=None):
    for user_wt_idx, user_wt in wavetables_interpolated.items():
        if save_only is not None and user_wt_idx not in save_only:
            continue

        fn = os.path.join(wt_dir, f"wtslot{user_wt_idx + M_UWT_OFFSET :02d}")

        m_wt = []

        for row in user_wt:
            x = row
            assert len(x) == 128

            if int_interp is not None:
                assert int_interp == 2
                assert int(int_interp) == int_interp
                int_interp = int(int_interp)

                # performs interpolation from https://gist.github.com/endolith/1297227#file-sinc_interp-py
                s = np.arange(len(x))
                u = np.arange(int_interp * len(x)) / int_interp
                sinc_conv = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))

                x = np.dot(x, np.sinc(sinc_conv)).round(0).astype(int)
            x = (x + 128).clip(0, 255)

            m_wt += list(x)

        with open(fn, "wb") as f:
            f.write(bytes(m_wt))


def bprd_to_file(sections, fn):
    with open(fn, "wb") as f:
        f.write(sections[BPRD])


def summary(programs, mwave_fn, out_dir):
    output = ""

    output += f"Microwave I cartridge dump file : {mwave_fn}\n"
    output += f"M output directory             : {out_dir}\n"
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


def plot_uwts(wavetables_interpolated, wavetables_user, fn):
    with PdfPages(fn) as pdf_writer:
        for wt_idx, data_interp in wavetables_interpolated.items():
            data_uwt = wavetables_user[wt_idx]

            f, axes = plt.subplots(8, 8, figsize=(15.2, 15))

            f.suptitle(f"wtslot{wt_idx + M_UWT_OFFSET :02d}",
                       x=0.1, y=.92,
                       horizontalalignment='left',
                       verticalalignment='top',
                       fontsize=15)

            interp_idxs = np.where(data_uwt == 0xFFFF)[0]

            for i, (r, c) in enumerate(product(range(8), range(8))):
                data = data_interp[i]

                ax = axes[r, c]
                if i > 60:
                    f.delaxes(ax)
                    continue

                ax.plot(data)

                ax.set_title(f"Wave {i}", c="darkgreen" if i not in interp_idxs else "gray")

                ax.set_xlim(-5, 132)
                ax.set_ylim(-133, 132)

                ax.xaxis.set_major_formatter(plt.NullFormatter())
                ax.yaxis.set_major_formatter(plt.NullFormatter())

                ax.set_xticks([0, 64, 127])
                ax.set_yticks([-128, 0, 127])

                ax.tick_params(axis="both", direction="in")

            pdf_writer.savefig(f)
            plt.close(f)


def process_cart_dump(fn, save_errors=False, make_plots=False):
    out_dir = os.path.join(os.path.dirname(fn), os.path.basename(fn)[:-4] + " M", "")
    m_fn = os.path.join(out_dir, os.path.basename(fn)[:-4] + "__SINGLE_BANK_M")

    try:
        os.makedirs(out_dir)
    except FileExistsError as e:
        print(f"WARNING: Output directory '{e.filename}' already exists, files will be overwritten!")

    try:
        sections = parse_sections(fn)
        programs = extract_programs(sections)

        used_uwts = list(set(pgm[2] for pgm in programs if pgm[2]))

        wavetables_user, waves_user = extract_waves(sections)

        waves_all = load_waves_rom()
        waves_all.update(waves_user)

        wavetables_interpolated = interpolate_uwts(wavetables_user, waves_all, used_uwts)
        assert len(wavetables_interpolated) == len(used_uwts)

        bprd_to_file(sections, m_fn)
        save_m_wavetable(wavetables_interpolated, out_dir, save_only=used_uwts, int_interp=None)

        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(summary(programs, fn, out_dir))

        if wavetables_interpolated and make_plots:
            pdf_fn = f'{os.path.basename(fn)[:-4]} UWT.pdf'
            pdf_fn = os.path.join(out_dir, pdf_fn)
            plot_uwts(wavetables_interpolated=wavetables_interpolated,
                      wavetables_user=wavetables_user,
                      fn=pdf_fn)
    except Exception as e:
        if save_errors:
            with open(os.path.join(out_dir, "summary.txt"), "w") as f:
                f.write("Exception while parsing cartridge dump:\n")
                f.write(traceback.format_exc())
        else:
            try:
                os.rmdir(out_dir)
            except Exception:
                pass
        raise e
