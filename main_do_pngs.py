from Patient import PatientDataBase

# mods: gaussian, rectangle, noise, noise_gauss
mods = 'noise_gauss'
DATA_PATH_IN = "../data/input_data"
DATA_PATH_OUT = "../data/output_data/" + mods
ct_data_out = "../data/output_data/png_org"


if __name__ == '__main__':
    print('start initialization')
    data_base = PatientDataBase(DATA_PATH_IN)
    gen = data_base.patient_generator()

    widths_gauss = [10]
    widths_rectangle = [2, 4, 6, 8, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]

    filtering = None
    if mods == 'gaussian' or mods == 'noise_gauss':
        filtering = 'gaussian'
    elif mods == 'rectangle':
        filtering = 'rectangle'
    elif not mods == 'noise':
        raise ValueError('mods not supported')

    for pat in gen:
        if pat == 'zzzCFPatient00':
            print(pat.id)
            for w in widths_gauss:
                pat.convolve_with_filter(w)
                pat.write_modified_as_png(safe_original=False)
            gen.__next__()
