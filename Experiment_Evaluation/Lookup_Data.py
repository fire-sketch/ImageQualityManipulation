class ExperimentParameter:

    def __init__(self, name, path, factor):
        self.name = name
        self.path = path
        self.factor = factor


class Wire4mm(ExperimentParameter):
    cor = []
    slices = []

    def __init__(self, name, path, factor):
        super().__init__(name, path, factor)


class ESF(ExperimentParameter):
    cor_x = []
    cor_y = []
    slices = []

    def __init__(self, name, path, factor):
        super().__init__(name, path, factor)


class CTExperimentData(ExperimentParameter):
    def __init__(self, name, path, factor, slices, cor):
        super().__init__(name, path, factor)
        self.slices = slices
        self.cor = cor


def init_experiment_data():
    Wire4mm.cor = [234, 210]
    ESF.cor_x = [186, 231]
    ESF.cor_y = [169, 183]
    Wire4mm.slices = [49, 123]
    ESF.slices = [49, 123]


# type ESF or LSF
def get_experiment_data(root, name, operation):
    names = ['FDG_151', 'FDG_152', 'FDG_201', 'FDG_202', 'FDG_301', 'FDG_302', 'FDG_PET_15',
             'FDG_PET_20', 'FDG_PET_6', 'FDG_61', 'FDG_11', 'FDG_12', 'FDG_62', 'GA_151', 'GA_152',
             'GA_201', 'GA_202'
             ]
    factors = [-1, 1, 1, 1, 1, 1 - 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
    path = root + '/' + name
    if operation == 'LSF':
        return Wire4mm(name, path, factors[names.index(name)])
    elif operation == 'ESF':
        return ESF(name, path, factors[names.index(name)])


def get_experiment_data_ct(root, name):
    names = ['ub039', 'b039', 'ub068', 'ub077', 'b097', 'b117']
    slices = [[101, 131], [20, 50], [141, 170], [91, 119], [21, 48], [24, 52]]
    cor = [[140, 320], [138, 320], [190, 287], [210, 277], [198, 283], [217, 274]]
    index = names.index(name)
    path = root + '/' + name
    return CTExperimentData(name, path, 1, slices[index], cor[index])

