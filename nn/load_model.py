import os
import sys


def load_model(model, model_path, flag, model_type):
    directories = os.listdir(model_path)
    submodel_names = ['generator', 'discriminator']
    submodels = [model.generator, model.discriminator]
    for i in range(2):
        submodel_name = submodel_names[i]
        submodel = submodels[i]

        filter_directories = list(filter(lambda x: x.find(flag) >= 0 and x.find(model_type) >= 0 and x.find(submodel_name) >= 0, directories))

        filter_directories.sort(reverse=False)
        load_model_name = filter_directories[-1][:-6]
        sys.stdout.write('load_model_name: ')
        sys.stdout.write(load_model_name)
        sys.stdout.write('\n')
        submodel.load_weights(model_path + '/' + load_model_name).expect_partial()

    return model

