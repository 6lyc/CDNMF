from sklearn.decomposition import NMF
from tqdm import tqdm
import pickle



class PreTrainer(object):

    def __init__(self, config):
        self.config = config
        self.net_shape = config['net_shape']
        self.att_shape = config['att_shape']
        self.pretrain_params_path = config['pretrain_params_path']
        self.seed = config['seed']
        self.pre_iterations = config['pre_iterations']

        self.U_init = {}
        self.V_init = {}

    def setup_z(self, i, modal):
            """
            Setup target matrix for pre-training process.
            """
            if i == 0:
                self.Z = self.A
            else:
                self.Z = self.V_init[modal + str(i-1)]

    def sklearn_pretrain(self, i):
            """
            Pretraining a single layer of the model with sklearn.
            :param i: Layer index.
            """
            nmf_model = NMF(n_components=self.layers[i],
                            init="random",
                            random_state=self.seed,
                            max_iter=self.pre_iterations)

            U = nmf_model.fit_transform(self.Z)
            V = nmf_model.components_
            return U, V

    def pre_training(self, data, module):
        self.A = data
        if module == 'net':
            self.layers = self.net_shape

        elif module == 'att':
            self.layers = self.att_shape
            """
            Pre-training each NMF layer.
            """
            print("\nLayer pre-training started. \n")

        for i in tqdm(range(len(self.layers)), desc="Layers trained: ", leave=True):
                self.setup_z(i, module)
                U, V = self.sklearn_pretrain(i)
                name = module + str(i)
                self.U_init[name] = U
                self.V_init[name] = V


        with open(self.pretrain_params_path, 'wb') as handle:
            pickle.dump([self.U_init, self.V_init], handle, protocol=pickle.HIGHEST_PROTOCOL)
