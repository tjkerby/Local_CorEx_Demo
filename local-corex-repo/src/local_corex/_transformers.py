from sklearn.decomposition import PCA

from .base import LinearCorex

class LatentTransformer:
    def transform(self, x):
        """
        Project input data (shape: [n_samples, n_features]) into the latent space.
        """
        raise NotImplementedError
    
    def inverse_transform(self, latent):
        """
        Reconstruct the original input from a latent representation.
        """
        raise NotImplementedError
    
    def get_component(self, component_index, latent_dim):
        """
        Return a vector relating component to the original feature space.
        """
        raise NotImplementedError

class PCAWrapper(LatentTransformer):
    def __init__(self, n_comp):
        self.pca = PCA(n_components=n_comp)
        self.n_comp = n_comp
        self.model_type = 'pca'
        
    def fit(self, x):
        self.pca.fit(x)
        
    def fit_transform(self, x):
        latents = self.pca.fit_transform(x)
        return latents

    def transform(self, x):
        return self.pca.transform(x)
    
    def inverse_transform(self, latent):
        return self.pca.inverse_transform(latent)
    
    def get_component(self, component_index):
        return self.pca.components_[component_index]

class CorExWrapper(LatentTransformer):
    def __init__(self, n_comp):
        self.corex = LinearCorex(n_hidden=n_comp, seed=42, gaussianize='outliers')
        self.n_comp = n_comp
        self.model_type = 'corex'
        
    def fit(self, x):
        self.corex.fit(x)
        
    def fit_transform(self, x):
        latents = self.corex.fit_transform(x)
        return latents
    
    def transform(self, x):
        latents = self.corex.transform(x)
        return latents
    
    def inverse_transform(self, latent):
        projected_x = self.corex.inverse_transform(latent)
        return projected_x 
    
    def get_component(self, component_index):
        return self.corex.moments['MI'][component_index]

