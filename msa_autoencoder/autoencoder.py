import torch.nn as nn

class Embedding(nn.Module):
    """
    """
    def __init__(self, d_in, branches, d_out):
        """
        """
        super(Embedding, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.activation = nn.Identity()
        self.branches = branches

    
    def forward(self, sequences):
        """
        """
        sequences = sequences.reshape(sequences.shape[0], self.branches, 21)
        sequences = self.linear(sequences)
        output = self.activation(sequences)
        return output.reshape(output.shape[0], output.shape[1]*output.shape[2])
    

class deEmbedding(nn.Module):
    """
    """
    def __init__(self, d_in, branches, d_out):
        """
        """
        super(deEmbedding, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.activation = nn.Softmax(dim=2)
        self.d_in = d_in
        self.branches = branches
    
    def forward(self, sequences):
        """
        """
        sequences = sequences.reshape(sequences.shape[0], self.branches, self.d_in)
        sequences = self.linear(sequences)
        output = self.activation(sequences)
        return output.reshape(output.shape[0], output.shape[1]*output.shape[2])
    

class Autoencoder_v1(nn.Module):
    """
    """
    def __init__(self, taille, AA_latent, latent_dim=50, dropout_rate=0.3):
        """
        """
        super(Autoencoder_v1, self).__init__()
        
        self.encoder = nn.Sequential(
            Embedding(21, taille, AA_latent),
            
            nn.Linear(AA_latent*taille, AA_latent*50),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(AA_latent*50, latent_dim),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, AA_latent*50),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(AA_latent*50, AA_latent*taille),
            nn.Identity(),
            
            deEmbedding(AA_latent, taille, 21)
        )

    def forward(self, x):
        """
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def latent(self, x):
        """
        """
        return self.encoder(x)
    
    def reconstruction(self, x):
        """
        """
        return self.decoder(x)
    

class Autoencoder_v2(nn.Module):
    """
    """
    def __init__(self, taille, AA_latent, latent_dim=50, dropout_rate=0.3):
        """
        """
        super(Autoencoder_v2, self).__init__()
        
        self.encoder = nn.Sequential(
            Embedding(21, taille, AA_latent),
            
            nn.Linear(AA_latent*taille, AA_latent*100),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(AA_latent*100, latent_dim),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, AA_latent*100),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(AA_latent*100, AA_latent*taille),
            nn.Identity(),
            
            deEmbedding(AA_latent, taille, 21)
        )

    def forward(self, x):
        """
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def latent(self, x):
        """
        """
        return self.encoder(x)
    
    def reconstruction(self, x):
        """
        """
        return self.decoder(x)
    


__all__ = [
    "Embedding",
    "deEmbedding",
    "Autoencoder_v1",
    "Autoencoder_v2"
]
