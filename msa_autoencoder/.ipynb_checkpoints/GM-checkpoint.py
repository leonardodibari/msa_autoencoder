import numpy as np

kb = 1.380649e-23  # Boltzmann constant in J/K

class GaussianMixture : 

    def __init__(self, weights, means, covariances) :

        self.W = weights
        self.M = means
        self.S = covariances 
        self.N = len(weights)
        self.D = means.shape[1]
        self.invS = np.array([np.linalg.inv(self.S[i]) for i in range(self.N)])
        self.detS = np.array([np.linalg.det(self.S[i]) for i in range(self.N)])

        self.x_M = np.array([])
        self.inSxM = np.array([])

        if len(self.W) != self.M.shape[0] or self.M.shape[0] != self.S.shape[0] or self.S.shape[0] != len(self.W) : 

            print(f"Different shape W : {len(self.W)} | M : {self.M.shape[0]} | S : {self.S.shape[0]}")

    def multigaussian(self, n) : 

        """Compute the value of the n-th Gaussian in the mixture"""

        norm = np.sqrt( (2 * np.pi )**self.D * self.detS[n])

        return self.W[n] * np.exp(-0.5 * self.x_M[n].T @ self.inSxM[n]) / (norm +1e-10) 
    
    def transform(self) : 

        GM = 0
        for i in range(self.N) : 
  
            norm = np.sqrt( (2 * np.pi )**self.D * self.detS[i])
            GM += self.W[i] * np.exp(-0.5 * self.x_M[i].T @ self.inSxM[i]) / (norm + 1e-10)
            
        return GM
    
    def transform2(self, x) :
        """Vectorized implementation of the potential calculation"""
        self.x_M = x - self.M
        self.inSxM = np.array([self.invS[i] @ self.x_M[i] for i in range(self.N)])
        GM = 0
        for i in range(self.N) : 
  
            norm = np.sqrt( (2 * np.pi )**self.D * self.detS[i])
            GM += self.W[i] * np.exp(-0.5 * self.x_M[i].T @ self.inSxM[i]) / (norm + 1e-10)
            
        return GM
    
    def grad(self, x) : 

        """Compute gradient"""

        grad_GM = np.zeros(x.shape)

        self.x_M = x - self.M
        self.inSxM = np.array([self.invS[i] @ self.x_M[i] for i in range(self.N)])
        
        for i in range(self.N) : 
            
            
            grad_GM += self.multigaussian(i) * self.inSxM[i] 
            
        return grad_GM / (self.transform() + 1e-10)
    
    def Overdamped_Langevin(self, x, a, T = 10, mean_noise=0, std_noise=1) : 

        """Compute the Overdamped Langevin dynamics step"""

        F = a / (200 * T) * self.grad(x) # kb ?
        Rnd = np.sqrt(a) * np.random.multivariate_normal(
                            np.ones(self.D) * mean_noise, np.identity(self.D) * std_noise)
        
        return x + F + Rnd
    

