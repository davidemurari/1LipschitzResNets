from torch.nn.functional import relu
import torch
import torch.nn as nn
from scripts.utils import power_method

class NonExpansiveBlock(torch.nn.Module):
    #Gradient Step with weights initialised as orthogonal
    def __init__(self, dim_inner=10):
        super().__init__()
        self.lin = torch.nn.Linear(dim_inner, dim_inner)
        W = torch.randn(dim_inner,dim_inner)
        self.lin.weight.data = torch.linalg.matrix_exp(0.5*(W-W.T))

    def forward(self, x, tau):
        #x = x - tau ReLU(x*W^T+b^T)*W
        x = x - tau * relu(self.lin(x)) @ self.lin.weight 
        return x

class AugmentedNonExpansiveBlock(torch.nn.Module):
    # Residual layer with gradient block together with 3 more components:
    # [max(x1,x2),min(x1,x2),x3,gradientBlock(x4)]
    def __init__(self, dim_inner=10):
        super().__init__()

        self.dim_inner = dim_inner
        assert dim_inner>=3
        if dim_inner>3: #In this case there is room for the gradient
          self.base_model = NonExpansiveBlock(dim_inner-3)

    def top_part(self,x):
        #Returns [max(x1,x2),min(x1,x2),x3] for each point in the batch
        w = torch.zeros((3,1),device=x.device)
        w[0] = -1
        w[1] = 1
        return x[:,:3] - relu(x[:,:3] @ w) @ w.T

    def forward(self, x, tau):
        if self.dim_inner>3:
            #Gradient step
            lower = self.base_model(x[:,3:],tau)
            #"Ordering" of the components on top
            upper = self.top_part(x)
            return torch.cat((upper,lower),dim=1)
        else:
            #Just the "ordering" of the components
            return self.top_part(x)

class IntermediateAffine(torch.nn.Module):
    #Affine maps in between to get universality theorem 4.1
    def __init__(self, dim_inner=10):
        super().__init__()

        self.dim_inner = dim_inner
        assert dim_inner>=3
        if dim_inner>3:
            #We model the lower block as zeros cat with orthogonal
            self.S = nn.Parameter(torch.randn(dim_inner-3,dim_inner-3))

        #Bias of the affine maps, initialised as random normal
        self.bias = nn.Parameter(torch.randn(1,dim_inner))

        #Three upper rows of the matrices A1,...,AL
        #We initialise them as e1,e2,e3 and then constrain them to have at most unit ell^1 norm (and hence also sum of ell^2 norms is correct): a\in R^{h+3}:
        #|a1|+|a2|+|a3|+\|a4\|_2\leq |a1|+|a2|+|a3|+\|a4\|_1 = \|a\|_1
        e1 = torch.zeros((1,dim_inner),device=self.bias.device)
        e2 = torch.zeros((1,dim_inner),device=self.bias.device)
        e3 = torch.zeros((1,dim_inner),device=self.bias.device)
        e1[0,0] = 1.
        e2[0,1] = 1.
        e3[0,2] = 1.

        #Initialise the rows so they are trainable parameters
        self.first_row = nn.Parameter(e1)
        self.second_row = nn.Parameter(e2)
        self.third_row = nn.Parameter(e3)
        

    def normalise_row(self,row):
        with torch.no_grad():
            l1 = row[0,0].abs() + \
                row[0,1].abs() + \
                row[0,2].abs() + \
                torch.linalg.norm(row[0,3:],ord=2)
            row.data /= torch.maximum(l1, torch.tensor(1.0, device=row.device))
    
    def normalise_block(self,):
        with torch.no_grad():
            norm = torch.linalg.norm(self.S,ord=2)
            norm = torch.maximum(norm, torch.tensor(1.0,device=norm.device))
            self.S.data /= norm

    def normalise(self):
        #Normalise all the first three rows. Called after each gradient step in the training process
        self.normalise_row(self.first_row)
        self.normalise_row(self.second_row)
        self.normalise_row(self.third_row)
        self.normalise_block()

    def forward(self, x):
        #Assemble the upper part of the matrix
        mat = torch.cat((self.first_row,self.second_row,self.third_row),dim=0)
        if self.dim_inner>3:
            #Assemble the lower part of the matrix
            ZZ = torch.zeros((self.dim_inner-3,3),device=x.device)
            A = torch.linalg.matrix_exp(0.5*(self.S-self.S.T))
            bottom = torch.cat((ZZ,A),dim=1)
            #Concatenate the two horizontal blocks
            mat = torch.cat((mat,bottom),dim=0)

        return x @ mat.T + self.bias

class InvNet(torch.nn.Module):
    #Assemble the full network
    #n_blocks = # layers, 
    #dim = input dimension
    #dim_inner = number of hidden neurons
    #L = threshold for final scalar rescaling (i.e. Lip constant)
    #theorem4 = True if the network is the one for Theorem 4.1
    def __init__(
        self, 
        n_blocks=3, 
        dim=2, 
        dim_inner=10, 
        L=1., 
        theorem4=True, 
        output_dim=None
        ):
        super().__init__()
        
        self.string_descrition = "InvNet, n_blocks={}, dim={}, dim_inner={}, L={}, theorem4={}".format(
            n_blocks, dim, dim_inner, L, theorem4)
        
        self.theorem4 = theorem4  # if True, use the architecture for theorem 4.1 from the paper, else theorem 3.1
        
        self.L = L
        self.dim = dim
        self.dim_inner = dim_inner
        self.n_blocks = n_blocks
        
        #If Theorem 4.1 we use both affine and gradient blocks
        if self.theorem4:
            self.blocks = torch.nn.ModuleList([AugmentedNonExpansiveBlock(dim_inner) for _ in range(n_blocks)])
            self.affines = torch.nn.ModuleList([IntermediateAffine(dim_inner=dim_inner) for _ in range(n_blocks)])
        #If Theorem 3.1 we just use gradient blocks
        else:
            self.blocks = torch.nn.ModuleList([NonExpansiveBlock(dim_inner) for _ in range(n_blocks)])

        # Store power-method states for warm-starting
        self.singular_vectors = [None for _ in self.blocks]
        
        #Timesteps for the gradient steps
        self.taus = nn.Parameter(2.*torch.ones(n_blocks))
        
        #Output rescaling to twick the Lipschitz constant
        self.c = torch.nn.Parameter(torch.tensor(1.))

        if dim_inner > 3:
            #In this way we constrain its ell^2 norm
            self.lift_lower = nn.Linear(dim,dim_inner-3)
            if dim_inner-3==dim:
                S = torch.randn(dim_inner-3,dim_inner-3)
                R = torch.linalg.matrix_exp(0.5*(S-S.T))
                self.lift_lower.weight.data = R#torch.eye(dim_inner-3)
                self.lift_lower.bias.data *= 0.
        #Here we constrain the ell^2 norm of each row
        self.lift_upper = nn.Linear(dim,3)
        self.normalise_lifting_upper()
        
        #Linear projection layer which we constrain in ell^1 norm
        #this is done by projecting after every gradient step
        if output_dim is None:
            self.last = torch.nn.Linear(dim_inner,1,bias=False)
        else:
            self.last = lambda x : x
        
        #Warm start the power iteration method
        self.update_spectral_norms(k=10_000)
        #Clamp the scaling parameter so it is in [-L,L]
        self.clip_scaling()
    
    def normalise_lifting_upper(self):
        """Normalise the lifting upper layer."""
        #This is called after each gradient step
        with torch.no_grad():
            w = self.lift_upper.weight.data
            self.lift_upper.weight.data = w / torch.maximum(torch.linalg.norm(w, ord=2, dim=1, keepdim=True), torch.tensor(1.0, device=w.device))
    
    def increase_dimension(self,x):
        """Increase the dimension of x from dim to dim_inner."""
        if self.dim_inner > 3:
            return torch.cat((self.lift_upper(x), self.lift_lower(x)), dim=1)
        else:
            return self.lift_upper(x)

    def update_spectral_norms(self, k=1):
        """Refresh ‖Aᵢ‖₂ for each block using k power-method steps."""
        if self.dim_inner>3:
            b_op_norms, b_singular_vectors = [], []
            for block, singular_vector in zip(self.blocks, self.singular_vectors):
                if self.theorem4:
                    o, v = power_method(block.base_model.lin, u_init=singular_vector, k=k)
                else:
                    o, v = power_method(block.lin, u_init=singular_vector, k=k)
                b_op_norms.append(o)
                b_singular_vectors.append(v)
            self.op_norms, self.singular_vectors = b_op_norms, b_singular_vectors

    def clip_scaling(self):
        """Clamp c so that |c| ≤ L (done in-place)."""
        torch.clamp_(self.c.data, -self.L, self.L)

    def normalise_affines(self,):
        if self.theorem4:
            for affine in self.affines:
                affine.normalise()

    def forward(self, x):

        z = self.increase_dimension(x)
        for i, block in enumerate(self.blocks):
            tau = self.taus[i]
            if (self.dim_inner>3 and self.theorem4) or \
                not self.theorem4:
                    threshold = 2 / self.op_norms[i]**2
                    tau = torch.clamp(self.taus[i],min=0.,max=threshold)
            z = block(z, tau=tau)
            if self.theorem4:
                z = self.affines[i](z)

        return self.c * self.last(z)


class InvNetClassifier(nn.Module):
    """InvNet backbone followed by a linear classifier head."""
    def __init__(
        self, 
        n_blocks, 
        first_dim, 
        L, 
        theorem4, 
        n_classes,
        INPUT_DIM
        ):
        super().__init__()
        self.backbone = InvNet(
            n_blocks=n_blocks,
            dim=first_dim,
            dim_inner=first_dim+3,
            L=L,
            theorem4=theorem4,
            output_dim=first_dim+3
        )
        self.downsample = nn.Linear(INPUT_DIM, first_dim)
        self.classifier = nn.Linear(first_dim+3, n_classes)

    def forward(self, x):
        # x: (N, 1, 28, 28)
        x = x.view(x.size(0), -1)      # flatten to (N, 784)
        x = self.downsample(x)          # (N, hidden_dim)
        feats = self.backbone(x)       # (N, hidden_dim)
        logits = self.classifier(feats)
        return logits