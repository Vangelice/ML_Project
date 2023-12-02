import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import array
import random
import jax
import h5py
from types import SimpleNamespace
from tqdm import tqdm



###OPTIMIZER###
class GradientDescentOptimizer:
    def init(self, parameters, **hyper_parameters):
        self.parameters = np.copy(parameters)
        self.step_size = hyper_parameters.get('step_size', 0.01)
        self.iter = 0

class Adam(GradientDescentOptimizer):
    def __init__(self, parameters, **hyper_parameters):
        GradientDescentOptimizer.init(self, parameters, **hyper_parameters)
        self.beta1 = hyper_parameters.get('beta1', 0.9)
        self.beta2 = hyper_parameters.get('beta2', 0.999)
        self.eps = hyper_parameters.get('eps', 1e-5)
        self.m = np.zeros(parameters.shape, dtype='double')
        self.v = np.zeros(parameters.shape, dtype='double')
        self.m_hat = np.zeros(parameters.shape, dtype='double')
        self.v_hat = np.zeros(parameters.shape, dtype='double')
    def step(self, gradient):
        self.iter += 1
        self.m = self.beta1 * self.m + (1-self.beta1) * gradient
        self.v = self.beta2 * self.v + (1-self.beta2) * gradient**2
        self.m_hat = self.m / (1-self.beta1**self.iter)
        self.v_hat = self.v / (1-self.beta2**self.iter)
        self.parameters -= self.step_size * self.m_hat / (np.sqrt(self.v_hat) + self.eps)


###PREPARATION###

prepare_U=np.array([[
    [1.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,1.0],
    [0.0,0.0,-1.0,0.0],
    [0.0,1.0,0.0,0.0]
],
[
    [1.0,0.0,0.0,0.0],
    [0.0,0.0,1.0,0.0],
    [0.0,0.0,0.0,1.0],
    [0.0,1.0,0.0,0.0]
],
[
    [1.0,0.0,0.0,0.0],
    [0.0,1.0,0.0,0.0],
    [0.0,0.0,1.0,0.0],
    [0.0,0.0,0.0,1.0]
]
])
measure_U=np.array([[
    [1.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,1.0],
    [0.0,0.0,-1.0,0.0],
    [0.0,1.0,0.0,0.0]
],
[
    [1.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,1.0],
    [0.0,1.0,0.0,0.0],
    [0.0,0.0,1.0,0.0]
],
[
    [1.0,0.0,0.0,0.0],
    [0.0,1.0,0.0,0.0],
    [0.0,0.0,1.0,0.0],
    [0.0,0.0,0.0,1.0]
]
])

zs =np.array([[1.0,0.0,0.0,1.0],
     [1.0,0.0,0.0,-1.0]])


def transform(x,n,base):
    ans=np.zeros(n)
    for i in range(n):
        r=(x%base)
        x=x//base
        ans[i]=r
    return ans

def constructN(params,dtype=float):
    c=np.zeros(4)
    c[1]=params[0]
    c[2]=params[1]
    c[3]=params[2]
    c[0]=1-sum(c[:])
    # Diag=fwht(c)
    temp=[[1.0,1.0,1.0,1.0],
        [1.0,1.0,-1.0,-1.0],
        [1.0,-1.0,1.0,-1.0],
        [1.0,-1.0,-1.0,1.0]
    ]
    temp=np.array(temp)
    Diag=np.matmul(temp,c)
    return np.diag(Diag).astype(float)

@jax.jit
def constructN1(params):
    tem=jnp.sum(params)
    c=jnp.insert(params,0,1-tem)
    temp=[[1.0,1.0,1.0,1.0],
        [1.0,1.0,-1.0,-1.0],
        [1.0,-1.0,1.0,-1.0],
        [1.0,-1.0,-1.0,1.0]
    ]
    temp=np.array(temp)
    Diag=jnp.matmul(temp,c)
    return jnp.diag(Diag)

def distribute(n,base,d,noise):
    b=transform(base,n,3)
    probs=np.zeros((3,2))
    prob=np.zeros(2**n)
    for i in range(3):
        state=np.matmul(prepare_U[i],zs[0])
        for t in range(d):
            state=np.dot(noise,state)
        state=np.dot(measure_U[i],state)
        probs[i,0]=np.dot(np.transpose(zs[0]),state)/2.0
        probs[i,1]=np.dot(np.transpose(zs[1]),state)/2.0
    for i in range(2**n):
        bi=transform(i,n,2)
        temp=1
        for j in range(n):
            temp*=probs[int(b[j]),int(bi[j])]
        prob[i]=temp
    return prob

def sampler(n,depth,basen,shot,noise):
    data=[]
    base=[random.randint(0,3**n-1) for i in range(basen)]
    for d in depth:
        temp=[]
        for i in range(basen):
            prob=distribute(n,base[i],d,noise)
            temp.append(random.choices(range(2**n),prob,k=shot))
        data.append(temp)
    return base, data




### SET CONFIG PARAMETERS ###
pauli = SimpleNamespace()


pauli.depth = [2, 4, 8, 16, 32]


pauli.num_bases = 60


pauli.shot = 500


pauli.batch_size = 5
## Remember to make the batch_size|shot
num_batches = pauli.shot // pauli.batch_size

pauli.n=5
pauli.true_params=np.array([0.03,0.012,0.007])
sin_noise=constructN(pauli.true_params)

###Gen data###

baselist, Data=sampler(pauli.n,pauli.depth,pauli.num_bases,pauli.shot,sin_noise)

def dist(n,base,d,noise,bit):
    b=transform(base,n,3)
    probs=[]
    for i in range(3):
        state=np.matmul(prepare_U[i],zs[0])
        for t in range(d):
            state=jnp.dot(noise,state)
        state=jnp.dot(measure_U[i],state)
        a=state[0]
        temp=[]
        temp.append(jnp.dot(np.transpose(zs[0]),state)/2.0)
        temp.append(jnp.dot(np.transpose(zs[1]),state)/2.0)
        probs.append(temp)
    bi=transform(bit,n,2)
    tem=1.0
    for j in range(n):
        tem*=probs[int(b[j])][int(bi[j])]
    return tem
def NLF(params,depth,bitlist,n,baselist):
    # p1=params[0]
    params=jnp.array(params,float)
    noise=constructN1(params)
    d=len(depth)
    nll=0
    count=0
    bn=len(baselist)
    size=len(bitlist[0][0])
    for i in range(d):
        for j in range(bn):
            for k in range(size):
                nll += -jnp.log(dist(n,baselist[j],depth[i],noise,int(bitlist[i][j][k])))
                count+=1
    return nll/count
    



### OPTIMIZE ### 
pauli.seed = 1111
rng = np.random.default_rng(pauli.seed)
pauli.initial_params = np.random.normal(0.1,0.03,len(pauli.true_params))
pauli.step_size_schedule = [(2, .003),(3,0.002),(3,0.0001)]
opt = Adam(pauli.initial_params)
loss_history=[]
param_history=[]
grad_history =[]
epoch_counter = 0
data_indeces = np.arange(pauli.shot)
try:
    for num_epochs, step_size in pauli.step_size_schedule:
        opt.step_size = step_size
        for _ in range(num_epochs):
            rng.shuffle(data_indeces)
            for indeces in tqdm(data_indeces.reshape(num_batches, pauli.batch_size)):
                v, g = jax.value_and_grad(NLF)(opt.parameters, pauli.depth, np.array(Data)[:,:,indeces].tolist(), pauli.n, baselist)
                loss_history.append(v)
                param_history.append(opt.parameters)
                grad_history.append(g)
                opt.step(g)
            epoch_counter += 1
            print(param_history[-1])
            print(f'Finished epoch {epoch_counter}: error = {np.linalg.norm(param_history[-1] - pauli.true_params)}')
finally:
    ### SAVE RESULTS ###
    with h5py.File('results.hdf5', 'w') as f:
        f.create_dataset('loss_history', data=loss_history)
        f.create_dataset('grad_history', data=grad_history)
        f.create_dataset('param_history', data=param_history)
        for key, value in pauli.__dict__.items():  # dump meta parameters into file
            f.attrs[key] = value
