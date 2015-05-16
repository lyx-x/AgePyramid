import numpy as np
import scipy.sparse as sp
import pylab as pl

#Initialisation

delta_a = 1;  #delta_a
delta_t = 0.1;  #delta_t
Temps = 10000;  #Temps
T = (int)(Temps / delta_t);  #Cycle
A = 100;  
N = (int)(A / delta_a);  #discretisation

mu = lambda a, P : 0.001 * a * P;
#mu = lambda a, P : P / (A - a);
betaFunction = lambda a : (a <= 40) * (a >= 20);
initFunction = lambda a : (a <= 5) * (a >= 0);  #Population initiale

Age = np.arange(N + 1) * delta_a;
rho = np.arange(N + 1) * delta_a;
rho = initFunction(rho) * 1;
population = sum(rho) * delta_a;


#Matrice

beta = betaFunction(Age) * delta_a;  #Taux de naissance, première ligne
diagonal = np.ones(N + 1) * (1 - delta_t / delta_a);
subDiag = np.ones(N + 1) * (delta_t / delta_a - mu(delta_a * Age - delta_a, population) * delta_t);
trans = sp.bmat([[beta],[sp.spdiags([subDiag, diagonal], [0, 1], N, N + 1)]]);

#Evolution

pl.ion()
line, = pl.plot(Age, rho) 

for n in range(T):
    if (n % 1 == 0):
        line.set_ydata(rho)
        pl.draw() 
    rho = trans.dot(rho);
    population = sum(rho) * delta_a;
    newSubDiag = np.ones(N + 1) * (delta_t / delta_a - mu(delta_a * Age - delta_a, population) * delta_t);
    trans = trans + sp.bmat([[np.zeros(N + 1)],[sp.spdiags([newSubDiag - subDiag], [0], N, N + 1)]]);  #Nouvelle matrice
    subDiag = newSubDiag;

pl.ioff()
pl.show()

