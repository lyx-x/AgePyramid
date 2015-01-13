from numpy import arange, zeros, ones;
from scipy.sparse import spdiags, bmat;
from matplotlib.pyplot import figure, plot, show;

#Initialisation

a = 1;  #delta_a
t = 1;  #delta_t
T = 100;  #Temps
A = 50;  
N = (int)(A / a);  #discretisation
rho = arange(N + 1);
mu = lambda a, P : 0.0001 * a * P;
betaFunction = lambda a : (a <= 40) * (a >= 20);
initFunction = lambda a : (a <= 5) * (a >= 0);
Age = arange(N + 1) * a;
beta = betaFunction(Age);  #Taux de naissance
rho = initFunction(rho) * 1;
population = sum(rho);

#Matrice

diagonal = ones(N + 1) * (1 - t / a);
subDiag = ones(N + 1) * (t / a - mu(a * Age, population));
trans = bmat([[beta],[spdiags([subDiag, diagonal], [0, 1], N, N + 1)]]);

#Transition

figure('Evolution');
for n in range(T):
    if (n % 1 == 0):
        plot(Age, rho);
    rho = trans.dot(rho);
    population = sum(rho);
    newSubDiag = ones(N + 1) * (t / a - mu(a * Age, population));
    trans = trans + bmat([[zeros(N + 1)],[spdiags([newSubDiag - subDiag], [0], N, N + 1)]]);
    subDiag = newSubDiag;
show();




