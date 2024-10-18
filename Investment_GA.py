#conda install -c https://conda.anaconda.org/biocore scikit-bio
#%%
import numpy as np, numpy.random
import random
import scipy.stats as ss
#%%
Total_Investment = 400000
''' Minimum investments in Banks/Business Lines '''
B1 = 500
B2 = 0
B3 = 3000
B4 = 25000
B5 = 2000
Investment = Total_Investment - (B1+B2+B3+B4+B5)
#%%
''' Number of lines of investment'''
nI = 5
''' Population in each generation'''
n = 100

''' k-way tournament selection'''
''' Enter the value of k below'''
k = 60#int(decimal/100)

'''Number of Generations'''
gen = 12

'''Probability of Cross over '''
Pc = 0.85
''' Dispersion Crossover'''
Nc = 200

''' Probability of Mutation'''
Pm = 0.05

'''Dispersion Mutation '''
Nm = 200


#%%
def pop_initialize(n,Investment):
    a = np.random.dirichlet(np.ones(n),size=1)*Investment
    return a

''' Computation of Beta which is useful in 
computation of cross_overs for offspring generation'''

def B_comp(Nc):
    u = random.uniform(0,1)
    if u <=0.5:
        beta = (2*u)**(1/(Nc+1))
    else:
        beta = (1/(2*(1-u)))**(1/(Nc+1))
    return beta

''' COmputation of Delta which is useful in 
computation of mutated offsprings'''

def D_comp(Nm):
    r=random.uniform(0,1)
    if r<0.5:
        delta = (2*r)**(1/(Nm+1))
    else:
        delta = (1/(2*(1-r)))**(1/(Nm+1))
    return delta



''' Ranking the objective value'''
def fitness_func(a):
    '''Use ss.rankdata if goal is to maximize '''
    b = ss.rankdata(a)
    '''Use len(a) - ss.rankdata if goal is to miniimize '''
    #b =len(a) - ss.rankdata(a, method="max") + 1
    b = [i/sum(b) for i in b]
    return b



''' Objective Function'''
def obj_function(array):
    ''' rij - interest/Return for ith BL '''
    B1 = 500
    B2 = 0
    B3 = 3000
    B4 = 25000
    B5 = 2000
        
    r11 = 3 # <1Lakh
    r12 = 5 # >1lakh
    
    r21 = 2.5 # <1Lakh
    r22 = 5.3 # >1lakh
    
    r31 = 4 # <1Lakh
    r32 = 3.3 # >1lakh
    
    r41 = 2.2 # <1Lakh
    r42 = 5.6 # >1lakh
    
    r51 = 3.25 # <1Lakh
    r52 = 5.55 # >1lakh
    if (array[0,0]+B1)>100000:
        SI_L1 = r11*100000/100 + r12*((array[0,0]+B1)-100000)/100    
    else:
        SI_L1 = r11*(array[0,0]+B1)/100   
        
    if (array[0,1]+B2)>100000:
        SI_L2 = r21*100000/100 + r22*((array[0,1]+B2)-100000)/100
    else:
        SI_L2 = r21*(array[0,1]+B2)/100 
        
    if (array[0,2]+B3)>100000:
        SI_L3 = r31*100000/100 + r32*((array[0,2]+B3)-100000)/100
    else:
        SI_L3 = r31*(array[0,2]+B3)/100 
        
    if (array[0,3]+B4)>100000:
        SI_L4 = r41*100000/100 + r42*((array[0,3]+B4)-100000)/100
    else:
        SI_L4 = r41*(array[0,3]+B4)/100 
        
    if (array[0,4]+B5)>100000:
        SI_L5 = r51*100000/100 + r52*((array[0,4]+B5)-100000)/100
    else:
        SI_L5 = r51*(array[0,4]+B5)/100 
        
    SI_L = SI_L1 + SI_L2 + SI_L3 + SI_L4 + SI_L5    
    return SI_L

    
''' Storing the  string of numpy array in a dictionary as a key and the corresponding fitness-values are values'''
def fitness_dict(pop,fitness):
    real_seq_fitness = {}
    for i in range(len(pop)):
        real_seq_fitness.update({str(pop[i]):fitness[i]})
    return real_seq_fitness

def tournament(real_seq,k,n,nv,real_seq_fitness):
    real_seq_mating = np.zeros((n,nI))
    for j in range(1,n):
        '''Tournament selection'''
        random_list = random.sample(range(n), k)
        real_seq2= np.zeros((k,nI))
        for i in range(len(random_list)):
            real_seq2[i] = real_seq[[random_list[i]],:]
        strings = [str(real_seq2[i]) for i in range(len(real_seq2))]
        k_real_fitness = [real_seq_fitness[strings[i]] for i in range(len(strings))]
        lll = np.array([strings[k_real_fitness.index(max(k_real_fitness))]])
        
        check = np.array([real_seq[i] for i in range(len(real_seq)) if str(real_seq[i])==lll])[0]
        real_seq_mating[j] = check    
    return real_seq_mating

'''Cross Over '''
def cross_over(nv,a,b,Pc):
    rnd_no = random.uniform(0,1)
    if rnd_no<Pc:
        beta = np.zeros((1,nv))
        for i in range(nv):
            beta[0][i] = B_comp(Nc)
        O1 = 0.5*((1+beta)*a + (1-beta)*b)
        O2 = 0.5*((1-beta)*a + (1+beta)*b)
    else:
        O1 =a
        O2 =b
    return np.vstack((O1, O2))

''' Mutation'''
def Mutation(array,Investment,Pm):
    rnd_no = random.uniform(0,1)
    if rnd_no<Pm:
        yy=np.random.normal(loc=0,scale=1,size=len(array))*Investment/250
        kk = array+yy
    else:
        kk = array
    return kk
'''
def mutation(a,diff,Nm,Pm):
    rnd_no = random.uniform(0,1)
    if rnd_no<Pm:
        b = a + diff*D_comp(Nm)
    else:
        b=a
    return b
'''
def Neg_val_bal(aa):
    for i in range(len(aa)):
        if aa[i] <0:
            aa[i]=0
    return aa

def limit_condition(a,limits,nv):
    for i in range(nv):
        if a[i] <limits[0][i]:
            a[i] = limits[0][i]
        if a[i] > limits[1][i]:
            a[i] = limits[1][i]
    return a

def sum_check(array,Investment):
    if array.sum()==Investment:
        l =0
    elif array.sum()>Investment:
        #print('changes_1')
        i = random.randrange(0,array.shape[1])
        if array[0,i]>0 and array[0,i]>abs(Investment-array.sum()):
            k = array[0,i]
            #print(i,k)
            array[0,i] = k+(Investment-array.sum())
        else:
            sum_check(array,Investment)
    elif array.sum()<Investment:
        i = random.randrange(0,array.shape[1])
        k = array[0,i]
        array[0,i]= k +(Investment-array.sum())
    return array
#%%
'''
def sum_check(array,Investment):
    if array.sum()==Investment:
        l =0
    else:
        #print('changes_1')
        i = random.randrange(0,array.shape[1])
        k = array[0,i]
        print(i,k)
        array[0,i] = k+(Investment-array.sum())         
    return array

def sum_check(array,Investment):
    if array.sum()==Investment:
        l =0
    else: 
        if array.sum()>Investment:
            #print('changes_1')
            i = random.randrange(0,array.shape[1]-1)
            array[0,i] = array[0,i]+(-Investment+array.sum())     
            
        elif array.sum()<Investment:
            #print('changes_2')
            i = random.randrange(0,array.shape[1])
            array[0,i] = array[0,i]+(-array.sum()+Investment)
    return array
'''

#%%

for i in range(gen):
  if i ==0:
    print(f'Generation:{i}')
    real_seq = np.zeros((n,nI),dtype=float)
    for i in range(len(real_seq)):
      real_seq[i] = pop_initialize(nI,Investment)
      #print(real_seq)

    for i in range(len(real_seq)):
      aa = real_seq[i]
      aa.resize(1,5)
      real_seq[i] = sum_check(aa,Investment)

    objective_val = []
    for i in range(len(real_seq)):
      aa = real_seq[i]
      aa.resize(1,5)
      objective_val.append(obj_function(aa))
    fitness = fitness_func(objective_val)
    real_seq_fitness = fitness_dict(real_seq,fitness)

    '''Elitism considering top two solutions of present generation in next generation'''
    sorted_fitness = fitness.copy()
    sorted_fitness.sort()
    first_max_second_max = sorted_fitness[-1:]
    max_2_indexes = [fitness.index(i) for i in first_max_second_max]
    max_2_array = np.zeros((1,nI))

    for i in range(len(max_2_indexes)):
      max_2_array[i] = real_seq[max_2_indexes[i]]
    
    real_seq = tournament(real_seq,k,n,nI,real_seq_fitness)

    for i in range(len(max_2_indexes)):
      real_seq[i] = max_2_array[i]

    '''Selection of successive sequences and performing the cross_over and storing them in cross_over array'''
    index = [i for i in range(len(real_seq)) if i%2==0 ]
    ''' Cross Over'''
    cross_seq = np.zeros((n,nI))

    for i in index:
      a = cross_over(nI,real_seq[i],real_seq[i+1],Pc)
      cross_seq[i] =a[0:1,:]
      cross_seq[i+1] = a[1:2,:]
    
    for i in range(len(cross_seq)):
      aa = cross_seq[i]
      cross_seq[i] = Neg_val_bal(aa)
    
    for i in range(len(cross_seq)):
      aa = cross_seq[i]
      aa.resize(1,5)
      cross_seq[i] = sum_check(aa,Investment)
      
    ''' Mutation'''
    mute_seq = np.zeros((n,nI))
    
    for i in range(len(real_seq)):
        aa = cross_seq[i]
        aa.resize(1,5)
        mute_seq[i]= Mutation(aa,Investment,Pm)
    
    for i in range(len(mute_seq)):
        aa = mute_seq[i]
        mute_seq[i] = Neg_val_bal(aa)
    
    for i in range(len(mute_seq)):
        aa = mute_seq[i]
        aa.resize(1,nI)
        mute_seq[i] = sum_check(aa,Investment)
    
    real_seq = mute_seq
    print(real_seq)
  else:
    print(f'Generation:{i}')
    for i in range(len(real_seq)):
      aa = real_seq[i]
      aa.resize(1,5)
      real_seq[i] = sum_check(aa,Investment)

    objective_val = []
    for i in range(len(real_seq)):
      aa = real_seq[i]
      aa.resize(1,5)
      objective_val.append(obj_function(aa))
    fitness = fitness_func(objective_val)
    real_seq_fitness = fitness_dict(real_seq,fitness)

    '''Elitism considering top two solutions of present generation in next generation'''
    sorted_fitness = fitness.copy()
    sorted_fitness.sort()
    first_max_second_max = sorted_fitness[-2:]
    max_2_indexes = [fitness.index(i) for i in first_max_second_max]
    max_2_array = np.zeros((2,nI))

    for i in range(len(max_2_indexes)):
      max_2_array[i] = real_seq[max_2_indexes[i]]
    
    real_seq = tournament(real_seq,k,n,nI,real_seq_fitness)

    for i in range(len(max_2_indexes)):
      real_seq[i] = max_2_array[i]

    '''Selection of successive sequences and performing the cross_over and storing them in cross_over array'''
    index = [i for i in range(len(real_seq)) if i%2==0 ]
    ''' Cross Over'''
    cross_seq = np.zeros((n,nI))

    for i in index:
      a = cross_over(nI,real_seq[i],real_seq[i+1],Pc)
      cross_seq[i] =a[0:1,:]
      cross_seq[i+1] = a[1:2,:]
    
    for i in range(len(cross_seq)):
      aa = cross_seq[i]
      cross_seq[i] = Neg_val_bal(aa)
    
    for i in range(len(cross_seq)):
      aa = cross_seq[i]
      aa.resize(1,5)
      cross_seq[i] = sum_check(aa,Investment)
      real_seq= cross_seq
      print(real_seq)
    ''' Mutation'''
    mute_seq = np.zeros((n,nI))
    for i in range(len(real_seq)):
        aa = cross_seq[i]
        aa.resize(1,5)
        mute_seq[i]= Mutation(aa,Investment,Pm)
    
    for i in range(len(mute_seq)):
        aa = mute_seq[i]
        mute_seq[i] = Neg_val_bal(aa)
    
    for i in range(len(mute_seq)):
        aa = mute_seq[i]
        aa.resize(1,nI)
        mute_seq[i] = sum_check(aa,Investment)
    
    real_seq = mute_seq
    print(real_seq)
#%%  
obj_1 = []
for i in range(len(real_seq)):
    aa = real_seq[i]
    aa.resize(1,5)
    obj_1.append(obj_function(aa))
max(obj_1)
#%%
obj_1.index(max(obj_1))
#%%
real_seq[obj_1.index(max(obj_1))]