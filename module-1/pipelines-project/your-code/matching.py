#!/usr/bin/env python
# coding: utf-8

# # Tinder para empresas
# 
# ## Imports

# In[1]:


# Imports

import numpy as np
import pandas as pd
from scipy.spatial import distance
from random import choice, shuffle
from beautifultable import BeautifulTable


from func_encuestas_web import web_translate_csv, DIC_COMPANIES, DIC_STUDENTS


# In[2]:


# Variables


# TO DO: pasar mediante los args, por ahora se queda así. 
COMPANIES_ENCUESTAS_CSV = './encuestas/Web Company Form (Responses) .csv'
STUDENTS_ENCUESTAS_CSV = './encuestas/Web Student Form (Responses) .csv'
RONDAS = 11

COMPANIES_CLEAN_CSV = './encuestas/companies.csv'
STUDENTS_CLEAN_CSV = './encuestas/students.csv'


# In[3]:


def adquisition_survey(): 
    companies_enc = web_translate_csv(path=COMPANIES_ENCUESTAS_CSV, dic=DIC_COMPANIES)
    students_enc = web_translate_csv(path=STUDENTS_ENCUESTAS_CSV, dic=DIC_STUDENTS)
    return companies_enc, students_enc

def save_clean_dfs(): 
    pass

def main(): 
    companies_enc, students_enc = adquisition_survey()
    display(companies_enc.columns)
    

if __name__=='__main__':
    main()


# ## Tratamiento de Encuestas
# 
# 1. Company Form
# ```
# https://docs.google.com/forms/d/1JOjjmV1W-ay7h00wVtpcQzD6FHNJU8Qg6RRz1ptDN4Q/edit
# ```
# 2. Student Form
# ```
# https://docs.google.com/forms/d/1w85z6lqZdQfByTMiefqUcth16-F9NnLHuZuh9CVHxyU/edit
# ```

# ### 1. Companies

# In[4]:


companies_enc = web_translate_csv(path=COMPANIES_ENCUESTAS_CSV, dic=DIC_COMPANIES)
# salvamos el csv limpio
companies_enc.to_csv(COMPANIES_CLEAN_CSV, index=True)


# ### 2. Students

# In[5]:


students_enc = web_translate_csv(path=STUDENTS_ENCUESTAS_CSV, dic=DIC_STUDENTS)

# salvamos el csv limpio
students_enc.to_csv(STUDENTS_CLEAN_CSV, index=True)


# In[6]:


students_enc.head()


# In[7]:


companies_enc.head()


# In[8]:


# Testing
companies_enc.columns == students_enc.columns


# ## Tratado de los CSVs

# In[9]:


def csv2dataframe(filepath): 
    return pd.read_csv(filepath, sep=',', index_col=0, lineterminator='\n')

def adquisition(s_clean_csv, c_clean_csv): 
    students = csv2dataframe(s_clean_csv)
    companies = csv2dataframe(c_clean_csv)
    return students, companies

students, companies = adquisition(STUDENTS_CLEAN_CSV, COMPANIES_CLEAN_CSV)


# In[10]:


students.head()


# In[11]:


companies.head()


# In[12]:


# normalizacion. Por columnas en alumnos o empresas hay un 0.0 y un 1.0

def normalize_2dfs(students, companies): 
    # print(students.min(), companies.max())
    
    mins = [min(s, c) for s, c in zip(students.min(), companies.min())]
    students = students-mins
    companies = companies-mins
    
    # maxs = [max(s, c, 1) for s, c in zip(students.max(), companies.max())]
    maxs = [max(s, c) for s, c in zip(students.max(), companies.max())]
    students = students/maxs
    companies = companies/maxs
    
    return students, companies


# In[13]:


students_norm, companies_norm = normalize_2dfs(students, companies)


# In[14]:


# No creo que haga falta

students_norm.fillna(0, inplace=True)
companies_norm.fillna(0, inplace=True)


# In[15]:


# Matching

# PESOS = (alumnos.columns)

'''
P = [2,1,1,1,3,1,1,1,1,5, 1, 1, 1, 1, 1, 1, 1 ,1, 1, 1, 1, 1 ,1, 1, 1]
W = {col: p for col, p in zip(students_norm.columns, P)}
'''

PESOS = {
    'english': 1,
    'spanish': 1,
    'portuguese': 1,
    'french': 1,
    'dutch': 1,
    'catalan': 1,
    'location': 1, 
    'offsite': 1,
    'position': 5, # importante
    'java': 5,     # importante las técnicas
    'caspnet': 5,
    'python': 5,
    'php': 5,
    'sql': 5,
    'angular': 5,
    'vue': 5,
    'firebase': 5,
    'aws': 5,
    'dockerkubernetes': 5,
    'design': 5,
    'motivation': 1,
    'coachability': 1,
    'teamwork': 1
}


def calc_dist(s_array, c_array, weights_val): 
    ''' 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.euclidean.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
    scipy.spatial.distance.cosine
    '''
    return calc_match(distance.euclidean(s_array, c_array, weights_val))

def calc_match(distance): 
    ''' [0, 1] -> [1, 0]'''
    return 1/(1+distance)

def match(student, company, weights=PESOS): 
    student = [min(s, c) for s, c in zip(student, company)] # añadimos para que la sobrecualificación de un estudiante no penalice
    return calc_dist(student, company, list(weights.values()))


# # Emparejamiento
# 
# En cada ronda se crea una pareja alumno-empresa empezando por el mejor %. Luego se pasa al siguente alumno. Si la empresa que hacía mejor match es la del caso anterior, pasa a su segundo mejor match, etc. 
# 
# En cada ronda se cambia el estudiante que elige primero, entre los que no han ido primeros

# In[16]:


# create dataframe de tamañ0 SxC

matching = pd.DataFrame(np.zeros((len(students), len(companies))) , columns=companies.index, index=students.index)
matching.shape


# In[17]:


def calculate_match(matching, df1, df2): 
    for s in matching.index: 
        for c in matching.columns:
            try: 
                matching.loc[s, c] = match(df1.loc[s], df2.loc[c], weights=PESOS)
            except: 
                matching.loc[s, c] = 0
    return matching

matching = calculate_match(matching, students_norm, companies_norm)


# In[18]:


matching.shape


# # Grafos

# In[19]:


import networkx as nx
import matplotlib

# https://networkx.github.io/documentation/networkx-1.10/reference/introduction.html

def create_graph(df): 
    G_res=nx.Graph()
    for a in df.index: 
        for c in df.columns: 
            
            G_res.add_edge(a ,c, weight=df.loc[a][c])
            print(a , c, df.loc[a][c])
    return G_res

def plot_bipartite_graph(G, set_X, set_Y): 
    X, Y = set_X, set_Y # df1.index, df1.columns
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
    pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
    nx.draw(G, pos=pos)

    '''
def get_best_match2(G, node): 
    # sort por peso DESCENCENTE
    for edge in sorted(G.edges(data=True), key=lambda x: - x[2]['weight']):
        if node in edge: 
            return edge
'''    
# sort por peso DESCENCENTE

def get_best_match(G, node, used): 
    print(node)
    # sort por peso DESCENCENTE
    for edge in sorted(G.edges(data=True), key=lambda x: - x[2]['weight']):
        if node in edge and all(u not in edge for u in used): 
            # return edge
            res = edge
            break
    else: 
        # elegimos uno random??
        options = [(node1, node2, w) for node1, node2, w in list(G.edges(data=True)) if node1 not in used and node2 not in used]
        aleatorio = choice(options)
        # print(aleatorio)
        # print("No hay emparejamiento posible") 
        # print(used)
        
        # return aleatorio
        res = aleatorio
    
    node1, node2, w = res
    return node1, node2, w['weight']


# # favorecemos a los estudiantes. más alumnos que empresas

# In[20]:


df_pruebas = matching.copy()     

G = create_graph(df_pruebas)
students_list = df_pruebas.index
companies_list = df_pruebas.columns

# plot_bipartite_graph(G, students_list, companies_list)
# G.edges()


# In[21]:


lista = list(students_list)
shuffle(lista)
student_queue = lista * RONDAS # vamos a recorrer la lista para ir emparejando


# In[22]:


matching.head()


# In[23]:


rondas = []

for r in range(RONDAS): 
    print(r)
    aux = []
    used = []
    while len(used) < 2 * len(companies): # ahora used tiene el doble de tamaño. Hay que modificarlo
        
        # estoy hay que modificarlo si o si, no me gusta como queda
        student = student_queue.pop(0) 
        
        print(student)
        node1, node2, w = get_best_match(G, student, used) # creo que se queda sin opciones
        
        # aux.append((node1, node2, w))
        G.remove_edge(node1, node2)
        used.extend([node1, node2])
        
    
        # el nodo1 es la compañía (nodo1 no es estudiante)
        if node1 != student: 
            # used.append(node1)
            aux.append((node1, node2, w))
        # nodo2 es la compañía
        else: 
            # used.append(node2)
            aux.append((node2, node1, w))
        # print(used)
    rondas.append(aux)
        
rondas


# In[24]:


def check_rondas(rondas): 
    res = []
    for ronda in rondas: 
        companies = [company for company, student, w in ronda]
        students = [student for company, student, w in ronda]
        res.append(len(set(companies)) == len(set(students)) )
    return all(e for e in res)
            
check_rondas(rondas)            


# In[25]:


def rondas2string(rondas):
    res = ''
    for i, ronda in enumerate(rondas): 
        res += '\n\nRonda {}'.format(i)
        for c, s, w in ronda: 
            res += '\nCompany: {} -> Student: {} -> Matching: {}'.format(c, s, w)
        
    return res
        
        
text = rondas2string(rondas)

with open("Rondas.txt", "w") as f:
    f.write(text)


# In[26]:


# beautifultable

table = BeautifulTable()
table.column_headers = ["ronda", "company", "student", "match"]
for i, ronda in enumerate(rondas): 
    for row in ronda: 
        table.append_row([i, row[0], row[1], row[2]])
        
# print(table)

with open("Rondas_table.txt", "w") as f:
    f.write(str(table))


# In[27]:


# agrupadas por compañía: 

table = BeautifulTable()
table.column_headers = ["company", "ronda", "student", "match"]

for company in companies_list: 
    for i, ronda in enumerate(rondas): 
        for row in ronda: 
            if row[0] == company: 
                table.append_row([row[0], i, row[1], row[2]])
        
# print(table)

with open("Rondas_table_by_company.txt", "w") as f:
    f.write(str(table))


# In[28]:


table = BeautifulTable()
table.column_headers = ["student", "company", "ronda", "match"]

for student in students_list: 
    for i, ronda in enumerate(rondas): 
        for row in ronda: 
            if row[1] == student: 
                table.append_row([row[1], row[0], i, row[2]])
'''

'''    
with open("Rondas_table_by_student.txt", "w") as f:
    f.write(str(table))


# In[29]:


student_rounds = {}

for student in students_list: 
    total = 0
    for ronda in rondas: 
        for row in ronda: 
            if row[1] == student: 
                total += 1
    student_rounds[student] = total

student_rounds


# In[30]:


print(min(student_rounds.values()), max(student_rounds.values()))


# In[ ]:




