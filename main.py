import pandas as pd
df=pd.read_csv('bioactivity_preprocessed_data.csv')
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


def lipinski(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData = np.arange(1, 1)
    i = 0
    for mol in moldata:

        desc_MolWt = Descriptors.MolWt(mol)
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_NumHDonors = Lipinski.NumHDonors(mol)
        desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

        row = np.array([desc_MolWt,
                        desc_MolLogP,
                        desc_NumHDonors,
                        desc_NumHAcceptors])

        if (i == 0):
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i + 1

    columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)

    return descriptors
df_lipinski=lipinski(df.canonical_smiles)
df_combined=pd.concat([df,df_lipinski],axis=1)
print(df_combined)
def pIC50(input):
    pIC50=[]
    for i in input['standard_value_norm']:
        molar = i*(10**-9)
        pIC50.append(-np.log10(molar))
    input['pIC50']=pIC50
    x=input.drop('standard_value_norm',1)
    return x
def norm_value(input):
    norm = []
    for i in input['standard_value']:
        if i>100000000:
            i=100000000
        norm.append(i)
    input['standard_value_norm']=norm
    x=input.drop('standard_value',1)
    return x
df_norm=norm_value(df_combined)
df_final=pIC50(df_norm)
df_2class=df_final[df_final.bioactivity_class != 'intermediate']
df_2class.to_csv('processed.csv',index=False)
import seaborn as sn
sn.set(style='ticks')
import matplotlib.pyplot as plt
plt.figure(figsize=(5.5,5.5))
sn.countplot(x='bioactivity_class',data=df_2class,edgecolor='black')
plt.xlabel('Bioactivity class',fontsize=14,fontweight='bold')
plt.ylabel('Frequency',fontsize=14,fontweight='bold')
plt.savefig('plt_bioactivity_class.pdf')
plt.figure(figsize=(5.5, 5.5))

sn.scatterplot(x='MW', y='LogP', data=df_2class, hue='bioactivity_class', size='pIC50', edgecolor='black', alpha=0.7)

plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0)
plt.show()
plt.savefig('plot_MW_vs_LogP.pdf')
plt.figure(figsize=(5.5, 5.5))

sn.boxplot(x = 'bioactivity_class', y = 'pIC50', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')
plt.show()
plt.savefig('plot_ic50.pdf')


def mannwhitney(descriptor, verbose=False):
    # https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
    from numpy.random import seed
    from numpy.random import randn
    from scipy.stats import mannwhitneyu

    # seed the random number generator
    seed(1)

    # actives and inactives
    selection = [descriptor, 'bioactivity_class']
    df = df_2class[selection]
    active = df[df.bioactivity_class == 'active']
    active = active[descriptor]

    selection = [descriptor, 'bioactivity_class']
    df = df_2class[selection]
    inactive = df[df.bioactivity_class == 'inactive']
    inactive = inactive[descriptor]

    # compare samples
    stat, p = mannwhitneyu(active, inactive)
    # print('Statistics=%.3f, p=%.3f' % (stat, p))

    # interpret
    alpha = 0.05
    if p > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distribution (reject H0)'

    results = pd.DataFrame({'Descriptor': descriptor,
                            'Statistics': stat,
                            'p': p,
                            'alpha': alpha,
                            'Interpretation': interpretation}, index=[0])
    filename = 'mannwhitneyu_' + descriptor + '.csv'
    results.to_csv(filename)
    return results
mannwhitney('pIC50')
plt.figure(figsize=(5.5, 5.5))

sn.boxplot(x = 'bioactivity_class', y = 'MW', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')

plt.savefig('plot_MW.pdf')
plt.show()
mannwhitney('MW')
plt.figure(figsize=(5.5, 5.5))

sn.boxplot(x = 'bioactivity_class', y = 'LogP', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')

plt.savefig('plot_LogP.pdf')
plt.show()
mannwhitney('LogP')
plt.figure(figsize=(5.5, 5.5))

sn.boxplot(x = 'bioactivity_class', y = 'NumHDonors', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')

plt.savefig('plot_NumHDonors.pdf')
plt.show()
mannwhitney('NumHDonors')
plt.figure(figsize=(5.5, 5.5))

sn.boxplot(x = 'bioactivity_class', y = 'NumHAcceptors', data = df_2class)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')

plt.savefig('plot_NumHAcceptors.pdf')
plt.show()
mannwhitney('NumHAcceptors')
import pandas as pd
df3=pd.read_csv('processed.csv')
selection = ['canonical_smiles','molecule_chembl_id']
df3_selection = df3[selection]
df3_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)
df3_X=pd.read_csv('descriptors_output.csv')
df3_X = df3_X.drop(columns=['Name'])
df3_Y = df3['pIC50']
print(df3_Y)
dataset3 = pd.concat([df3_X,df3_Y], axis=1)
print(dataset3)
dataset3.to_csv('data.csv', index=False)
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
dff=pd.read_csv('data.csv')
X = dff.drop('pIC50', axis=1)
Y = dff.pIC50
print(X.shape)
from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = selection.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)
r2 = model.score(X_test, Y_test)
Y_pred = model.predict(X_test)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(color_codes=True)
sns.set_style("white")

ax = sns.regplot(Y_test, Y_pred, scatter_kws={'alpha':0.4})
ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.figure.set_size_inches(5, 5)
plt.show()
