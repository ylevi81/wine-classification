#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn import datasets as dt
from sklearn import tree as tr
import numpy as np
from sklearn import model_selection as ms
from sklearn import metrics as mt
from matplotlib import pyplot as plt


# In[2]:


import gradio as gr


# In[5]:


wine = dt.load_wine()

x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = ms.train_test_split(x,y,test_size=0.2)


# In[6]:


#definição dos parâmetros do algoritmo

wineModelTree = tr.DecisionTreeClassifier(max_depth = 10)

#treinamento do algoritmo
wineModelTree.fit(X_train,y_train)


# In[7]:


fig_m = plt.figure(tight_layout=True)
tr.plot_tree(wineModelTree, 
            feature_names= ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids',
                              'nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines',
                              'proline'],
            class_names= ['class_0', 'class_1', 'class_2'],
            filled = True
             );
plt.tight_layout()


# In[9]:


labels = wineModelTree.predict(X_test)
acc = mt.accuracy_score(y_test,labels)
print( 'Accuracy Score is: {:.2f}%'.format( 100*acc ) )


# In[10]:


def predict(*args):
    new_data = np.array([args]).reshape(1,-1)
    
    labels = wineModelTree.predict_proba(new_data)
    
    return{'Classe 0':labels[0][0], 'Classe 1': labels[0][1], 'Classe 2': labels[0][2]}


# In[11]:


def tree_plot(*args):
    
    fig_m = plt.figure(tight_layout=True)
    tr.plot_tree(wineModelTree, 
            feature_names= ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids',
                              'nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines',
                              'proline'],
            class_names= ['class_0', 'class_1', 'class_2'],
            filled = True
             );
    plt.tight_layout()
    
    return fig_m


# In[ ]:


with gr.Blocks() as demo:
    gr.Markdown('Wine Classification')
    
    with gr.Row():
        with gr.Column():
            alcohol = gr.Slider( label='alcohol', minimum=11.0, maximum=14.8, step=0.1, randomize=True)
            malic_acid = gr.Slider( label='malic_acid', minimum=0.74, maximum=5.80, step=0.1, randomize=True)
            ash = gr.Slider( label='ash',minimum=1.36, maximum=3.23, step=0.1, randomize=True)
            alcalinity_of_ash =gr.Slider( label='alcalinity_of_ash', minimum=10.6, maximum=30.0, step=1, randomize=True)
            magnesium =gr.Slider( label='magnesium', minimum=70.0, maximum=162.0, step=0.1, randomize=True)
            total_phenols =gr.Slider( label='total_phenols', minimum=0.98, maximum=3.88, step=0.1, randomize=True)
            flavanoids =gr.Slider( label='flavanoids', minimum=0.34, maximum=5.08, step=0.1, randomize=True)
            nonflavanoid_phenols =gr.Slider( label='nonflavanoid_phenols', minimum=0.13, maximum=0.66, step=0.1, randomize=True)
            proanthocyanins =gr.Slider( label='proanthocyanins', minimum=0.41, maximum=3.58, step=0.1, randomize=True)
            color_intensity =gr.Slider( label='color_intensity', minimum=1.3, maximum=13.0, step=0.1, randomize=True)
            hue =gr.Slider( label='hue', minimum=0.48, maximum=1.71, step=0.1, randomize=True)
            od315_of_diluted_wines =gr.Slider( label='od315_of_diluted_wines', minimum=1.27, maximum=4.00, step=0.1, randomize=True)
            proline =gr.Slider( label='proline', minimum=278, maximum=1680, step=0.1, randomize=True)
            
        with gr.Column():
                label = gr.Label()
                
                plot = gr.Plot()
                
                predict_btn = gr.Button(value = 'Predict')
                
                tree_btn = gr.Button( value='Tree Plot' )
                
                predict_btn.click(fn=predict, 
                          inputs=[ alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols,
                                 proanthocyanins, color_intensity, hue, od315_of_diluted_wines, proline],
                          outputs=[label],)
                
                tree_btn.click( fn=tree_plot,
                                outputs=[plot],)
    
demo.launch(debug=True, share=False)


# In[ ]:




