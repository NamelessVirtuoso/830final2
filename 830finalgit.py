import streamlit as st 
import vega
import seaborn as sns
import altair as alt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import linear_model, decomposition, datasets
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier

def hex_to_RGB(hex_str):
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

st.markdown("<h1 style='text-align: center; color: green;'>Mushroom Classification</h1>", unsafe_allow_html=True)

st.write("Which ML method is best for defining a mushroom type?")

st.write('Take a look at the dataset')
data = pd.read_csv("https://raw.githubusercontent.com/NamelessVirtuoso/830final/main/mushrooms.csv")
df = data
st.dataframe(data.head())

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write('Take a look at the missing values')
st.write(data.isna().sum())

X = df.drop(['class'], axis=1)  
y = df["class"]

X=pd.get_dummies(X,columns=X.columns,drop_first=True)

st.write('Lets also look at the correlation heatmap.')
fig = plt.figure(figsize=(16,12))
sns.heatmap(X.corr(),linewidths=.0,cmap = "twilight_shifted")
plt.yticks(rotation=0)

st.pyplot(fig)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

sc = StandardScaler()
sc.fit(X_train)
X2 = sc.fit_transform(X)
X_test = sc.transform(X_test)


st.write("Choosing parameters for different classifiers.")
option = st.selectbox(
    'Which method would you like to see?',
    ('Logistic Regression', 'KNN', 'Tree'))

st.write('You selected:', option)

hyperparams = [0.001, 0.01, 0.1, 0.5, 1]
acc_log = []
for param in hyperparams:
    clf = linear_model.LogisticRegression(C=param)
    scores = cross_val_score(clf, X, y, cv=5)
    acc_log.append(scores.mean())

print(acc_log)



fig2 = plt.figure()
plt.plot(hyperparams,acc_log)


hyperparams2 = [1,2,3,4,5,10,15]
acc_knn = []
for param in hyperparams2:
    clf = neighbors.KNeighborsClassifier(param)
    scores = cross_val_score(clf, X, y, cv=5)
    acc_knn.append(scores.mean())

print(acc_knn)
fig3 = plt.figure()
plt.plot(hyperparams2,acc_knn)


hyperparams3 = [1,5,10,50,100]
acc_tree = []
for param in hyperparams3:
    clf = tree.DecisionTreeClassifier(max_depth=param)
    scores = cross_val_score(clf, X, y, cv=5)
    acc_tree.append(scores.mean())

print(acc_tree)
fig4 = plt.figure()
plt.plot(hyperparams3,acc_tree)

if (option =='Logistic Regression'):
    st.write("Logistic Regression model")
    st.pyplot(fig2)

if (option =='KNN'):
    st.write('KNN')
    st.pyplot(fig3)

if (option =='Tree'):
    st.write("Tree")
    st.pyplot(fig4)




tt = np.ones(10)
for i in range(1,10,1):
    clf_svm = linear_model.LogisticRegression(C=i/10)
    tt[i-1] = clf_svm.fit(X_train, y_train).score(X_test,y_test)


tt2 = np.ones(10)
for i in range(1,10,1):
    n_neighbors = i
    my_classifier = neighbors.KNeighborsClassifier(n_neighbors)
    my_model = my_classifier.fit(X_train, y_train)
    tt2[i-1] = my_model.score(X_test,y_test)

tt3 = np.ones(20)
for i in range(1,20,1):
    my_classifier = tree.DecisionTreeClassifier(max_depth = i)
    my_model = my_classifier.fit(X_train, y_train)
    k = 0
    tt3[k] = my_model.score(X_test,y_test)
    k = k+1

fig_t, axs = plt.subplots(3)
axs[0].plot(tt)
axs[0].title.set_text('gamma')
axs[1].plot(tt2)
axs[1].title.set_text('knn k=i')
axs[2].plot(tt3)
axs[2].title.set_text('Tree')


st.write("Finding the best parameter for each method")
st.pyplot(fig_t)



testAcc = [0,0,0]
test2 = X_test
x = [1,2,3]

clf = tree.DecisionTreeClassifier(max_depth = 10)
clf = clf.fit(X_train,y_train)
Y_predTest = clf.predict(test2)
testAcc[2] = accuracy_score(y_test, Y_predTest)


clf = neighbors.KNeighborsClassifier(3)
clf = clf.fit(X_train,y_train)
Y_predTest = clf.predict(test2)
testAcc[1] = accuracy_score(y_test, Y_predTest)

clf = linear_model.LogisticRegression(C=0.1)
clf = clf.fit(X_train,y_train)
Y_predTest = clf.predict(test2)
testAcc[0] = accuracy_score(y_test, Y_predTest)

testAcc

fig_cv = plt.figure()
plt.ylim([0.99, 1])
plt.bar(x = x,height = testAcc)


st.write("Finding the best parameter for each method with cross-validation")
st.pyplot(fig_cv)


tr = tree.DecisionTreeClassifier(max_depth = 10)
kn = neighbors.KNeighborsClassifier(3)
nnw = MLPClassifier(random_state=1, max_iter=300)

option2 = st.selectbox(
    'Do you want to see the the confision matrix?(True positive/false negative)',
    ("Tree","KNN","Neural Network"))

if (option2 == "Tree"):
    tr.fit(X_train, y_train);
    ConfusionMatrixDisplay.from_estimator(tr, X_test, y_test)
    fig_c = plt.show()
    st.pyplot(fig_c)
if (option2 == "KNN"):
    kn.fit(X_train, y_train);
    ConfusionMatrixDisplay.from_estimator(kn, X_test, y_test)
    fig_kn = plt.show()
    st.pyplot(fig_kn)
if (option2 == "Neural Network"):
    nnw.fit(X_train, y_train);
    ConfusionMatrixDisplay.from_estimator(nnw, X_test, y_test)
    fig_nnw = plt.show()
    st.pyplot(fig_nnw)


st.write("Here for the required interactive plot")
tst = df.iloc[1:5000,:]

chart = alt.Chart(tst, width=800, height=400).mark_point().encode(
    x='habitat',
    y='class',
    size='gill-color',
    color='cap-color',
    tooltip = ["cap-shape","cap-surface","odor","gill-size"]
).interactive()
chart


op3 = st.selectbox("Now you can choose what you want to visualize!",("Yes","No"))

if (op3 == "Yes"):
    options4 = st.multiselect(
    'What do you want to include in the plot?',
    ["cap-shape","cap-surface","odor","gill-size"])

    st.write('You selected:', options4)
    tst = df.iloc[1:5000,:]
    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"
    chart = alt.Chart(tst, width=800, height=400).mark_point(shape = hexagon).encode(
        x='habitat',
        y='class',
        size='gill-color',
     color='cap-color',
     tooltip = options4
    ).interactive()
    chart
