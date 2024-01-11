from utils import scroll_top
import streamlit as st
# Standard scientific Python imports
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def initialize():
    return {
        "button1": False,
        "button2": False,
        "button3": False,
        "button4": False,
        "button5": False,
        "button6": False,
        "cell1": False,
        "cell2": False,
        "cell3": False,
        "cell4": False,
        "cell5": False,
        "cell6": False,
        "digits": None,
        "class_names": None,
        "clf": None,
        "data": None,
        "predicted": None,
        "X_test": None,
        "Y_test": None,
        "reset": False
    }

def on_click():
    return True

def reset_toggle():
    for key in st.session_state.state.keys():
        if "button" in key:
            st.session_state.state[key] = False

def get_mnist_notebook():
    reset = st.sidebar.button('Reset', key='reset')
    if reset:
        reset_toggle()
        scroll_top()
        st.rerun()

    st.markdown("## Recognizing hand-written digits")
    st.markdown("This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.")
    body1 = '''# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
    # License: BSD 3 clause

    # Standard scientific Python imports
    import matplotlib.pyplot as plt

    # Import datasets, classifiers and performance metrics
    from sklearn import datasets, metrics, svm
    from sklearn.model_selection import train_test_split'''
    st.code(body=body1)
    st.session_state.state['button1'] = st.toggle("Run Code",key='Cell1',value=st.session_state.state['button1'])

    if st.session_state.state['button1']:
        # Need to define ipmorts before.
        st.session_state.state["cell1"] = True

    body2 = '''digits = datasets.load_digits()

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)'''

    st.markdown("## Digits dataset")
    st.markdown("The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.")
    st.markdown("Note: if we were working from image files (e.g., ‘png’ files), we would load them using matplotlib.pyplot.imread.")

    st.code(body=body2)
    st.session_state.state['button2'] = st.toggle("Run Code",key='Cell2',value=st.session_state.state['button2'])
    output2 = st.empty()
    if st.session_state.state['button2'] and st.session_state.state["cell1"]:
        try:
            st.session_state['digits'] = datasets.load_digits()
            digits = st.session_state['digits']
            st.session_state['class_names'] = np.unique(st.session_state['digits'])
            fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
            for ax, image, label in zip(axes, digits.images, digits.target):
                ax.set_axis_off()
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
                ax.set_title("Training: %i" % label)

            output2.pyplot(fig)
            st.session_state.state["cell2"] = True
        except:
            st.warning('Make sure you have ran previous code cells.')

    st.markdown("## Classification")
    st.markdown("To apply a classifier on this data, we need to flatten the images, turning each 2-D array of grayscale values from shape (8, 8) into shape (64,). Subsequently, the entire dataset will be of shape (n_samples, n_features), where n_samples is the number of images and n_features is the total number of pixels in each image.")
    st.markdown("We can then split the data into train and test subsets and fit a support vector classifier on the train samples. The fitted classifier can subsequently be used to predict the value of the digit for the samples in the test subset.")

    body3 = '''# flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)'''

    st.code(body=body3)
    st.session_state.state['button3'] = st.toggle('Run Code',key='Cell3',value=st.session_state.state['button3'])
    output3 = st.empty()
    if st.session_state.state['button3'] and st.session_state.state["cell1"] and st.session_state.state["cell2"]:
        try:
            # flatten the images
            n_samples = len(st.session_state['digits'].images)
            st.session_state['data'] = st.session_state['digits'].images.reshape((n_samples, -1))

            # Create a classifier: a support vector classifier
            st.session_state['clf'] = svm.SVC(gamma=0.001)

            # Split data into 50% train and 50% test subsets
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state['data'], st.session_state['digits'].target, test_size=0.5, shuffle=False
            )

            # Learn the digits on the train subset
            st.session_state['X_test'] = X_test
            st.session_state['Y_test'] = y_test
            st.session_state['clf'].fit(X_train, y_train)
            output3.write("Succesfully fit the SVM.")
            st.session_state['predicted'] = st.session_state['clf'].predict(X_test)
            st.session_state.state["cell3"] = True
        except:
            st.warning('Make sure you have ran previous code cells.')


    st.markdown("Below we visualize the first 4 test samples and show their predicted digit value in the title.")
    body4 = '''_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")'''

    st.code(body4)
    st.session_state.state['button4'] = st.toggle('Run Code',key='cell4',value=st.session_state.state['button4'])
    output4 = st.empty()
    if st.session_state.state['button4'] and st.session_state.state["cell3"] and st.session_state.state["cell1"] and st.session_state.state["cell2"]:
        try:
            fig1, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
            for ax, image, prediction in zip(axes, st.session_state['X_test'], st.session_state['predicted']):
                ax.set_axis_off()
                image = image.reshape(8, 8)
                ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
                ax.set_title(f"Prediction: {prediction}")

            output4.pyplot(fig1)
            st.session_state.state["cell4"] = True
        except:
            st.warning('Make sure you have ran previous code cells.')

    st.markdown("classification_report builds a text report showing the main classification metrics.")
    body5 = '''print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )'''

    st.code(body5)
    st.session_state.state['button5'] = st.toggle('Run Code',key='cell5',value=st.session_state.state['button5'])
    output5 = st.empty()
    if st.session_state.state['button5'] and st.session_state.state["cell4"] and st.session_state.state["cell3"] and st.session_state.state["cell1"] and st.session_state.state["cell2"]:
        try:
            output5.code(
            f"Classification report for classifier {st.session_state['clf']}:\n"
            f"{metrics.classification_report(st.session_state['Y_test'], st.session_state['predicted'])}\n"
            )
            st.session_state.state["cell5"] = True
        except:
            st.warning('Make sure you have ran previous code cells.')

    st.markdown("We can also plot a confusion matrix of the true digit values and the predicted digit values.")
    body6 = '''matrix = metrics.confusion_matrix(y_test,predicted)
        print("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()'''
    st.code(body6)
    st.session_state.state['button6']= st.toggle('Run Code',key='cell6',value=st.session_state.state['button6'])
    output6 = st.empty()
    if st.session_state.state['button6'] and st.session_state.state["cell5"] and st.session_state.state["cell4"] and st.session_state.state["cell3"] and st.session_state.state["cell1"] and st.session_state.state["cell2"]:
        try:
            matrix = metrics.confusion_matrix(st.session_state['Y_test'], st.session_state['predicted'])
            st.write("Confusion Matrix:")
            fig2, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            output6.pyplot(fig2)
        except:
            st.warning('Make sure you have ran previous code cells.')
