import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import logistic
import os
import pickle

FEATURES = ["1x", "2x", "4x", "8x", "16x"]

METADATA = ["Date:", "Time:", "Measurement mode:", "Excitation wavelength:",
            "Emission wavelength:", "Excitation bandwidth:", "Emission bandwidth:",
            "Gain (Manual):", "Number of reads:", "FlashMode:", "Integration time:", "Lag time:",
            "Z-Position (Manual):"]

LABELS = {0: "Fluorescein",
          1: "Rhodamine"}

class PipetteTutorial:
    def __init__(self, excelname, save_loc, labels=2):
        excel = os.path.join(save_loc, excelname)
        self.df = pd.read_excel(excel, sheet_name=0)
        self.metadata = self.parseMetadata(labels)
        self.data = self.filterData(self.parseData(labels))
    def parseMetadata(self, labels):
        df = self.df
        all_metadata = []
        for label in range(labels):
            metadata = {}
            def parse(name):
                df2 = df.loc[df[df.columns[0]]==name].dropna(axis=1)
                return df2[df2.columns[1]].iloc[0]
            for item in METADATA:
                metadata[item] = parse(item)
            all_metadata.append(metadata)
        return all_metadata
    def parseData(self, labels):
        all_data = []
        for label in range(labels):
            data = {}
            df = self.df
            start_index = df.loc[df[df.columns[0]]=='<>'].index[label]
            end_index = start_index + 9
            df2 = df.iloc[start_index+1:end_index]
            index = 'A'
            for row in df2.iterrows():
                df_row = list(row[1][1:])
                if "..." not in df_row:
                    data[index] = df_row
                    index = chr(ord(index) + 1)
            all_data.append(pd.DataFrame(data))
        return all_data
    def dilutionLine(self, row, label, save_loc):
        Output = []
        plt.figure(figsize=(5,5))
        plt.plot(self.data[label][row])
        plt.title(row + " " + LABELS[label])
        Output.append(LABELS[label] + " " + str(row) + ": " +  str(linregress(self.data[label][row], self.data[label].index)))
        Output.append("Logistic Regression (mean, variance)" + ": " + str(logistic.fit(list(self.data[label][row]))))
        plt.savefig(os.path.join(save_loc, row + " " + LABELS[label]+"_lineplot.png"))
        plt.close()
        return Output
    def filterData(self, df_arr):
        output_arr = []
        self.norms = []
        def findNorms(df):
            norm = {}
            for norm_index in range(len(df.columns)):
                norm[df.columns[norm_index]] = np.mean([int(df.iloc[0][norm_index]), int(df.iloc[5][norm_index])])
            return norm
        for index in range(len(df_arr)):
            df = df_arr[index]
            #Label 1, Fluorescein
            if index == 0:
                df = df.drop(np.arange(6, 12))
                self.norms.append(findNorms(df))
                df.loc[0] = self.norms[0]
                df = df[:-1]
            #Label 2, Rhodamine
            elif index == 1:
                df = df.drop(np.arange(0,6))
                df = df.reset_index(drop=True)
                self.norms.append(findNorms(df))
                df.loc[0] = self.norms[1]
                df = df[:-1]
            output_arr.append(df.astype('float64'))
        return output_arr


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class NNModel:
    def __init__(self):
        self.model = None

    def formatData(self, df, cat):
        data = df.drop(cat, axis=1)
        vals = df[cat]
        return data, vals
    def MLP(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(15,10,10),
            max_iter=250)
    def RandomForest(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=2,
            random_state=0)
    def fitData(self, train, label):
        self.model.fit(train, label)

    def predict(self, df):
        return self.model.predict(df)
    def predictAnalysis(self, df, correctdf):
        predictions = self.predict(df)
        print(confusion_matrix(correctdf, predictions))
        print(classification_report(correctdf, predictions))

def runAnalysis(excelname, save_loc, model_directory):
    exp = PipetteTutorial(excelname, save_loc)
    documentation = {}
    results = {}

    #Dilution lines
    #Predict Accuracy / Problem
    fluorescein = NNModel()
    rhodamine = NNModel()
    with open('fluorescein.pkl', 'rb') as fid:
        fluorescein.model = pickle.load(fid)
    with open('rhodamine.pkl', 'rb') as fid:
        rhodamine.model = pickle.load(fid)

    prediction = {1: "Pipetting was done correctly.",
                  0: "Incorrect: Pipetting under the correct amount (not enough liquid).",
                  2: "Incorrect: Pipetting over the correct  amount (too much liquid)."}
    for label in range(2):
        index = 'A'
        for _ in range(8):
            documentation[index + " " + LABELS[label]] = exp.dilutionLine(index, label, save_loc)
            if label == 0:
                results[index + " " + LABELS[label]] = prediction[fluorescein.predict(exp.data[label][[index]].T)[0]]
            elif label == 1:
                results[index + " " + LABELS[label]] = prediction[rhodamine.predict(exp.data[label][[index]].T)[0]]
            index = chr(ord(index) + 1)

    return documentation, results, exp.metadata
