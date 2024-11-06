import sys
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from scipy import stats

class CEventsTable:    
    def __init__(self):
        self.table = pd.DataFrame()
    
    def appendFromCsv(self, filepath, isSignal, eventType):
        subtable = pd.read_csv(filepath, low_memory=False)
        subtable['y'] = isSignal
        subtable['type'] = eventType
        subtable['weight'] = 1.0
        
        subtable = subtable[subtable.columns.drop(list(subtable.filter(regex='Unnamed.*')))]

        if eventType == 'nonp':
            # aby nebylo timhle redukovano
            subtable['leptons_PLIVtight']=1
            subtable['weight'] = subtable['mm_weight']
            subtable.drop(columns = ['mm_weight'], inplace = True)
        
        self.table = pd.concat([self.table, subtable])
        self.table.index = range(self.table.shape[0])
        
        
    def printTypesNumbers(self, weighted=False):
        df = self.table
        
        print("{:<10} {:10} {:<10}".format('type', 'n_samples','weighted_events'))
        for t in df['type'].unique():
            print("{:<10} {:<10} {:<10}".format(t,round(df[df['type']==t].shape[0],1),round(df[df['type']==t]['weight'].sum(),1)))
            
        print("")
        print("total:", df.shape[0], df['weight'].sum())


    def subSample(self, ptype, ratio, rs = 10):
        df = self.table
        cnt =  int(df[df['type'] == ptype].shape[0] * (1-ratio))
        index = df[df['type'] == ptype].sample(cnt, random_state = rs).index
        self.table = df.drop(index = index)
        self.table.index = range(self.table.shape[0])


class MLDatasets:    
    def __init__(self, df_all, split_func, split_samples, create_val = False, rs = 10): # asi jiny pro vsechny potomky
        self.val_used = create_val
        self.ref_orig = df_all
        self.print_order = ['tH', 
                            'tt+b', 
                            'tt+c', 
                            'tt+light', 
                            'ttH', 
                            'ttZ', 
                            'ttW', 
                            'tZq', 
                            'tWZ', 
                            'single_t+W', 
                            'single_t+t', 
                            'single_t+s', 
                            'WZ', 
                            'VV']
        
        dft = df_all[[
            'rapgap_higgsb_fwdjet',
            'bbs_top_m',
            'chi2_min',
            'njets_CBT2',
            'higgs_bb_m',
            'chi2_min_tophad_m',
            'chi2_min_tophad_pt',
            'chi2_min_tophad_eta',
            'chi2_min_Whad_pt',
            'rapgap_maxptjet',
            'chi2_min_bbnonbjet_m',
            'njets_CBT5',
            'chi2_min_higgs_m',
            'chi2_min_Imvmass_tH',
            'chi2_min_higgs_pt',
            'chi2_min_top_m',
            'chi2_min_top_pt',
            'chi2_min_DeltaPhi_tH',
            'chi2_min_DeltaEta_tH',
            'sphericity',
            'inv3jets',
            'rapgap_top_fwdjet',
            'nfwdjets',
            'nnonbjets',
            'chi2_min_deltaRq1q2',
            'foxWolfram_2_momentum',
            'foxWolfram_3_momentum',
            'weight',
            'type',
            'y'
        ]]
                
        
        X_train, X_test = split_func(dft, split_samples, rs=rs)        
        
        if create_val:
            X_val, X_test = split_func(X_test,split_ratio=0.5, rs=rs)
            
            weights_val = X_val['weight']
            y_val = X_val['y']
            types_val = X_val['type']            
            X_val.drop(columns=['weight','type','y'], inplace=True)

            self.X_val, self.y_val, self.weights_val, self.types_val = X_val, y_val, weights_val, types_val

        
        y_train = X_train['y']
        y_test = X_test['y']

        weights_train = X_train['weight']
        weights_test = X_test['weight']
        
        types_train = X_train['type']
        types_test = X_test['type']

        X_train.drop(columns=['weight','type','y'], inplace=True)
        X_test.drop(columns=['weight','type','y'], inplace=True)
        
        self.X_train, self.y_train, self.weights_train, self.types_train = X_train, y_train, weights_train, types_train
        self.X_test, self.y_test, self.weights_test, self.types_test = X_test, y_test, weights_test, types_test
    
    
    def removeOutliersTrain(self, z):
        X_train = self.X_train
        
        print("with outliers:", X_train.shape)
        
        X_train = X_train[(np.abs(stats.zscore(X_train)) < z).all(axis=1)]
        
        print("without outliers:", X_train.shape)
        
        self.y_train = self.y_train.loc[X_train.index]        
        self.weights_train =  self.weights_train.loc[X_train.index]  
        self.types_train =  self.types_train.loc[X_train.index]  
                
        self.X_train = X_train
    
    
    def getTrains(self, types=False):
        return (self.X_train, self.y_train)
    
    def getTests(self, types=False):
        return (self.X_test, self.y_test)
    
    def getVals(self, types=False): 
        if self.val_used:
            return (self.X_val, self.y_val)
        else:
            print('Error: no validation set')
    
    
    def printInfo(self):
        X_train = self.X_train
        X_train['type'] = self.types_train
        X_train['weight'] = self.weights_train
        
        X_test = self.X_test
        X_test['type'] = self.types_test
        X_test['weight'] = self.weights_test

        if self.val_used:
            X_val = self.X_val
            X_val['type'] = self.types_val
            X_val['weight'] = self.weights_val
            print("total size:", X_train.shape[0] + X_test.shape[0] + X_val.shape[0] )
        else:
            print("total size",  X_train.shape[0] + X_test.shape[0])
        
        print("\nTrain size:", X_train.shape)
        
        print("{:<10} {:10} {:<10}".format('type', 'n_samples','weighted_events'))
        for t in self.print_order:
            sl = X_train[X_train['type']==t]
            
            print("{:<10} {:<10} {:<10}".format(t,sl.shape[0],round(sl['weight'].sum(),1)))
                    
        print("\nTest size:", X_test.shape)
        
        print("{:<10} {:10} {:<10}".format('type', 'n_samples','weighted_events'))
        for t in self.print_order:
            sl = X_test[X_test['type']==t]
            
            print("{:<10} {:<10} {:<10}".format(t,sl.shape[0],round(sl['weight'].sum(),1)))        
        
        
        if self.val_used:
            print("\nValidation size:", X_val.shape)
            
            print("{:<10} {:10} {:<10}".format('type', 'n_samples','weighted_events'))
            for t in self.print_order:
                sl = X_val[X_val['type']==t]

                print("{:<10} {:<10} {:<10}".format(t,sl.shape[0],round(sl['weight'].sum(),1)))        
        
        X_train.drop(columns = ['type','weight'], inplace = True)
        X_test.drop(columns = ['type','weight'], inplace = True)
        if self.val_used:
            X_val.drop(columns = ['type','weight'])
    

    def createStandartizedDatasets(self):
        # standartize balanced dataset
        # display(self.X_train.mean())
        # display(self.X_train.std())

        self.X_train_s = (self.X_train-self.X_train.mean())/self.X_train.std()
        self.X_test_s = (self.X_test-self.X_train.mean())/self.X_train.std()
        
        if self.val_used:
            self.X_val_s = (self.X_val-self.X_train.mean())/self.X_train.std() 

class CSampledDatasetAcc(MLDatasets):
    def __init__(self, df_all, n_samples, rs = 10):
        super().__init__(df_all, self.sampledDatasetAcc, n_samples, rs = rs) 
        
    @staticmethod
    def sampledDatasetAcc(df, n_samples=(20000,100000), rs=10):
        samples_s = n_samples[0]
        samples_bg = n_samples[1]
        
        sig = df[df['y']==1].sample(samples_s, random_state = rs)
        
        total_bg = df[df['y']==0].sample(frac = 1.0, random_state = rs)
        
        bgs = pd.DataFrame()
        
        for t in df['type'].unique():
            bgt = total_bg[total_bg['type']==t]
            type_size = bgt.shape[0]
            type_ratio = type_size / total_bg.shape[0]
            n_type = int(type_ratio * samples_bg)
            bgts = bgt.sample(n_type, random_state = rs)
            bgs = pd.concat([bgs, bgts])

        df_train = pd.concat([bgs, sig])
        df_train = df_train.sample(frac = 1.0, random_state = rs)
        
        df_test = df.drop(index = df_train.index)
        
        return df_train, df_test  


class CSampledDatasetExpEvents(MLDatasets):
    def __init__(self, df_all, n_samples, rs = 10):
        super().__init__(df_all, self.sampledDatasetExpEvents, n_samples, True, rs = rs) 
        
    @staticmethod
    def sampledDatasetExpEvents(df, n_samples=(20000,100000), rs=10):
        samples_s = n_samples[0]
        samples_bg = n_samples[1]

        print(df.columns)
        
        total_bg = df[df['y']==0].sample(frac = 1.0, random_state = rs)
        weight_sum = total_bg['weight'].sum()
        
        sig = df[df['y']==1].sample(samples_s, random_state = rs)        
        
        bgs = pd.DataFrame()
        
        for t in df['type'].unique():
            bgt = total_bg[total_bg['type']==t]
            type_size = bgt['weight'].sum()
            type_ratio = type_size / weight_sum
            n_type = int(type_ratio * samples_bg)
            bgts = bgt.sample(n_type, random_state = rs)
            bgs = pd.concat([bgs, bgts])

        df_train = pd.concat([bgs, sig])
        df_train = df_train.sample(frac = 1.0, random_state = rs)
        
        df_test = df.drop(index = df_train.index)
        
        return df_train, df_test  
    


class AEvaluationMetric:
    def __init__(self, filepath=None):
        if filepath == None:
            self.saved_model_names = []
            self.saved_infos = []
            self.saved_metrics = []
            self.saved_ytrues = []
            self.saved_evals = []

            self.best_model_idx = -1
            self.best_res = -1
        else:
            str_a1 = []
            str_a2 = []
            fl_a = []
            fl_aoa = []
            int_aoa = []

            best_res = -1
            best_idx = -1

            file = open(filepath, "r")
            lines = file.readlines()

            l = lines[0].replace('\n','')
            lines.pop(0)
            best_idx = int(l)  
            l = lines[0].replace('\n','')
            lines.pop(0)
            best_res = int(float(l))  

            cnt = 0
            for l in lines:     
                l = l.replace('\n','')
                if l == 'end':
                    break                

                if cnt%5 == 0:
                    str_a1.append(l)

                if cnt%5 == 1:
                    str_a2.append(l)

                if cnt%5 == 2:
                    fl_a.append(float(l))            

                if cnt%5 == 3:
                    l = l.split(',')

                    floats = []
                    for fl in l:
                        if fl == '':
                            break
                        floats.append(float(fl))
                    fl_aoa.append(floats)

                if cnt%5 == 4:
                    l = l.split(',')

                    ints = []
                    for i in l:
                        if i == '':
                            break
                        ints.append(int(i))
                    int_aoa.append(ints)

                cnt +=1

            self.saved_model_names = str_a1
            self.saved_infos = str_a2
            self.saved_metrics = fl_a
            self.saved_ytrues = int_aoa
            self.saved_evals = fl_aoa

            self.best_model_idx = best_idx
            self.best_res = best_res
            
    def saveModel(self, filepath):        
        str_a1 = self.saved_model_names
        str_a2 = self.saved_infos 
        fl_a = self.saved_metrics
        fl_aoa = self.saved_evals
        int_aoa = self.saved_ytrues

        file = open(filepath, "w")
        file.write(str(round(self.best_model_idx)))
        file.write('\n')
        file.write(str(self.best_res))
        file.write('\n')

        for m in range(len(str_a1)):
            file.write(str_a1[m])
            file.write('\n')
            file.write(str_a2[m])
            file.write('\n')
            file.write(str(fl_a[m]))
            file.write('\n')

            for f in fl_aoa[m]:
                file.write(str(round(f,8) ))
                file.write(',')

            file.write('\n')

            for i in int_aoa[m]:
                file.write(str(i))
                file.write(',')

            file.write('\n')

        file.write('end\n')

        file.close()
    


    
    
    
class CEvaluationAUC(AEvaluationMetric):
    def __init__(self, filepath=None):
        super().__init__(filepath)
        
    
    
    def showAllResults(self):        
        for i in range(len(self.saved_model_names)):
            print("Model",i,"info:")
            print(self.saved_model_names[i], self.saved_infos[i])
            self.evaluate(self.saved_evals[i], self.saved_ytrues[i], show_graph=True)
            print('\n----------------------------------------------------------\n')
        
        return        
        
            
    def showBestModel(self):
        idx = self.best_model_idx
        self.evaluate(self.saved_evals[idx], self.saved_ytrues[idx], show_graph=True)
        
    
    # ulozi vsechny parametry pro vyhodnocovaci funkci, aby mohla byt zavolana znovu 
    def saveLastResult(self, model_name, additional_info):
        self.saved_model_names.append(model_name)
        self.saved_infos.append(additional_info)
        self.saved_metrics.append(self.last_auc)
        self.saved_ytrues.append(self.last_true_label)
        self.saved_evals.append(self.last_preds)
        
        if self.last_auc > self.best_res:
            self.best_res = self.last_auc
            self.best_model_idx = len(self.saved_evals)-1
        
        return
        
    
    def evaluate(self, model_preds, true_label, print_score = True, show_graph = False):
        fpr, tpr, thresholds = skm.roc_curve(true_label, model_preds)
        auc = skm.auc(fpr, tpr)

        if show_graph == True:
            plt.figure(figsize=(10,7))
            plt.grid(True)
            plt.plot(fpr, tpr, color='red', label='ROC')
            plt.plot([0, 1], [0, 1], color='green', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic Curve')
            plt.legend()
            plt.show()        
            print('AUC of the classificator:', skm.auc(fpr, tpr))
            
        if print_score:
            print("AUC:", auc)
            
        self.last_preds = model_preds
        self.last_true_label = true_label
        self.last_auc = auc

        return auc
    
    

class CEvaluationSignificance(AEvaluationMetric):
    def __init__(self, filepath=None):
        if filepath == None:
            super().__init__()          
            self.saved_weights = []
            self.saved_types = []

            self.best_sigs = -1
            self.best_bgs = -1
        else:
            super().__init__(filepath)     
            fl_aoa = []
            str_aoa = []


            file = open(filepath + "2", "r")
            lines = file.readlines()

            cnt = 0
            for l in lines:     
                l = l.replace('\n','')
                if l == 'end':
                    break                      

                if cnt%2 == 0:
                    l = l.split(',')

                    floats = []
                    for fl in l:
                        if fl == '':
                            break
                        floats.append(float(fl))
                    fl_aoa.append(floats)

                if cnt%2 == 1:
                    l = l.split(',')

                    strs = []
                    for i in l:
                        if i == '':
                            break
                        strs.append(i)
                    str_aoa.append(strs)

                cnt +=1
                
            self.saved_types = str_aoa
            self.saved_weights = fl_aoa

    
    def showAllResults(self):        
        for i in range(len(self.saved_model_names)):
            print("Model",i,"info:")
            print(self.saved_model_names[i], self.saved_infos[i])
            self.evaluate(self.saved_evals[i], self.saved_ytrues[i], self.saved_weights[i], show_graph=True)
            print('\n----------------------------------------------------------\n')
        
        return   
            
            
            
    def showBestModel(self):
        idx = self.best_model_idx
        self.evaluate(self.saved_evals[idx], self.saved_ytrues[idx], self.saved_weights[idx], show_graph=True)
                
    
    
    # ulozi vsechny parametry pro vyhodnocovaci funkci, aby mohla byt zavolana znovu 
    def saveLastResult(self, model_name, additional_info, types):
        self.saved_model_names.append(model_name)
        self.saved_infos.append(additional_info)
        self.saved_metrics.append(self.last_significance)
        self.saved_ytrues.append(self.last_true_label)
        self.saved_evals.append(self.last_preds)
        
        self.saved_weights.append(self.last_weights)    
        self.saved_types.append(types)
        
        if self.last_significance > self.best_res:
            self.best_res = self.last_significance
            self.best_sigs = self.last_sigs
            self.best_bgs = self.last_bgs
            self.best_model_idx = len(self.saved_evals)-1
        
    
    #def evaluate(self, model_preds, true_label, print_score = True, show_graph = False):
        
        
    def saveModel(self, filepath):
        super().saveModel(filepath)
        
        print('aaa')
        fl_aoa = self.saved_weights
        string_aoa = self.saved_types

        file = open(filepath + "2" , "w")

        for m in range(len(fl_aoa)):
            for f in fl_aoa[m]:
                file.write(str(round(f,8) ))
                file.write(',')

            file.write('\n')

            for i in string_aoa[m]:
                file.write(i)
                file.write(',')

            file.write('\n')

        file.write('end\n')

        file.close()
        

        
        
    def evaluate(self, model_preds, true_label, weights, print_score = True,  show_graph = False, fast=True):
        n_boxes = 200
        step = 1/n_boxes

        x_points = []
        y_points = []

        y_points1 = []
        sum_signal = 0

        y_points0 = []
        sum_bg = 0

        best_signif = -1
        best_thr = -1
        best_sig = -1
        best_bg = -1    

        X = pd.DataFrame({'probs_s': model_preds, 'y' : true_label, 'weight' : weights })

        sf_s = 73.2/(X[X['y'] == 1]['weight'].sum())
        sf_bg = 220434.8/(X[X['y'] == 0]['weight'].sum())


        for low in np.linspace(0.0,1.0,num = n_boxes+1):
            n_backgrounds = X[(X['probs_s']>=low)&(X['y']==0)]['weight'].sum() * sf_bg
            if fast and n_backgrounds<800:
                break
            
            y_points0.append(n_backgrounds)
            
            x_points.append(low)

            n_signals = X[(X['probs_s']>=low)&(X['y']==1)]['weight'].sum() * sf_s
            y_points1.append(n_signals)

            if n_backgrounds != 0:
                significance = (n_signals) / (np.sqrt(n_backgrounds))
            else:
                significance = 0

                          

            if significance > best_signif and n_backgrounds>=800 and (abs(best_signif-significance)>0.0120):
                best_signif = significance
                best_thr = low
                best_sig = n_signals
                best_bg = n_backgrounds

            if(n_backgrounds != 0):
                y_points.append(significance) 
            else:
                y_points.append(0)


        if show_graph:
            plt.figure(figsize=(10,3.5))
            
            #plt.subplot(2,2,1)
            #plt.plot(x_points, (y_points1))
            #plt.grid(True)
            #plt.ylabel('Signal events')
            #plt.xlabel('Threshold')
            #plt.plot()

            #plt.subplot(2,2,2)
            #plt.yscale("symlog",)
            #plt.plot(x_points, (y_points0))

            #plt.grid(True)
            #plt.ylabel('Background events')
            #plt.xlabel('Threshold')
            #plt.plot()

            plt.subplot(1,2,1)
            plt.yscale("symlog",)
            plt.plot(x_points, (y_points1))
            plt.plot(x_points, (y_points0))
            plt.grid(True)
            plt.legend(['Signal','Background'])
            plt.ylabel('Expected events')
            plt.xlabel('Threshold')

            plt.subplot(1,2,2)
            plt.plot(x_points, (y_points))
            plt.grid(True)
            plt.ylabel('Significance')
            plt.xlabel('Threshold')
            plt.plot()

            plt.show()
        
        if(print_score):
            print('Best significance:',round(best_signif,3), '. Best threshold:', round(best_thr,3))
            print('Number of signals:', round(best_sig,3),'. Number of backgrounds:',round(best_bg,3))

        #return best_signif
        
        self.last_preds = model_preds
        self.last_true_label = true_label
        self.last_weights = weights
        self.last_significance = best_signif
        self.last_sigs = best_sig
        self.last_bgs = best_bg

        return (best_signif, best_thr, best_sig, best_bg)
    
    
    def showThresholdFiltration(self, model_preds, true_label, weights, types):
        n_boxes = 40
        step = 1/n_boxes

        x_points = []
        y_points1 = []
        y_points0 = []

        X = pd.DataFrame({'probs_s': model_preds, 'y' : true_label, 'weight' : weights, 'type':types})

        sf_s = 73.2/(X[X['y'] == 1]['weight'].sum())
        sf_bg = 220434.8/(X[X['y'] == 0]['weight'].sum())

        y_points0_each = pd.DataFrame(columns=types.unique())
        bg_types = X[~(X['type']=='tH')]['type'].unique()

        for low in np.linspace(0.0,1.0,num = n_boxes+1):
            x_points.append(low)

            n_signals = X[(X['probs_s']>=low)&(X['y']==1)]['weight'].sum() * sf_s
            y_points1.append(n_signals)


            for bg_name in bg_types:
                n_bg = X[X['type']==bg_name]
                n_bg = n_bg[(n_bg['probs_s']>=low)]['weight'].sum() * sf_bg
                y_points0_each.loc[low, bg_name] = n_bg            


        plt.figure(figsize=(14,10))

        print(y_points1[0])

        plt.yscale("symlog",)
        plt.plot(x_points, y_points1, label = 'tH')

        for bg_name in bg_types:
            plt.plot(x_points, y_points0_each[bg_name], label = bg_name)

        plt.legend(['tH']+list(bg_types))

        plt.grid(True)
        plt.ylabel('Background events')
        plt.xlabel('Threshold')
        plt.plot()

        # display(y_points0_each.head())
