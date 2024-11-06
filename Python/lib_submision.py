import pandas as pd
import numpy as np
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from scipy import stats
import sys
import time
import matplotlib
import xgboost as xgb
import seaborn as s



class CEventsTable:    
    def __init__(self):
        self.table = pd.DataFrame()
    
    def appendFromCsv(self, filepath, isSignal, eventType):
        subtable = pd.read_csv(filepath)
        subtable['y'] = isSignal
        subtable['type'] = eventType
        subtable['weight'] = 0
        
        
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
    
    def applyPreselection(self, info = True):
        df = self.table
        
        if info:        
            print('Before selection')
            print('signal samples:', df[df['y']==1].shape[0])
            print('background samples:', df[df['y']==0].shape[0])
            print('signal/background samples ratio:',df[df['y']==1].shape[0]/df[df['y']==0].shape[0])
        
        
        df = df[(df['tau_pt']<1) & (df['leptons_PLIVtight']>0)]
        df = df[~((df['njets']>=5) & (df['nbjets']>=4))]
        
        if info:
            print('\nAfter selection:')
            print('signal samples:', df[df['y']==1].shape[0])
            print('background samples:', df[df['y']==0].shape[0])
            print('signal/background samples ratio:',df[df['y']==1].shape[0]/df[df['y']==0].shape[0])
            
        self.table = df
        
        print("")

    def calculateWeights(self, info = True):
        df = self.table
        
        #save nonp weights, assign later
        nonp_weights = df[df['type']=='nonp']['weight']
        
        df['luminosity'] = -1
        df.loc[(df['runNumber']<290000),'luminosity'] = 36207.66
        df.loc[(df['runNumber']>=290000)&(df['runNumber']<310000),'luminosity'] = 44307.4
        df.loc[(df['runNumber']>=310000),'luminosity'] = 58450.1

        df['weight'] = df['luminosity']*df['weight_mc'] * df['xsec_weight'] * df['weight_pileup'] * df['weight_bTagSF_DL1r_Continuous'] * df['weight_jvt'] * df['weight_forwardjvt']*df['weight_leptonSF']
        df['weight'] /= df['totalEventsWeighted']
        
        df.loc[df['type']=='nonp', 'weight'] = nonp_weights
        
        if info:
            sums = 0

            for t in df['type'].unique():
                summ = round(df[df['type']==t]['weight'].sum(),1)
                print('Type:', t, summ)
                                
                sums += summ

            print('total:')
            
        self.table.index = range(self.table.shape[0])
        print("")
        
        
    def subSample(self, ptype, ratio, rs = 10):
        df = self.table
        cnt =  int(df[df['type'] == ptype].shape[0] * (1-ratio))
        index = df[df['type'] == ptype].sample(cnt, random_state = rs).index
        self.table = df.drop(index = index)
        self.table.index = range(self.table.shape[0])


        
        
# CREATING ML DATASET
        
        
        
class MLDatasets:    
    def __init__(self, df_all, split_func, split_samples, create_val = False, rs = 10): # asi jiny pro vsechny potomky
        self.val_used = create_val
        self.ref_orig = df_all
        self.print_order = ['tH', 'ttb', 'ttc', 'ttL', 'ttH', 'ttZ', 'ttW', 'tZq', 'tWZ',
                            'single_tW', 'single_tt', 'single_ts', 'WZ', 'VV', 'nonp']
        
        dft = df_all[['njets_CBT5','nnonbjets','sphericity','aplanarity','nonbjets_eta',
          'rapgap_top_fwdjet','fwdjets_pt','chi2_min_DeltaEta_tH','tagnonb_eta',
          'tagnonb_topb_m','nfwdjets','chi2_min_tophad_m_ttAll','rapgap_maxptjet',
          'inv3jets','nbjets','chi2_min_toplep_pt','nonbjets_pt','chi2_min_deltaRq1q2',
          'chi2_min_Whad_m_ttAll','leptons_charge','foxWolfram_2_momentum','chi2_min_Imvmass_tH',
          'chi2_min_bbnonbjet_m','chi2_min_higgs_m','weight','type','y']]
                
        #X_train, X_test = createDfRatioAll(dft, split_ratio=split_ratio, rs=10)
        
        X_train, X_test = split_func(dft, split_samples, rs=rs)
        
        #if split_ratio == None:
        #    X_train, X_test = split_func(dft, n_samples=n_samples, rs=rs)
        #else:
        #    X_train, X_test = split_func(dft, split_ratio=split_ratio, rs=rs)
        #X_train, X_test = bgVarious(dft, split_ratio=split_ratio, rs=10)

        
        
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
        
        
        
class CCreate5CatDatatet(MLDatasets):
    def __init__(self, df, split_ratio, rs = 10):
        super().__init__(df, self.createDatasets, split_ratio,  rs = 10)

    @staticmethod
    def createDatasets(df, split_ratio = 0.8, rs = 10):
        df['y2'] = df['type'].copy()
        i = df[(df['y2']!='tH')&(df['y2']!='ttb')&(df['y2']!='ttc')&(df['y2']!='ttL')].index
        print(len(i))
        df.loc[i, 'y2'] = 'other'

        X_train = pd.DataFrame(columns = df.columns)
        y_train = []

        X_test = pd.DataFrame(columns = df.columns)
        y_test = []
        

        for y_type in df['y2'].unique():            
            print(y_type)
            s = df[df['y2'] == y_type]
            print(s.shape[0])

            size = int(s.shape[0] * split_ratio)

            X_train = pd.concat([X_train, s.iloc[:size,:]], ignore_index=True)
            X_test = pd.concat([X_test, s.iloc[size:,:]], ignore_index=True)
            print(X_train.shape[0])
            print(X_test.shape[0])

        X_train = X_train.sample(frac=1, random_state = rs, ignore_index=True)
        X_test = X_test.sample(frac=1, random_state = rs, ignore_index=True)

        X_train.drop(columns = ['y2'], inplace = True)
        X_test.drop(columns = ['y2'], inplace = True)
        #y_train = X_train['y']
        #X_train.drop(columns = ['y'], inplace=True)
        #y_test = X_test['y']
        #X_test.drop(columns = ['y'], inplace=True)
        # display(X_train.head(20))
        # display(X_test.head(20))

        return X_train, X_test

        
        
        
class CCreateDfRatioAll(MLDatasets):
    def __init__(self, df_all, split_ratio, create_val = True, rs = 10):
        super().__init__(df_all, self.createDfRatioAll, split_ratio, create_val, rs = rs)
        
        
    @staticmethod
    # vytvori dve mnoziny, kde prvni mnozina ma kazdej signal a background podle split ratia a druha zbytek
    def createDfRatioAll(df, split_ratio, rs=10):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()

        for t in df['type'].unique():        
            ts = int(df[df['type']==t].shape[0]*split_ratio)

            s = df[df['type']==t].sample(frac = 1, random_state = rs)


            s_train = s.iloc[:ts,:]
            s_test = s.iloc[ts:,:]

            df_train = pd.concat([df_train, s_train])
            df_test = pd.concat([df_test, s_test])

        df_train = df_train.sample(frac = 1, random_state=rs)   
        df_test = df_test.sample(frac = 1, random_state=rs)   

        return df_train, df_test
        
    
        
class CBalancedDataset(MLDatasets):
    def __init__(self, df_all, total_samples, rs = 10):
        super().__init__(df_all, self.getBalancedDataset, total_samples, rs = rs) 
        
    @staticmethod
    def getBalancedDataset(df, n_samples=14000, rs=10):
        samples = int(n_samples/2)
        
        dfs0 = df[df['y']==0].sample(samples, random_state = rs)        
        dfs1 = df[df['y']==1].sample(samples, random_state = rs)

        df_train = pd.concat([dfs0, dfs1])
        df_train = df_train.sample(df_train.shape[0], random_state = rs)
        
        df_test = df.drop(index = df_train.index)
        
        return df_train, df_test
    
    
class CSampledDataset(MLDatasets):
    def __init__(self, df_all, n_samples, rs = 10):
        super().__init__(df_all, self.sampledDataset, n_samples, rs = rs) 
        
    @staticmethod
    def sampledDataset(df, n_samples=(20000,100000), rs=10):
        samples_s = n_samples[0]
        samples_bg = n_samples[1]
        
        dfs0 = df[df['y']==0].sample(samples_bg, random_state = rs)        
        dfs1 = df[df['y']==1].sample(samples_s, random_state = rs)

        df_train = pd.concat([dfs0, dfs1])
        df_train = df_train.sample(df_train.shape[0], random_state = rs)
        
        df_test = df.drop(index = df_train.index)
        
        return df_train, df_test  
    
    
    
    
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
        super().__init__(df_all, self.sampledDatasetExpEvents, n_samples, rs = rs) 
        
    @staticmethod
    def sampledDatasetExpEvents(df, n_samples=(20000,100000), rs=10):
        samples_s = n_samples[0]
        samples_bg = n_samples[1]
        
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
        
        
    
    
class CVariousBg(MLDatasets):
    def __init__(self, df_all, split_ratio, rs = 10):
        super().__init__(df_all, self.bgVarious, split_ratio, create_val=False, rs = rs)
        
        
    @staticmethod
    # rozdeli dataset na signal a background(oba zamicha), pak obe mnoziny rozdeli podle ratia a nasledne spoji 
    # signaly a backgroundy, obe mnoziny pak opet zamicha
    def bgVarious(df, split_ratio, rs = 10):
        sig = df[df['y'] == 1]
        sig = sig.sample(frac = 1, random_state = rs)
        sig_tr_s = int(sig.shape[0]*split_ratio) 

        bg = df[df['y'] == 0]
        bg = bg.sample(frac = 1, random_state = rs)
        bg_tr_s = int(bg.shape[0]*split_ratio)

        sig_tr = sig.iloc[:sig_tr_s,:]
        sig_te = sig.iloc[sig_tr_s:,:]

        bg_tr = bg.iloc[:bg_tr_s,:]
        bg_te = bg.iloc[bg_tr_s:,:]

        df_train = pd.concat([sig_tr, bg_tr])
        df_test = pd.concat([sig_te, bg_te])

        df_train = df_train.sample(frac = 1, random_state = rs+1)
        df_test = df_test.sample(frac = 1, random_state = rs+1)

        return df_train, df_test

        

        
    
class CRandomSplit(MLDatasets):
    def __init__(self, df_all, split_ratio, rs = 10):
        super().__init__(df_all, self.splitRand, split_ratio, rs = rs)
        
    @staticmethod
    # zamicha dataset a vrati rozdeli na dve mnozina, kde prvni obsahuje split_ratio cast celeho datasetu a druha zbytek

    def splitRand(df, split_ratio = 0.6, rs = 10):
        s = int(df.shape[0]*split_ratio)

        df = df.sample(frac = 1, random_state = rs)
        df_train = df.iloc[:s, :]
        df_test = df.iloc[s:, :]

        return df_train, df_test
        
        
        
        
        
        
        
        
        
        
        
# ---------------------------------- EVALUATION METRICS -----------------------------------
        

    
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

    

    
# --------------------------------- ML MODELS --------------------------------
    
    
class XGBWrapper:
    def __init__(self):
        pass
    
    #def train(self, X_train, y_train, X_val, y_val, n_rounds = 10):
    def train(self, train_m, n_estimators, max_depth, learning_rate):
        params = {}
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'auc'
        params['booster'] = 'gbtree'
        params['max_depth'] = max_depth
        params['random_state'] = 10
        params['eta'] = learning_rate
        
        #evallist = [(dtest, 'eval')]
        num_round = n_estimators
        
        start = time.time()
        model = xgb.train(params,train_m,num_round)
        self.model = model
        end = time.time()
        self.elapsed_time = end - start
        
    def predict(self, X_test):
        p = xgb.DMatrix(X_test)
        return self.model.predict(p)



class SKLearnWrapper:
    def __init__(self, model):
        self.model = model 
        self.elapsed_time = 0
        
    def train(self, X_train, y_train):
        start = time.time()
        self.model.fit(X_train, y_train)
        end = time.time()
        self.elapsed_time = end - start

    def predict(self, X_test):
        return self.model.predict_proba(X_test)[:,1]
    
class SKRandomForest(SKLearnWrapper):
      def __init__(self, n_etimators, max_depth, criterion):
        model = RandomForestClassifier(n_estimators=n_etimators, max_depth=max_depth, criterion = criterion, random_state = 10)
        super().__init__(model)
    
class SKGradientBoosting(SKLearnWrapper):
    def __init__(self, n_etimators, max_depth, lr):
        model = GradientBoostingClassifier(n_estimators=n_etimators, max_depth=max_depth, learning_rate = lr, random_state = 10)
        super().__init__(model)
        
class SKAdaBoost(SKLearnWrapper):
     def __init__(self, n_etimators, max_depth, lr):
        model =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),n_estimators=n_etimators, learning_rate = lr, random_state = 10)
        super().__init__(model)
    
    
    
    
# --------------------------------- MULTICLASSIFIERS -----------------------------

   
class AMulticlassifier:
    def __init__(self, model):        
        self.model = model
    
    def fit(self):
        pass
        
    def convertToCategories(self, types):
        out = []

        for t in types:
            if t == 'tH':
                out.append(0)
            elif t == 'ttb':
                out.append(1)
            elif t == 'ttc':
                out.append(2)
            elif t == 'ttL':
                out.append(3)
            else:
                out.append(4)

        return out
    
    
    def givePredsForClass(self, preds, class_idx):   
        out = []

        for pred in preds:
            out.append(pred[class_idx])

        return out 


    def predict(self, X_test, class_idx = 0):
        preds_proba = np.array(self.model.predict_proba(X_test))
        sig_preds = self.givePredsForClass(preds_proba, class_idx)
        
        return np.array(sig_preds)
    
    
    
class XGBMulticlassifier(AMulticlassifier):
    def __init__(self):
        model = xgb.XGBClassifier(objective = 'multi:prob',
                    n_estimators = 5, 
                    #eval_metric='auc', 
                    #booster = 'gbtree',
                    max_depth = 5,
                    use_label_encoder=False,
                    random_state = 10
                   )
        super().__init__(model)

    def fit(self, X_train, types_train, X_test = None, types_test = None):
        y_train = self.convertToCategories(types_train)
        
        self.model.fit(np.array(X_train), np.array(y_train), verbose=True, 
        eval_set = [(np.array(X_train), np.array(y_train))], #sample_weight = np.array(X_train_weights),
        early_stopping_rounds=5)
    
  
class SKLearnMulticlassifier(AMulticlassifier):
    def __init__(self, model):
        super().__init__(model)

    def fit(self, X_train, types_train):
        y_train = self.convertToCategories(types_train)

        self.model.fit(X_train, y_train)
    

def calcFscore(preds, y_true, threshold):
    predicted = []
    
    for p in preds:
        if p<=threshold:
            predicted.append(0)
        else:
            predicted.append(1)
           
    f1 = f1_score(y_true, predicted)
    print("F1 score:", f1)
        
    return f1
    

    
# odstraneni outlieru
def filterDfZ(df, z, both = False):      
    df1 = df[df['y']==1]
    y1 = df1['y']
    df1 = df1.drop(columns = ['y'])
    df1 = df1[(np.abs(stats.zscore(df1)) < z).all(axis=1)]
    df1['y'] = y1
    
    df0 = df[df['y']==0]
    y0 = df0['y']
    df0 = df0.drop(columns = ['y'])
    df0 = df0[(np.abs(stats.zscore(df0)) < z).all(axis=1)]
    df0['y'] = y0

    df01 = pd.concat([df1, df0])
    
    df01 = df01.sample(frac=1, random_state = rs)
    
    return df01

# odstraneni outlieru graf
def plotSizeDrops(col_start = 0, col_end = 15, par_start = 2, par_end = 6):
    cut = df.iloc[:,col_start:col_end]
    cut['y'] = df['y']
    sig = cut[cut['y']==1]
    scores = np.abs(stats.zscore(sig))
    scores['y'] = sig['y']
    scores = scores[scores['y']==1]

    plt.figure(figsize=(16,10))
    
    legends = []
    for c in scores.drop(columns = ['y']).columns:
        x = []
        y = []

        for t in np.linspace(par_end, par_start, 41):
            x.append(t)
            y.append(scores[scores[c]<t].shape[0])

        plt.plot(x, y)
        plt.xlim(par_end, par_start)
        legends.append(c.replace('_','.'))
    
    plt.legend(legends)
    plt.show()
   




    
    
    
    
    