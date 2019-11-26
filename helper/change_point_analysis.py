import numpy as np
class ChangePointAnalysis:
    def find_change_point(self, data):
        raise NotImplementedError
    
    def calculate_position(self,teststat,criterion,tau):
        if teststat > criterion:
            return tau
        else:
            return None

class CPA_Mean(ChangePointAnalysis):
    def find_change_point(self, data):
        n = len(data)
        tau = np.arange(1,n)
        lmbd = 2*np.log(n) #Bayesian Information Criterion
        eps = 1.e-8 #to avoid zeros in denominator
        mu0 = np.mean(data)
        s0 = np.sum((data-mu0)**2)
        s1 = np.asarray([np.sum((data[0:i]-np.mean(data[0:i]))**2) for i in range(1,n)])
        s2 = np.asarray([np.sum((data[i:]-np.mean(data[i:]))**2) for i in range(1,n)])
        R  = s0-s1-s2
        G  = np.max(R)
        taustar = int(np.where(R==G)[0]) + 1
        sd1 = np.std(data[0:taustar-1])
        sd2 = np.std(data[taustar-1:])
        #use pooled standard deviation
        var = ( taustar*sd1**2 + (n-taustar)*sd2**2 ) / n
        return self.calculate_position(2*G,var*lmbd,taustar)

class CPA_Variance(ChangePointAnalysis):
    def find_change_point(self, data):
        n = len(data)
        tau = np.arange(1,n)
        lmbd = 2*np.log(n) #Bayesian Information Criterion
        eps = 1.e-8 #to avoid zeros in denominator
        std0 = np.std(data)
        std1 = np.asarray([np.std(data[0:i]) for i in range(1,n)],dtype=float) + eps
        std2 = np.asarray([np.std(data[i:]) for i in range(1,n)],dtype=float) + eps
        R = n*np.log(std0) - tau*np.log(std1) - (n-tau)*np.log(std2)
        G  = np.max(R)
        taustar = int(np.where(R==G)[0]) + 1
        return self.calculate_position(2*G,lmbd,taustar)

class CPA_BernoulliMean(ChangePointAnalysis):
    def find_change_point(self, data):
        data = (data-min(data))/(max(data)-min(data))

        n = len(data)
        tau = np.arange(1,n)
        lmbd = 2*np.log(n) #Bayesian Information Criterion
        eps = 1.e-8 #to avoid zeros in denominator
        m0 = np.sum(data)
        m1 = np.cumsum(data)[:-1]
        m2 = m0 - m1
        p0 = np.mean(data)
        p1 = m1 / tau
        p2 = m2 / (n-tau+1)

        #take care of possible NaN
        p1 = p1 + eps
        p1[p1>1] = 1. - eps
        p2 = p2 + eps #to avoid zero
        p2[p2>1] = 1. - eps

        R  = m1*np.log(p1) + (tau-m1)*np.log(1-p1) + m2*np.log(p2) + (n-tau-m2)*np.log(1-p2) - m0*np.log(p0) - (n-m0)*np.log(1-p0)
        G  = np.max(R)
        taustar = int(np.where(R==G)[0]) + 1


        return self.calculate_position(2*G,lmbd,taustar)

class CPA_PoissonMean(ChangePointAnalysis):
    def find_change_point(self, data):
        n = len(data)
        lmbd = 2*np.log(n) #Bayesian Information Criterion
        eps = 1.e-8 #to avoid zeros in denominator
        lambda0 = np.mean(data)
        lambda1 = np.asarray([np.mean(data[0:i]) for i in range(1,n)],dtype=float) + eps
        lambda2 = np.asarray([np.mean(data[i:]) for i in range(1,n)],dtype=float) + eps
        m0 = np.sum(data)
        m1 = np.cumsum(data)[:-1]
        m2 = m0 - m1
        R  = m1*np.log(lambda1) + m2*np.log(lambda2) - m0*np.log(lambda0)
        G  = np.max(R)
        taustar = int(np.where(R==G)[0]) + 1
        return self.calculate_position(2*G,lmbd,taustar)